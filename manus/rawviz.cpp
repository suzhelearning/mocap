// rawviz.cpp — Manus Metaglove Pro raw 骨架 25 关键点采集器
// 以 Integrated 模式连接 dongle，把每帧骨架节点位置输出到 stdout。
//
// 输出协议（一行一条，供 viz.py 解析）：
//   HAND <gloveId hex8> <Left|Right> <nodeCount>
//   EDGE <gloveId hex8> <childNodeId> <parentNodeId> <chainType>
//   FRAME <seq>
//   POS  <gloveId hex8> <x0> <y0> <z0> <x1> <y1> <z1> ...   (nodeCount 个节点，按数组顺序)
//
// 编译（pixi 工具链）：
//   g++ -std=c++17 -I<SDK>/ManusSDK/include rawviz.cpp \
//       -L<SDK>/ManusSDK/lib -l:libManusSDK_Integrated.so \
//       /lib/x86_64-linux-gnu/libudev.so.1 /lib/x86_64-linux-gnu/libusb-1.0.so.0 \
//       -Wl,-rpath,<SDK>/ManusSDK/lib -o rawviz.out

#include <cstdio>
#include <cstring>
#include <csignal>
#include <mutex>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <map>

#include "ManusSDK.h"

// ---------------------------------------------------------------------------
// 最新帧缓存（SDK 回调线程写入，主循环读取）
// ---------------------------------------------------------------------------
struct NodePose { float x, y, z; };
struct SkeletonPose
{
	uint32_t gloveId = 0;
	uint32_t side = Side_Invalid;
	std::vector<NodePose> nodes;
};
struct RawFrame
{
	uint64_t seq = 0;
	std::vector<SkeletonPose> skeletons;
};

static std::mutex g_FrameMutex;
static RawFrame g_LatestFrame;
static std::atomic<bool> g_Running{ true };

// 拓扑信息缓存：gloveId -> (nodeId, parentId) 列表（与节点数组同序）
static std::mutex g_TopoMutex;
static std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> g_Topology;

// ---------------------------------------------------------------------------
// 信号处理：干净退出（先关 SDK 再退出，避免 dongle 停在异常状态）
// ---------------------------------------------------------------------------
static void HandleSignal(int)
{
	g_Running = false;
}

// ---------------------------------------------------------------------------
// Raw skeleton 流回调：把最新帧拷进缓存
// ---------------------------------------------------------------------------
static void OnRawSkeletonStream(const SkeletonStreamInfo* const p_Info)
{
	if (!p_Info) return;

	RawFrame t_Frame;
	t_Frame.seq = ++g_LatestFrame.seq;

	t_Frame.skeletons.resize(p_Info->skeletonsCount);
	for (uint32_t i = 0; i < p_Info->skeletonsCount; i++)
	{
		RawSkeletonInfo t_SkelInfo{};
		if (CoreSdk_GetRawSkeletonInfo(i, &t_SkelInfo) != SDKReturnCode_Success)
			continue;

		SkeletonPose& t_Skel = t_Frame.skeletons[i];
		t_Skel.gloveId = t_SkelInfo.gloveId;
		t_Skel.nodes.resize(t_SkelInfo.nodesCount);

		if (t_SkelInfo.nodesCount > 0)
		{
			std::vector<SkeletonNode> t_Nodes(t_SkelInfo.nodesCount);
			if (CoreSdk_GetRawSkeletonData(i, t_Nodes.data(), t_SkelInfo.nodesCount) == SDKReturnCode_Success)
			{
				for (uint32_t n = 0; n < t_SkelInfo.nodesCount; n++)
				{
					t_Skel.nodes[n] = { t_Nodes[n].transform.position.x,
					                    t_Nodes[n].transform.position.y,
					                    t_Nodes[n].transform.position.z };
				}
			}
		}
	}

	std::lock_guard<std::mutex> t_Lock(g_FrameMutex);
	g_LatestFrame = std::move(t_Frame);
}

// ---------------------------------------------------------------------------
// SDK 日志改道到 stderr（stdout 只留给协议数据）
// ---------------------------------------------------------------------------
static void OnSdkLog(LogSeverity, const char* const p_Log, uint32_t p_Length)
{
	fprintf(stderr, "%.*s\n", (int)p_Length, p_Log ? p_Log : "");
	fflush(stderr);
}

// ---------------------------------------------------------------------------
// 首次见到某只手套时，获取节点层级（父子关系）并输出
// ---------------------------------------------------------------------------
static void EnsureTopology(uint32_t p_GloveId)
{
	std::lock_guard<std::mutex> t_Lock(g_TopoMutex);
	if (g_Topology.count(p_GloveId) > 0)
		return;

	uint32_t t_NodeCount = 0;
	if (CoreSdk_GetRawSkeletonNodeCount(p_GloveId, t_NodeCount) != SDKReturnCode_Success || t_NodeCount == 0)
		return;

	std::vector<NodeInfo> t_NodeInfo(t_NodeCount);
	if (CoreSdk_GetRawSkeletonNodeInfoArray(p_GloveId, t_NodeInfo.data(), t_NodeCount) != SDKReturnCode_Success)
		return;

	// RawSkeletonInfo 没有 side 字段，从 NodeInfo 取第一个有效的 side
	uint32_t t_Side = Side_Invalid;
	for (uint32_t i = 0; i < t_NodeCount; i++)
	{
		if (t_NodeInfo[i].side != Side_Invalid)
		{
			t_Side = t_NodeInfo[i].side;
			break;
		}
	}

	printf("HAND %08X %s %u\n", p_GloveId,
	       t_Side == Side_Left ? "Left" : (t_Side == Side_Right ? "Right" : "Unknown"),
	       t_NodeCount);
	fflush(stdout);

	std::vector<std::pair<uint32_t, uint32_t>> t_Topo;
	for (uint32_t i = 0; i < t_NodeCount; i++)
	{
		t_Topo.emplace_back(t_NodeInfo[i].nodeId, t_NodeInfo[i].parentId);
		if (t_NodeInfo[i].parentId != 0 && t_NodeInfo[i].parentId != t_NodeInfo[i].nodeId)
		{
			printf("EDGE %08X %u %u %u\n", p_GloveId,
			       t_NodeInfo[i].nodeId, t_NodeInfo[i].parentId,
			       (uint32_t)t_NodeInfo[i].chainType);
		}
	}
	fflush(stdout);

	g_Topology[p_GloveId] = std::move(t_Topo);
}

// ---------------------------------------------------------------------------
// 连接流程（Integrated 模式）
// ---------------------------------------------------------------------------
static bool ConnectIntegrated()
{
	// 等待设备就绪：先初始化 SDK
	SDKReturnCode t_Result = CoreSdk_InitializeIntegrated();
	if (t_Result != SDKReturnCode_Success)
	{
		fprintf(stderr, "[rawviz] InitializeIntegrated failed: %d\n", (int)t_Result);
		return false;
	}

	// 坐标系：z-up、右手系、x 朝观察者、单位米、世界坐标（与官方示例一致）
	CoordinateSystemVUH t_VUH;
	CoordinateSystemVUH_Init(&t_VUH);
	t_VUH.handedness = Side_Right;
	t_VUH.up = AxisPolarity_PositiveZ;
	t_VUH.view = AxisView_XFromViewer;
	t_VUH.unitScale = 1.0f;

	if (CoreSdk_InitializeCoordinateSystemWithVUH(t_VUH, true) != SDKReturnCode_Success)
	{
		fprintf(stderr, "[rawviz] Coordinate system init failed\n");
		return false;
	}

	// 注册回调
	if (CoreSdk_RegisterCallbackForOnLog(OnSdkLog) != SDKReturnCode_Success)
	{
		fprintf(stderr, "[rawviz] Register log callback failed\n");
		return false;
	}
	if (CoreSdk_RegisterCallbackForRawSkeletonStream(OnRawSkeletonStream) != SDKReturnCode_Success)
	{
		fprintf(stderr, "[rawviz] Register callback failed\n");
		return false;
	}

	// 找主机并连接（Integrated 模式下 LookForHosts 找本地 Core）
	for (int t_Try = 0; t_Try < 10 && g_Running; t_Try++)
	{
		CoreSdk_LookForHosts(1, true);

		uint32_t t_NumHosts = 0;
		if (CoreSdk_GetNumberOfAvailableHostsFound(&t_NumHosts) == SDKReturnCode_Success && t_NumHosts > 0)
		{
			std::vector<ManusHost> t_Hosts(t_NumHosts);
			if (CoreSdk_GetAvailableHostsFound(t_Hosts.data(), t_NumHosts) == SDKReturnCode_Success)
			{
				SDKReturnCode t_Conn = CoreSdk_ConnectToHost(t_Hosts[0]);
				if (t_Conn == SDKReturnCode_Success)
					return true;
				fprintf(stderr, "[rawviz] ConnectToHost failed: %d\n", (int)t_Conn);
			}
		}
		fprintf(stderr, "[rawviz] waiting for dongle...\n");
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	return false;
}

// ---------------------------------------------------------------------------
// 主循环：约 30fps 读最新帧并输出
// ---------------------------------------------------------------------------
int main()
{
	signal(SIGINT, HandleSignal);
	signal(SIGTERM, HandleSignal);

	if (!ConnectIntegrated())
	{
		fprintf(stderr, "[rawviz] failed to connect to dongle\n");
		return 1;
	}
	fprintf(stderr, "[rawviz] connected, streaming raw skeleton...\n");

	uint64_t t_LastSeq = 0;
	while (g_Running)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(33));

		RawFrame t_Frame;
		{
			std::lock_guard<std::mutex> t_Lock(g_FrameMutex);
			if (g_LatestFrame.seq == t_LastSeq)
				continue;
			t_Frame = g_LatestFrame;
		}
		t_LastSeq = t_Frame.seq;

		printf("FRAME %llu\n", (unsigned long long)t_Frame.seq);
		for (const SkeletonPose& t_Skel : t_Frame.skeletons)
		{
			if (t_Skel.nodes.empty())
				continue;
			EnsureTopology(t_Skel.gloveId);
			printf("POS %08X", t_Skel.gloveId);
			for (const NodePose& t_N : t_Skel.nodes)
				printf(" %.6f %.6f %.6f", t_N.x, t_N.y, t_N.z);
			printf("\n");
		}
		fflush(stdout);
	}

	CoreSdk_ShutDown();
	return 0;
}
