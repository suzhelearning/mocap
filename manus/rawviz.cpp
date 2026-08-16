// rawviz.cpp — Manus Metaglove Pro raw 骨架 25 关键点采集器
// 以 Integrated 模式连接 dongle，把每帧骨架节点的局部位姿与源时间输出到 stdout。
//
// 输出协议（一行一条，供 viz.py/zenoh_pub.py 解析）：
//   HAND <gloveId hex8> <Left|Right> <nodeCount>
//   EDGE <gloveId hex8> <childNodeId> <parentNodeId> <chainType>
//   POSE <gloveId> <seq> <source_monotonic_ns> <sdk_publish_time>
//        <x0> <y0> <z0> <qw0> <qx0> <qy0> <qz0> ...
//        (每手套独立 seq；source_monotonic_ns 在 SDK 回调入口采集)
//
// 编译（系统工具链）：
//   /usr/bin/g++ -std=c++17 -O2 -pthread -I<SDK>/ManusSDK/include rawviz.cpp \
//       -L<SDK>/ManusSDK/lib -l:libManusSDK_Integrated.so \
//       /usr/lib/x86_64-linux-gnu/libudev.so.1 \
//       /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 \
//       /usr/lib/x86_64-linux-gnu/libz.so.1 \
//       -Wl,-rpath,<SDK>/ManusSDK/lib -o rawviz.out

#include <cstdio>
#include <cstring>
#include <csignal>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <map>
#include <unistd.h>
#include <limits.h>

#include "ManusSDK.h"

// 节点数上限:正常 25 节点/手套;恶意/损坏 dongle 上报超大计数时拒绝分配
static const uint32_t kMaxNodesPerGlove = 64;
// 校准文件大小上限(正常 ~几 KB;防被替换为超大文件触发大分配)
static const long kMaxCalibrationBytes = 4 * 1024 * 1024;

// 校准用户:--user <名字> 指定后使用 calibration/<名字>Left/RightMetaglovePro.mcal;
// 为空则使用无前缀的 Left/RightMetaglovePro.mcal(旧行为,文件缺失仅警告)
static std::string g_CalibrationUser;

// 构造校准文件路径:<exe目录>/calibration/[<user>]<Base>,写入 out
static void BuildCalibrationPath(const char* t_Base, char* t_Out, size_t t_OutSize)
{
	char t_Exe[PATH_MAX];
	ssize_t t_Len = readlink("/proc/self/exe", t_Exe, sizeof(t_Exe) - 1);
	if (t_Len > 0)
	{
		t_Exe[t_Len] = '\0';
		char* t_Slash = strrchr(t_Exe, '/');
		if (t_Slash && t_Slash != t_Exe)
		{
			*t_Slash = '\0';
			snprintf(t_Out, t_OutSize, "%s/calibration/%s%s",
			         t_Exe, g_CalibrationUser.c_str(), t_Base);
			return;
		}
	}
	snprintf(t_Out, t_OutSize, "calibration/%s%s",
	         g_CalibrationUser.c_str(), t_Base);
}

// ---------------------------------------------------------------------------
// 最新帧缓存（SDK 回调线程写入，主循环读取）——按手套独立缓存,
// 每手套自己的 seq(左右手解耦:一只遮挡不影响另一只)
// ---------------------------------------------------------------------------
struct NodePose
{
	float x, y, z;
	float qw, qx, qy, qz;
};
struct GloveStream
{
	uint64_t seq = 0;
	uint64_t sourceMonotonicNs = 0;
	uint64_t sdkPublishTime = 0;
	std::vector<NodePose> nodes;
};

static std::mutex g_FrameMutex;
static std::map<uint32_t, GloveStream> g_LatestFrames;   // gloveId -> 最新帧
static std::atomic<bool> g_Running{ true };

// 拓扑信息缓存：gloveId -> (nodeId, parentId) 列表（与节点数组同序）
static std::mutex g_TopoMutex;
static std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> g_Topology;

// ---------------------------------------------------------------------------
// 校准文件加载（仿 wuji-hand-teleop：CoreSdk_SetGloveCalibration）
// 校准显著改善骨架质量；文件随项目分发（manus/calibration/*.mcal）
// ---------------------------------------------------------------------------
static void LoadCalibration(uint32_t p_GloveId, uint32_t p_Side)
{
	const char* t_Base = (p_Side == Side_Left) ? "LeftMetaglovePro.mcal"
	                                           : "RightMetaglovePro.mcal";
	// 路径 = <exe目录>/calibration[/<user>]/<文件名>,不依赖启动时的 CWD
	char t_Path[PATH_MAX * 2];
	BuildCalibrationPath(t_Base, t_Path, sizeof(t_Path));

	FILE* t_File = fopen(t_Path, "rb");
	if (!t_File)
	{
		fprintf(stderr, "[rawviz] 校准文件不存在: %s(跳过)\n", t_Path);
		return;
	}
	fseek(t_File, 0, SEEK_END);
	long t_Length = ftell(t_File);
	fseek(t_File, 0, SEEK_SET);
	if (t_Length <= 0)
	{
		fprintf(stderr, "[rawviz] 校准文件为空: %s\n", t_Path);
		fclose(t_File);
		return;
	}
	if (t_Length > kMaxCalibrationBytes)
	{
		fprintf(stderr, "[rawviz] 校准文件过大(%ld B,上限 %ld B),跳过: %s\n",
		        t_Length, kMaxCalibrationBytes, t_Path);
		fclose(t_File);
		return;
	}
	std::vector<unsigned char> t_Data(t_Length);
	if (fread(t_Data.data(), 1, t_Length, t_File) != (size_t)t_Length)
	{
		fprintf(stderr, "[rawviz] 校准文件读取失败: %s\n", t_Path);
		fclose(t_File);
		return;
	}
	fclose(t_File);

	SetGloveCalibrationReturnCode t_Result;
	// API 返回 SDKReturnCode(调用是否成功),实际结果在 t_Result 中
	if (CoreSdk_SetGloveCalibration(p_GloveId, t_Data.data(), (int)t_Length, &t_Result)
	    == SDKReturnCode_Success && t_Result == SetGloveCalibrationReturnCode_Success)
	{
		fprintf(stderr, "[rawviz] 校准已加载: %s (glove %08X)\n", t_Path, p_GloveId);
	}
	else
	{
		fprintf(stderr, "[rawviz] 校准加载失败 %s: %d\n", t_Path, (int)t_Result);
	}
}

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
	const uint64_t t_SourceMonotonicNs = (uint64_t)std::chrono::duration_cast<
		std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count();

	std::lock_guard<std::mutex> t_Lock(g_FrameMutex);
	for (uint32_t i = 0; i < p_Info->skeletonsCount; i++)
	{
		RawSkeletonInfo t_SkelInfo{};
		if (CoreSdk_GetRawSkeletonInfo(i, &t_SkelInfo) != SDKReturnCode_Success)
			continue;

		GloveStream& t_Stream = g_LatestFrames[t_SkelInfo.gloveId];
		t_Stream.seq++;                       // 每手套独立自增
		t_Stream.sourceMonotonicNs = t_SourceMonotonicNs;
		t_Stream.sdkPublishTime = t_SkelInfo.publishTime.time;
		t_Stream.nodes.clear();

		if (t_SkelInfo.nodesCount > 0)
		{
			if (t_SkelInfo.nodesCount > kMaxNodesPerGlove)
			{
				fprintf(stderr, "[rawviz] 跳过异常节点数 %u (glove %08X)\n",
				        (unsigned)t_SkelInfo.nodesCount, (unsigned)t_SkelInfo.gloveId);
				continue;
			}
			std::vector<SkeletonNode> t_Nodes(t_SkelInfo.nodesCount);
			if (CoreSdk_GetRawSkeletonData(i, t_Nodes.data(), t_SkelInfo.nodesCount) == SDKReturnCode_Success)
			{
				t_Stream.nodes.reserve(t_SkelInfo.nodesCount);
				for (uint32_t n = 0; n < t_SkelInfo.nodesCount; n++)
				{
					const ManusTransform& t_Transform = t_Nodes[n].transform;
					t_Stream.nodes.push_back({
						t_Transform.position.x, t_Transform.position.y, t_Transform.position.z,
						t_Transform.rotation.w, t_Transform.rotation.x,
						t_Transform.rotation.y, t_Transform.rotation.z
					});
				}
			}
		}
	}
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
	if (CoreSdk_GetRawSkeletonNodeCount(p_GloveId, t_NodeCount) != SDKReturnCode_Success
	    || t_NodeCount == 0 || t_NodeCount > kMaxNodesPerGlove)
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

	// 首次见手套:加载该侧校准文件(质量提升,失败不阻塞)
	if (t_Side == Side_Left || t_Side == Side_Right)
		LoadCalibration(p_GloveId, t_Side);

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
				{
					// 原始骨架模式(不做手势估计),与 wuji-hand-teleop 一致
					CoreSdk_SetRawSkeletonHandMotion(HandMotion_None);
					return true;
				}
				fprintf(stderr, "[rawviz] ConnectToHost failed: %d\n", (int)t_Conn);
			}
		}
		fprintf(stderr, "[rawviz] waiting for dongle...\n");
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	return false;
}

// ---------------------------------------------------------------------------
// 启动前校验:显式 --user 时校准文件必须存在,缺失报错退出
// ---------------------------------------------------------------------------
static bool CheckCalibrationFilesExist()
{
	if (g_CalibrationUser.empty())
		return true;                    // 未指定用户:走旧路径,加载失败只警告
	bool t_Ok = true;
	for (const char* t_Name : {"LeftMetaglovePro.mcal", "RightMetaglovePro.mcal"})
	{
		char t_Path[PATH_MAX * 2];
		BuildCalibrationPath(t_Name, t_Path, sizeof(t_Path));
		FILE* t_F = fopen(t_Path, "rb");
		if (!t_F)
		{
			fprintf(stderr, "[rawviz] 错误: 用户 '%s' 的校准文件不存在: %s\n",
			        g_CalibrationUser.c_str(), t_Path);
			t_Ok = false;
		}
		else
		{
			fclose(t_F);
		}
	}
	return t_Ok;
}

// ---------------------------------------------------------------------------
// 主循环：约 120fps 读最新帧并输出（SDK 回调可达 120Hz,输出不得再压到 30fps）
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--user") == 0 && i + 1 < argc)
		{
			g_CalibrationUser = argv[++i];
		}
		else
		{
			fprintf(stderr, "[rawviz] 未知参数: %s\n", argv[i]);
			fprintf(stderr, "用法: %s [--user <用户名>]  (用 calibration/<用户名>Left/RightMetaglovePro.mcal)\n",
			        argv[0]);
			return 2;
		}
	}
	if (!CheckCalibrationFilesExist())
	{
		fprintf(stderr, "[rawviz] 校准文件缺失,退出。请先为用户 '%s' 准备 "
		        "manus/calibration/%sLeftMetaglovePro.mcal 与 %sRightMetaglovePro.mcal\n",
		        g_CalibrationUser.c_str(),
		        g_CalibrationUser.c_str(), g_CalibrationUser.c_str());
		return 2;
	}

	signal(SIGINT, HandleSignal);
	signal(SIGTERM, HandleSignal);

	if (!ConnectIntegrated())
	{
		fprintf(stderr, "[rawviz] failed to connect to dongle\n");
		return 1;
	}
	fprintf(stderr, "[rawviz] connected, streaming raw skeleton...\n");

	std::map<uint32_t, uint64_t> t_LastSeq;     // 每手套已输出序号
	while (g_Running)
	{
		// 2ms 轮询:SDK 回调 ~102Hz,8ms 轮询 + printf/fflush 开销会丢帧(实测 57%)
		std::this_thread::sleep_for(std::chrono::milliseconds(2));

		std::map<uint32_t, GloveStream> t_Latest;
		{
			std::lock_guard<std::mutex> t_Lock(g_FrameMutex);
			t_Latest = g_LatestFrames;
		}

		// 每手套独立检查新帧并输出(左右手解耦)
		for (const auto& t_Entry : t_Latest)
		{
			const uint32_t t_GloveId = t_Entry.first;
			const GloveStream& t_Stream = t_Entry.second;
			if (t_Stream.seq == t_LastSeq[t_GloveId] || t_Stream.nodes.empty())
				continue;
			t_LastSeq[t_GloveId] = t_Stream.seq;

			EnsureTopology(t_GloveId);
			printf("POSE %08X %llu %llu %llu", t_GloveId,
			       (unsigned long long)t_Stream.seq,
			       (unsigned long long)t_Stream.sourceMonotonicNs,
			       (unsigned long long)t_Stream.sdkPublishTime);
			for (const NodePose& t_N : t_Stream.nodes)
				printf(" %.6f %.6f %.6f %.7f %.7f %.7f %.7f",
				       t_N.x, t_N.y, t_N.z, t_N.qw, t_N.qx, t_N.qy, t_N.qz);
			printf("\n");
		}
		fflush(stdout);
	}

	CoreSdk_ShutDown();
	return 0;
}
