# mocap — 动捕与手部数据采集(合并主项目)

以 mocap 为主文件夹合并了原采集链路的四个独立项目(2026-08-07):

```
mocap/
├── README.md            # 本文件(合并总览)
├── pixi.toml            # 动捕消费/可视化(原 mocap 根)
├── src/mocap_viewer/    #   Viser viewer(mocap/hands/frame → 网页)
├── scripts/             #   zenoh_consumer / demo_publisher
├── CONSUMING.md         #   消费指南(帧格式/relay/router 模式)
├── net.md               #   网络拓扑记录(Windows ↔ Ubuntu)
├── manus/               # ← 原 /home/current/syz/manus
│   ├── rawviz.cpp/out   #   Manus 手套 dongle 读取(C++,Integrated 模式)
│   ├── zenoh_pub.py     #   骨架 → Zenoh(manus/raw_skeleton/*,~30Hz)
│   ├── zenoh_sub.py     #   订阅验证
│   ├── manus_pub.py     #   骨架 → ROS2 Jazzy(rmw_zenoh)
│   ├── viz.py           #   viser 骨架可视化
│   └── .pixi/           #   python 3.14 环境(含 zenohd Router 二进制!)
├── natnet/              # ← 原 /home/current/syz/natnet(git 仓库,Windows 桥)
│   ├── src/natnet_zenoh/#   NatNet → Zenoh 发布/订阅库(schema/transport)
│   ├── Samples/PythonClient/  # 官方 NatNet SDK 4.4
│   └── .pixi/           #   Windows publish / Ubuntu subscribe 双 feature
└── acquisition/         # ← 原 /home/current/syz/acquisition
    ├── src/acquisition/ #   采集主程序(拼接/录制/可视化/按键)
    ├── scripts/         #   start_zenohd / e2e_check / inspect / replay / demo_manus
    ├── config.yaml      #   刚体映射/offset/轴变换配置
    └── .pixi/           #   python 3.12 环境
```

## 依赖关系(严格 DAG)

```
manus(手套数据源) ──zenohd 二进制──┐
natnet(动捕桥,叶子) ──editable path─> mocap(消费/可视化) ──demo task──> acquisition
natnet ──editable path───────────────────────────────────────────────> acquisition
```

四个项目各自独立 pixi 环境(**python 版本冲突,不可统一**:manus 需 3.14、
natnet/acquisition/mocap 用 3.12)。环境间通过 Zenoh Router 与 editable path 依赖互联。

## 启动顺序

```bash
# 0. Zenoh Router(常驻)
bash acquisition/scripts/start_zenohd.sh            # 或 systemd 服务
# 1. 数据源(真实设备)
./manus/rawviz.out | manus/zenoh_pub.py --router tcp/localhost:7447   # 手套(30Hz)
#   Windows: natnet publisher(参数不变,连 169.254.1.0:7447)            # 动捕(120Hz)
# 2. 可视化 / 消费
pixi run view -- --connect-endpoint tcp/127.0.0.1:7447 --no-relay      # 动捕 viewer
# 3. 数据采集(拼接 + 按键录制)
cd acquisition && pixi run record                   # http://127.0.0.1:8081
```

常用命令(`cd` 到对应子目录后):

| 位置 | 命令 | 作用 |
|---|---|---|
| `mocap/` | `pixi run view / subscribe / demo-publish / test` | 动捕可视化/消费/合成数据/测试 |
| `mocap/acquisition/` | `pixi run record / demo-manus / demo-mocap / inspect / replay / test` | 采集/合成/检查/回放/测试 |
| `mocap/acquisition/` | `pixi run python scripts/e2e_check.py --seconds 5` | 端到端验证 |

## 关键说明

- **zenohd Router** 二进制位于 `manus/.pixi/`(acquisition 脚本按相对位置推导,
  移动整个 mocap 目录无需改脚本)
- **natnet 是 git 仓库**(remote: `git@github.com:suzhelearning/natnet.git`),
  移动不影响 git;Windows 侧发布端不受目录结构影响(仅网络连接)
- **路径约定**:项目间依赖一律用相对路径(`natnet`、`../natnet`、`../../manus`),
  不硬编码绝对路径;新增跨项目引用请沿用
- 其他项目(wuji-*/moshpp/MANUS_Core_3.0.1_SDK)保持在 `/home/current/syz/` 根,不在本合并范围

## 测试

```bash
pixi run test                      # mocap: 29 项
cd acquisition && pixi run test    # acquisition: 47 项(含真实 zenohd 集成)
```
