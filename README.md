# mocap — 动捕与手部数据采集(合并主项目)

以 mocap 为主文件夹合并了原采集链路的四个独立项目(2026-08-07):

```
mocap/
├── README.md            # 本文件(合并总览)
├── pixi.toml            # 动捕消费/可视化(原 mocap 根)
├── src/mocap_viewer/    #   Viser viewer(mocap/hands/frame → 网页)
├── scripts/             #   zenoh_consumer
├── CONSUMING.md         #   消费指南(帧格式/relay/router 模式)
├── net.md               #   网络拓扑记录(Windows ↔ Ubuntu)
├── manus/               # ← 原 /home/current/syz/manus
│   ├── rawviz.cpp/out   #   Manus 手套 dongle 读取(C++,Integrated 模式)
│   ├── zenoh_pub.py     #   骨架 → Zenoh(manus/raw_skeleton/*,120Hz)
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
    ├── scripts/         #   start_zenohd / e2e_check / inspect / replay
    ├── config.yaml      #   刚体映射/offset/轴变换配置
    └── .pixi/           #   python 3.12 环境
```

## 依赖关系(严格 DAG)

```
manus(手套数据源) ──zenohd 二进制──┐
natnet(动捕桥,叶子) ──editable path─> mocap(消费/可视化) ──> acquisition
natnet ──editable path──────────────────────────────────────> acquisition
```

四个项目各自独立 pixi 环境(**python 版本冲突,不可统一**:manus 需 3.14、
natnet/acquisition/mocap 用 3.12)。环境间通过 Zenoh Router 与 editable path 依赖互联。

## 启动顺序(三条独立路线)

```bash
# 0. Zenoh Router(常驻,三条路线共同依赖)
bash acquisition/scripts/start_zenohd.sh            # 或 systemd 服务

# 路线 1/3:Manus 手套发布(三个终端并行)
bash manus_pub.sh                   # 真实手套 120Hz
# 路线 2/3:Windows rigid 发布(Motive 动捕,ssh 挂载)
bash windows_pub.sh                 # Ctrl-C 断开自动清理
# 路线 3/3:数据 web 订阅与收集
bash record.sh                      # http://127.0.0.1:8081
```

三条路线**独立启动/停止,任意顺序**(router 在即可)。无数据流入时采集程序
会提示「⏳ 等待设备启动」。各脚本说明:

| 脚本 | 作用 | 退出行为 |
|---|---|---|
| `manus_pub.sh` | 手套发布(rawviz \| zenoh_pub,120Hz);`--user <名>` 选人校准(缺失报错) | Ctrl-C 即停;dongle 独占 |
| `windows_pub.sh` | Windows motive 发布(ssh 会话挂载) | 断开 ssh + 主动停 Windows 进程 |
| `record.sh` | 采集程序(web 8081 + 录制按钮 + 采集频率) | q / Ctrl-C,录制中先丢弃 |

web 左侧「录制控制」按钮与键盘(r/s/d)等价;「采集频率」下拉默认
100Hz 可运行时调。

常用命令(`cd` 到对应子目录后):

| 位置 | 命令 | 作用 |
|---|---|---|
| `mocap/` | `pixi run view / subscribe / test` | 动捕可视化/消费/测试 |
| `mocap/acquisition/` | `pixi run record / inspect / replay / test` | 采集/检查/回放/测试 |
| `mocap/acquisition/` | `pixi run python scripts/e2e_check.py --seconds 5` | 端到端验证(需真实流) |

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
