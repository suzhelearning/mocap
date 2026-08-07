# 网络拓扑记录

Motive → NatNet → Zenoh 数据链路中两台主机的网络信息。

## 主机一览

| 主机 | 角色 | 有线网卡 | 有线 link-local IP | WiFi IP |
|---|---|---|---|---|
| **Ubuntu**（本机） | Zenoh 数据接收端（subscriber） | `enp127s0` | `169.254.1.0/16` | `192.168.108.244/22`（wlp128s0） |
| **Windows** `DESKTOP-48V8LO8` | Motive + Zenoh publisher | 以太网（索引 18） | `169.254.213.247/16` | `192.168.110.6/22` |

## 网段说明

- **数据通道（有线直连）**：link-local `169.254.0.0/16`，APIPA 自动配置，不依赖 DHCP/路由器。
- **管理通道（WiFi）**：`192.168.108.0/22`，走路由器 `192.168.110.1`，SSH 使用。
- ⚠️ Windows 只对 `169.254.0.0/16` 有回包路由，**Ubuntu 必须用 169.254 段地址作源**。

## SSH（管理通道，走 WiFi）

```bash
ssh -i ~/.ssh/id_ed25519_windows -l 'current robotics' 192.168.110.6
```

## Zenoh 数据通道（有线）

- Ubuntu subscriber 监听：`tcp/0.0.0.0:7447`（所有接口）
- Windows publisher 连接：`--zenoh-endpoint tcp/169.254.1.0:7447`（**必须用 169.254 段地址**）
- key：`mocap/hands/frame`；顺序：先起 Ubuntu subscriber，再起 Windows publisher
- 有线实测 ~0.3ms、0% 丢包（2026-08-07）

## Motive 侧（Windows）

- Motive 3.4 Tracker，Data Streaming：Broadcast Frame Data 开启、Unicast、Local Interface `127.0.0.1`（publisher 同机 loopback）
- publisher 参数：`--motive-server-ip 127.0.0.1 --natnet-client-ip 127.0.0.1`
- Windows 工作空间：`C:\Users\current robotics\Desktop\syz\NatNetSDK`

## 防火墙

- Windows：无自定义规则（临时诊断规则已删）。
- Ubuntu：若开 ufw，需 `sudo ufw allow from 169.254.213.247 to any port 7447 proto tcp`。
