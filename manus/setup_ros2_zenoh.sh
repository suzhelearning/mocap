#!/usr/bin/env bash
# 安装 ROS2 Jazzy + rmw_zenoh（需要 sudo）
# 用法: sudo bash setup_ros2_zenoh.sh
set -euo pipefail

echo "==> [1/4] 添加 ROS2 官方软件源 (Ubuntu 24.04 noble / Jazzy)"
apt-get update
apt-get install -y software-properties-common curl
add-apt-repository -y universe || true
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2.list
apt-get update

echo "==> [2/4] 安装 ROS2 Jazzy (ros-base，含 rclpy)"
apt-get install -y ros-jazzy-ros-base

echo "==> [3/4] 安装 rmw_zenoh（ROS2 的 Zenoh 传输实现）"
apt-get install -y ros-jazzy-rmw-zenoh

echo "==> [4/4] 安装 ROS2 CLI 工具（ros2 topic echo 等，可选但推荐）"
apt-get install -y ros-jazzy-ros2cli ros-jazzy-ros2topic || true

echo ""
echo "✅ 安装完成。每次使用前执行:"
echo "    source /opt/ros/jazzy/setup.bash"
echo "    export RMW_IMPLEMENTATION=rmw_zenoh_cpp"
