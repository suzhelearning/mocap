#!/usr/bin/env python3
"""manus_pub.py — Manus raw 骨架 → ROS2 (Jazzy) / Zenoh 发布器

从 stdin 读取 rawviz 协议（HAND/EDGE/FRAME/POS），发布为 ROS2 话题：

  /manus/raw_skeleton/<side>        geometry_msgs/PoseArray   25 节点位置（帧率 ~30Hz）
  /manus/skeleton_edges/<side>      std_msgs/Int32MultiArray  (child,parent) 对，一次性

用法（Zenoh 传输）:
    source /opt/ros/jazzy/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    ./rawviz.out | python manus_pub.py

验证:
    ros2 topic list
    ros2 topic hz /manus/raw_skeleton/left
    ros2 topic echo /manus/raw_skeleton/left --once
"""

import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Int32MultiArray
import numpy as np


class ManusRawPublisher(Node):
    def __init__(self):
        super().__init__("manus_raw_publisher")
        self.pos_pubs = {}   # side -> PoseArray publisher
        self.edge_pubs = {}  # side -> edges publisher
        self.hands = {}      # glove_id -> Hand
        self.frame_no = 0

    # -- 话题创建 -----------------------------------------------------------
    def _ensure_pubs(self, side, glove_id):
        if side in self.pos_pubs:
            return
        self.pos_pubs[side] = self.create_publisher(
            PoseArray, f"/manus/raw_skeleton/{side.lower()}", qos_profile_sensor_data)
        self.edge_pubs[side] = self.create_publisher(
            Int32MultiArray, f"/manus/skeleton_edges/{side.lower()}", qos_profile_sensor_data)
        self.get_logger().info(
            f"手套 {side} ({glove_id}) → /manus/raw_skeleton/{side.lower()}")

    # -- 解析 ---------------------------------------------------------------
    def handle_line(self, line):
        parts = line.split()
        if not parts:
            return
        tag = parts[0]
        if tag == "HAND" and len(parts) >= 4:
            gid, side, n = parts[1], parts[2], int(parts[3])
            self.hands[gid] = {"side": side, "node_count": n,
                               "edges": [], "pos": np.zeros((n, 3)), "valid": False}
        elif tag == "EDGE" and len(parts) == 5:
            h = self.hands.get(parts[1])
            if h:
                h["edges"].append((int(parts[2]) - 1, int(parts[3]) - 1))
        elif tag == "POS" and len(parts) >= 4:
            h = self.hands.get(parts[1])
            if h:
                vals = np.asarray(parts[2:2 + h["node_count"] * 3], dtype=float)
                if vals.size == h["node_count"] * 3:
                    h["pos"] = vals.reshape(h["node_count"], 3)
                    h["valid"] = True
        elif tag == "FRAME":
            self.frame_no += 1
            self.publish()

    # -- 发布 ---------------------------------------------------------------
    def publish(self):
        for gid, h in self.hands.items():
            if not h["valid"]:
                continue
            self._ensure_pubs(h["side"], gid)

            # 25 节点位置 → PoseArray
            msg = PoseArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f"manus_{h['side'].lower()}"
            for p in h["pos"]:
                pose = Pose()
                pose.position.x = float(p[0])
                pose.position.y = float(p[1])
                pose.position.z = float(p[2])
                msg.poses.append(pose)
            self.pos_pubs[h["side"]].publish(msg)

            # 边信息（每只手发布一次）
            if h["edges"] and not getattr(self, f"_edges_sent_{gid}", False):
                em = Int32MultiArray()
                em.data = [v for e in h["edges"] for v in e]
                self.edge_pubs[h["side"]].publish(em)
                setattr(self, f"_edges_sent_{gid}", True)


def main():
    rclpy.init()
    node = ManusRawPublisher()
    try:
        for line in sys.stdin:
            node.handle_line(line.strip())
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("\n[stdin 关闭，发布结束]")


if __name__ == "__main__":
    main()
