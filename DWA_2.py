#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import math

DT = 0.1
PREDICT_TIME = 3.0
MAX_SPEED = 1.0
MIN_SPEED = 0.3
MAX_YAWRATE = 1.0
MAX_ACCEL = 0.2
MAX_DYAWRATE = 1.0

# Cost function weights
alpha = 2.0  # heading
beta = 1.5   # obstacle distance
gamma = 0.5   # velocity

# Static goal
goal = np.array([-5.0, -6.0])  # Change as needed

class DWAController(Node):
    def __init__(self):
        super().__init__('dwa_controller')
        self.cmd_pub = self.create_publisher(Twist, '/bcr_bot/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/bcr_bot/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/bcr_bot/scan', self.dist_from_obstacle_callback, 10)

        self.obstacle_pts = []
        self.state = np.array([0.0, 0.0, 0.0])  # x, y, yaw
        self.v = 0.0
        self.w = 0.0
        self.timer = self.create_timer(DT, self.timer_callback)

        self.get_logger().info("DWA controller running with dynamic obstacle avoidance")

    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        twist = msg.twist.twist
        self.state[0] = pose.position.x
        self.state[1] = pose.position.y
        yaw = self.quaternion_to_yaw(pose.orientation)
        self.state[2] = yaw
        self.v = twist.linear.x
        self.w = twist.angular.z

    def dist_from_obstacle_callback(self, msg: LaserScan):
        self.obstacle_pts = []
        for i, r in enumerate(msg.ranges):
            if msg.range_min < r < msg.range_max and r < 2.5:  # filter far-away noise
                angle = msg.angle_min + i * msg.angle_increment
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                self.obstacle_pts.append((x, y))

    def timer_callback(self):
        distance_to_goal = np.hypot(goal[0] - self.state[0], goal[1] - self.state[1])
        if distance_to_goal < 0.1:
            self.get_logger().info("Goal reached!")
            self.publish_stop()
            rclpy.shutdown()
            return

        u, _ = self.dwa_control(self.state, self.v, self.w)
        cmd = Twist()
        cmd.linear.x = u[0]
        cmd.angular.z = u[1]
        self.cmd_pub.publish(cmd)

    def dwa_control(self, state, v, w):
        dw = self.calc_dynamic_window(v, w)
        best_cost = float("inf")
        best_u = [0.0, 0.0]

        for vt in np.linspace(dw[0], dw[1], 5):
            for wt in np.linspace(dw[2], dw[3], 5):
                traj = self.calc_trajectory(state.copy(), vt, wt)
                heading = self.calc_heading_cost(traj)
                dist_cost = self.calc_obstacle_cost(traj)
                vel_cost = MAX_SPEED - vt
                total_cost = alpha * heading + beta * dist_cost + gamma * vel_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_u = [vt, wt]
        return best_u, []

    def calc_dynamic_window(self, v, w):
        v_min = max(MIN_SPEED, v - MAX_ACCEL * DT) if v < MIN_SPEED else max(0.0, v - MAX_ACCEL * DT)
        return [
            v_min,
            min(MAX_SPEED, v + MAX_ACCEL * DT),
            max(-MAX_YAWRATE, w - MAX_DYAWRATE * DT),
            min(MAX_YAWRATE, w + MAX_DYAWRATE * DT)
        ]

    def calc_trajectory(self, state, v, w):
        traj = []
        for _ in range(int(PREDICT_TIME / DT)):
            state = self.motion(state, v, w)
            traj.append(state.copy())
        return np.array(traj)

    def motion(self, state, v, w):
        x, y, theta = state
        x += v * np.cos(theta) * DT
        y += v * np.sin(theta) * DT
        theta += w * DT
        return np.array([x, y, theta])

    def calc_heading_cost(self, traj):
        dx = goal[0] - traj[-1][0]
        dy = goal[1] - traj[-1][1]
        goal_theta = np.arctan2(dy, dx)
        error = goal_theta - traj[-1][2]
        return abs(np.arctan2(np.sin(error), np.cos(error)))

    def calc_obstacle_cost(self, traj):
        if not self.obstacle_pts:
            return 0.0

        min_dist = float('inf')
        for x, y, _ in traj:
            for ox_r, oy_r in self.obstacle_pts:
                yaw = self.state[2]
                # Transform from robot to world frame
                ox_w = ox_r * math.cos(yaw) - oy_r * math.sin(yaw) + self.state[0]
                oy_w = ox_r * math.sin(yaw) + oy_r * math.cos(yaw) + self.state[1]
                dist = np.hypot(ox_w - x, oy_w - y)
                if dist < min_dist:
                    min_dist = dist

        if min_dist < 0.3:
            return float('inf')  # collision can happen, discourage this path as much as possible
        return 2.0 / min_dist  # closer is worse (more the distance, lower the cost)

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = DWAController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
