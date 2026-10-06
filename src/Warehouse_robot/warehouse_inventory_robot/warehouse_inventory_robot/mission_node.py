"""
mission_node.py — Student entry point.
"""

import os
import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy

import yaml
import math
from pathlib import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from geometry_msgs.msg import Twist, TwistStamped
from ament_index_python.packages import get_package_share_directory
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptyService
from nav_msgs.msg import OccupancyGrid

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# TODO: Add any other necessary imports (e.g., for Nav2 actions, or behavior tree libraries).

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose

# ─────────────────────────────────────────────────────────────────────────────
# WHERE THE MISSION GOES, PER GRADE
#
# Every grade collects the cube from the same source box. They differ only in
# which box it is placed on, and config/shelves.yaml holds all of them.
#
# The grade comes from a ROS parameter rather than being written in here, so it
# cannot silently disagree with the grade the simulation was launched with. Set
# both from one place:
#
#     GRADE=c pixi run mission            # world, odometry, localization
#     GRADE=c pixi run mission-node       # this node
#
# MissionNode logs the grade it is running as, so a mismatch is visible in the
# first line of output rather than showing up as the robot driving to the wrong
# box twenty metres away.
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_BOX = 'shelf_7_ID11'

DROP_BOX_BY_GRADE = {
    'e': 'shelf_7_ID10',   # target-1, near the start
    'c': 'shelf_7_ID20',   # target-2, across the warehouse
    'a': 'shelf_7_ID20',   # target-2, same as C
}

SAFE_JOINTS = [0.0, 0.87, 1.57, 0.0, -1.57, 0.0]
PICK_JOINTS = [0.0, 1.50, 1.68, 0.0, 0.26, 0.0]
PLACE_JOINTS = [0.0, 1.50, 2.13, 0.0, 0.53, 0.0]


def load_shelf(name: str) -> PoseStamped:
    for shelf in load_shelves():
        if shelf['name'] == name:
            return shelf['pose']
    known = [s['name'] for s in load_shelves()]
    raise KeyError(f"no box named '{name}' in shelves.yaml; known boxes: {known}")


def load_shelves(priority_first: bool = False) -> list[dict]:
    pkg_share = get_package_share_directory('warehouse_inventory_robot')
    yaml_path = Path(pkg_share) / 'config' / 'shelves.yaml'

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    shelves = []
    for shelf in data.get('shelves', []):
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose.position.x = float(shelf['pose']['x'])
        ps.pose.position.y = float(shelf['pose']['y'])
        ps.pose.position.z = 0.0
        yaw = float(shelf['pose'].get('yaw', 0.0))
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        shelves.append({
            'name':      shelf['name'],
            'marker_id': int(shelf['marker_id']),
            'priority':  shelf.get('priority', 'normal'),
            'pose':      ps,
        })

    if priority_first:
        shelves.sort(key=lambda s: 0 if s['priority'] == 'high' else 1)

    return shelves

def load_home_base() -> PoseStamped:
    pkg_share = get_package_share_directory('warehouse_inventory_robot')
    yaml_path = Path(pkg_share) / 'config' / 'shelves.yaml'

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    hb = data['home_base']
    ps = PoseStamped()
    ps.header.frame_id = 'map'
    ps.pose.position.x = float(hb['pose']['x'])
    ps.pose.position.y = float(hb['pose']['y'])
    yaw = float(hb['pose'].get('yaw', 0.0))
    ps.pose.orientation.z = math.sin(yaw / 2.0)
    ps.pose.orientation.w = math.cos(yaw / 2.0)
    return ps

class MissionNode(Node):

    def __init__(self):
        super().__init__('mission_node')

        # Which grade this run is for. Read from the GRADE environment
        # variable, so one spelling works whether you go through
        # `GRADE=c pixi run mission-node` or call ros2 run yourself inside a
        # pixi shell. An explicit -p grade:=c overrides it. Use the same value
        # you launched the simulation with.
        self.declare_parameter('grade', os.environ.get('GRADE', 'e'))
        self.grade = str(self.get_parameter('grade').value).strip().lower()
        if self.grade not in DROP_BOX_BY_GRADE:
            self.get_logger().warn(
                f"Unknown grade '{self.grade}'; falling back to 'e'. "
                f"Valid grades: {sorted(DROP_BOX_BY_GRADE)}")
            self.grade = 'e'

        self.source_box = SOURCE_BOX
        self.drop_box = DROP_BOX_BY_GRADE[self.grade]

        self.get_logger().info(
            f"Mission node started for grade '{self.grade}'. "
            f'Collect from {self.source_box}, place on {self.drop_box}.')

        self._attach_pub = self.create_publisher(Empty, '/vacuum_gripper/attach', 10)
        self._detach_pub = self.create_publisher(Empty, '/vacuum_gripper/detach', 10)
        self.is_docked = None
        self.create_subscription(
            DockStatus, '/dock_status', self._dock_status_callback,
            qos_profile_sensor_data)
        self._undock_client = ActionClient(self, Undock, '/undock')
        self._arm_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/lite6_traj_controller/follow_joint_trajectory'
        )

        # TODO: Define other necessary subscribers, publishers, and action clients (e.g., for navigation with Nav2).

        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        if self.grade == 'a':
            self._global_localization = self.create_client(
                EmptyService, '/reinitialize_global_localization')
            self._turn_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
            self._map_received = False
            self._amcl_pose = None
            map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(
                OccupancyGrid, '/map', lambda msg: setattr(self, '_map_received', True), map_qos)
            self.create_subscription(
                PoseWithCovarianceStamped, '/amcl_pose',
                lambda msg: setattr(self, '_amcl_pose', msg), qos_profile_sensor_data)

    def _dock_status_callback(self, msg):
        self.is_docked = msg.is_docked

    def localize_globally(self):
        """Find the grade A start pose without using the examiner's clicked point."""
        deadline = time.monotonic() + 30.0
        while not self._map_received and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self._map_received or not self._global_localization.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('AMCL map or global localization service is unavailable')
            return False

        self._amcl_pose = None
        future = self._global_localization.call_async(EmptyService.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            self.get_logger().error('AMCL global localization request failed')
            return False

        self.get_logger().info('AMCL searching the map; rotating to collect laser scans')
        start = self.get_clock().now()
        # Complete a full turn before trusting a pose; repeated aisles can look alike.
        turn_duration = 2.0 * math.pi / 0.4
        wall_deadline = time.monotonic() + 90.0
        stable_updates = 0
        last_stamp = None
        try:
            while time.monotonic() < wall_deadline and \
                    (self.get_clock().now() - start).nanoseconds * 1e-9 < 2 * turn_duration:
                elapsed = (self.get_clock().now() - start).nanoseconds * 1e-9
                cmd = TwistStamped()
                cmd.header.stamp = self.get_clock().now().to_msg()
                cmd.header.frame_id = 'base_link'
                cmd.twist.angular.z = 0.4
                self._turn_pub.publish(cmd)
                rclpy.spin_once(self, timeout_sec=0.1)

                pose = self._amcl_pose
                if pose is None or pose.header.stamp == last_stamp:
                    continue
                last_stamp = pose.header.stamp
                covariance = pose.pose.covariance
                if elapsed >= turn_duration and covariance[0] < 0.25 and \
                        covariance[7] < 0.25 and covariance[35] < 0.12:
                    stable_updates += 1
                    if stable_updates >= 5:
                        self.get_logger().info('AMCL pose converged')
                        return True
                else:
                    stable_updates = 0
        finally:
            stop = TwistStamped()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.header.frame_id = 'base_link'
            self._turn_pub.publish(stop)

        self.get_logger().error('AMCL did not converge within the search window')
        return False

    def undock_robot(self):
        # TODO: Implement undocking logic using the Undock action client (self._undock_client).
        #       Return True once the base is undocked, False if it refused.

        rclpy.spin_once(self, timeout_sec=1.0)
        if self.is_docked is False:
            return True

        if not self._undock_client.wait_for_server(timeout_sec=10.0):
            return False

        goal_future = self._undock_client.send_goal_async(Undock.Goal())
        rclpy.spin_until_future_complete(self,goal_future)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            rclpy.spin_once(self, timeout_sec=0.5)
            return self.is_docked is False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if (result.status == GoalStatus.STATUS_SUCCEEDED):
            self.is_docked = False
            return True
        else:
            return False

    def go_to_pose(self, pose_stamped):
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(f"Navigating to x: {pose_stamped.pose.position.x}, y: {pose_stamped.pose.position.y}")
        # TODO: Implement navigation to the given pose using Nav2's NavigateToPose action.
        #       Return True once the robot has arrived, False if it did not. Callers
        #       read the return value as "did this work", so falling off the end and
        #       returning None counts as failure.

        goal = NavigateToPose.Goal()
        goal.pose = pose_stamped

        goal_future = self._nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if (result.status == GoalStatus.STATUS_SUCCEEDED):
            return True
        else:
            return False

    def toggle_vacuum(self, enable=True):
        state = "ENGAGING" if enable else "RELEASING"
        self.get_logger().info(f'{state} vacuum gripper...')
        (self._attach_pub if enable else self._detach_pub).publish(Empty())
        time.sleep(1.5)

    def move_arm_to_joint_angles(self, angles, duration_sec=4):
        """Generic helper function to send the arm to any 6-DOF joint configuration."""
        if not self._arm_client.wait_for_server(timeout_sec=120.0):
            self.get_logger().error(
                'Arm action server /lite6_traj_controller/follow_joint_trajectory '
                'never appeared. Is lite6_traj_controller active? Check with: '
                'ros2 control list_controllers')
            return False


        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            'arm_joint1', 'arm_joint2', 'arm_joint3', 
            'arm_joint4', 'arm_joint5', 'arm_joint6'
        ]

        point = JointTrajectoryPoint()
        point.positions = angles 
        point.time_from_start = Duration(sec=duration_sec, nanosec=0)
        goal_msg.trajectory.points.append(point)

        # Every wait below is bounded, so a misbehaving controller will show an
        # error instead of an indefinite hang with no output.
        send_goal_future = self._arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=15.0)
        if not send_goal_future.done():
            self.get_logger().error('Arm trajectory goal was never acknowledged!')
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Arm trajectory goal was rejected!')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=180.0)
        if not result_future.done():
            self.get_logger().error('Arm trajectory did not finish in time!')
            return False
        result = result_future.result()
        return result is not None and result.status == GoalStatus.STATUS_SUCCEEDED


    # =========================================================================
    # THE MAIN MISSION SEQUENCE
    # =========================================================================

    def run_mission(self):
        self.get_logger().info('Starting mission.')

        # Ensure gripper starts in a known-detached state
        time.sleep(1.0)  # allow ROS->bridge->gz discovery to complete across all hops
        self._detach_pub.publish(Empty())
        time.sleep(1.0)

        # Load waypoints. Which box the cube is placed on depends on the grade,
        # so both boxes are looked up BY NAME through the constants at the top
        # of this file rather than by their position in shelves.yaml. Indexing
        # into the list instead would tie the mission to the order of that file
        # and quietly send the robot to the wrong box when it changed.
        home_base = load_home_base()
        pick_pose = load_shelf(self.source_box)
        drop_pose = load_shelf(self.drop_box)

        # TODO: Implement the mission logic (either State Machine or Behavior Tree).

        state = 'ARM_SAFE'
        attempts = 0
        max_attempts = 10

        while state not in ('DONE', 'FAILED'):
            self.get_logger().info(f'{state} state running')

            if state == 'ARM_SAFE':
                done = self.move_arm_to_joint_angles(SAFE_JOINTS)
                next_state = 'UNDOCK'

            elif state == 'UNDOCK':
                done = self.undock_robot()
                next_state = 'LOCALIZE' if self.grade == 'a' else 'GO_TO_PICK'

            elif state == 'LOCALIZE':
                done = self.localize_globally()
                next_state = 'GO_TO_PICK'
            
            elif state == 'GO_TO_PICK':
                done = self.go_to_pose(pick_pose)
                next_state = 'ARM_TO_PICK'

            elif state == 'ARM_TO_PICK':
                done = self.move_arm_to_joint_angles(PICK_JOINTS)
                next_state = 'PICK'

            elif state == 'PICK':
                self.toggle_vacuum(True)
                done = True
                next_state = 'ARM_SAFE_AFTER_PICK'

            elif state == 'ARM_SAFE_AFTER_PICK':
                done = self.move_arm_to_joint_angles(SAFE_JOINTS)
                next_state = 'GO_TO_DROP'

            elif state == 'GO_TO_DROP':
                done = self.go_to_pose(drop_pose)
                next_state = 'ARM_TO_PLACE'

            elif state == 'ARM_TO_PLACE':
                done = self.move_arm_to_joint_angles(PLACE_JOINTS)
                next_state = 'DROP'
            
            elif state == 'DROP':
                self.toggle_vacuum(False)
                done = True
                next_state = 'ARM_SAFE_AFTER_DROP'

            elif state == 'ARM_SAFE_AFTER_DROP':
                done = self.move_arm_to_joint_angles(SAFE_JOINTS)
                next_state = 'GO_TO_BASE'
                
            elif state == 'GO_TO_BASE':
                done = self.go_to_pose(home_base)
                next_state = 'DONE'

            else:
                state = 'FAILED'
                continue

            if done:
                state = next_state
                attempts = 0

            else:
                attempts += 1

                if attempts >= max_attempts:
                    state = 'FAILED'
                
                else:
                    time.sleep(0.5)
        
        if state == 'DONE':
            return True
        
        else:
            return False

def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()

    try:
        node.run_mission()
    except KeyboardInterrupt:
        node.get_logger().info("Mission interrupted by user.")
    except Exception as e:
        node.get_logger().fatal(f"Mission failed: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
