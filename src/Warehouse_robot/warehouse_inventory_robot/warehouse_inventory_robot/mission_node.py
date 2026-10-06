"""
mission_node.py — Student entry point.
"""

import os
import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data

import yaml
import math
import py_trees
from pathlib import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped

from ament_index_python.packages import get_package_share_directory
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, String
from std_srvs.srv import Empty as EmptyService

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

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

# Joint targets for the arm mounted in tb4_lite6.urdf.xacro. The first is the
# provided initial (stowed) configuration. The other two reach down over the
# source and target boxes from the approach poses in shelves.yaml.
SAFE_JOINTS = (0.0, 0.87, 1.57, 0.0, -1.57, 0.0)
PICK_JOINTS = (0.0, 1.50, 1.68, 0.0, 0.26, 0.0)
PLACE_JOINTS = (0.0, 1.50, 2.13, 0.0, 0.53, 0.0)
ARM_JOINT_NAMES = tuple(f'arm_joint{i}' for i in range(1, 7))


class Check(py_trees.behaviour.Behaviour):
    """A condition in the mission tree, evaluated whenever its parent ticks."""

    def __init__(self, name, predicate):
        super().__init__(name)
        self.predicate = predicate

    def update(self):
        return (py_trees.common.Status.SUCCESS if self.predicate()
                else py_trees.common.Status.FAILURE)


class MissionAction(py_trees.behaviour.Behaviour):
    """Run an action and retry a bounded number of times on failure."""

    def __init__(self, name, action, attempts=1):
        super().__init__(name)
        self.action = action
        self.max_attempts = attempts
        self.attempts = 0

    def update(self):
        if self.action():
            return py_trees.common.Status.SUCCESS
        self.attempts += 1
        if self.attempts >= self.max_attempts:
            return py_trees.common.Status.FAILURE
        time.sleep(0.5)
        return py_trees.common.Status.RUNNING


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

        self.declare_parameter('source_box', SOURCE_BOX)
        self.declare_parameter('drop_box', DROP_BOX_BY_GRADE[self.grade])
        self.source_box = str(self.get_parameter('source_box').value)
        self.drop_box = str(self.get_parameter('drop_box').value)

        self.get_logger().info(
            f"Mission node started for grade '{self.grade}'. "
            f'Collect from {self.source_box}, place on {self.drop_box}.')

        self._attach_pub = self.create_publisher(Empty, '/vacuum_gripper/attach', 10)
        self._detach_pub = self.create_publisher(Empty, '/vacuum_gripper/detach', 10)
        self.gripper_attached = None
        self.is_docked = None
        self.joint_positions = {}
        self.amcl_covariance = None
        self.amcl_update_count = 0
        self.localization_initialized = False
        self.odom_yaw = None
        self.rotation_progress = 0.0
        self.create_subscription(String, '/vacuum_gripper/state', self.on_gripper_state, 10)
        self.create_subscription(DockStatus, '/dock_status', self.on_dock_status,
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_joint_state,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_amcl_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom,
                                 qos_profile_sensor_data)
        self._velocity_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self._undock_client = ActionClient(self, Undock, '/undock')
        self._arm_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/lite6_traj_controller/follow_joint_trajectory'
        )

        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._global_localization = self.create_client(
            EmptyService, '/reinitialize_global_localization')

    def on_gripper_state(self, msg):
        value = msg.data.strip().lower()
        if value in ('true', '1', 'attached'):
            self.gripper_attached = True
        elif value in ('false', '0', 'detached'):
            self.gripper_attached = False

    def on_dock_status(self, msg):
        self.is_docked = msg.is_docked

    def on_joint_state(self, msg):
        self.joint_positions.update(zip(msg.name, msg.position))

    def on_amcl_pose(self, msg):
        self.amcl_covariance = msg.pose.covariance
        self.amcl_update_count += 1

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.odom_yaw is not None:
            delta = math.atan2(math.sin(yaw - self.odom_yaw),
                               math.cos(yaw - self.odom_yaw))
            self.rotation_progress += abs(delta)
        self.odom_yaw = yaw

    def arm_is_safe(self):
        return all(name in self.joint_positions and
                   abs(self.joint_positions[name] - angle) < 0.12
                   for name, angle in zip(ARM_JOINT_NAMES, SAFE_JOINTS))

    def localized(self):
        if not self.localization_initialized or self.amcl_covariance is None:
            return False
        covariance = self.amcl_covariance
        return (covariance[0] < 0.25 and covariance[7] < 0.25 and
                covariance[35] < 0.25)

    def undock_robot(self):
        if not self._undock_client.wait_for_server(timeout_sec=60.0):
            self.get_logger().error('Undock action server is unavailable')
            return False

        goal_future = self._undock_client.send_goal_async(Undock.Goal())
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=15.0)
        if not goal_future.done() or goal_future.result() is None:
            self.get_logger().error('Undock goal was not acknowledged')
            return False
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Undock goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error('Undock timed out')
            goal_handle.cancel_goal_async()
            return False
        result = result_future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.is_docked:
            self.get_logger().error(f'Undock failed (status {result.status})')
            return False
        self.is_docked = False
        return True

    def go_to_pose(self, pose_stamped):
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(f"Navigating to x: {pose_stamped.pose.position.x}, y: {pose_stamped.pose.position.y}")
        if not self._nav_client.wait_for_server(timeout_sec=120.0):
            self.get_logger().error('NavigateToPose action server is unavailable')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = pose_stamped

        goal_future = self._nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=30.0)
        if not goal_future.done() or goal_future.result() is None:
            self.get_logger().error('Navigation goal was not acknowledged')
            return False
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal was rejected')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=600.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error('Navigation timed out; cancelling goal')
            goal_handle.cancel_goal_async()
            return False
        result = result_future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(f'Navigation failed (status {result.status})')
            return False
        return True

    def toggle_vacuum(self, enable=True):
        state = "ENGAGING" if enable else "RELEASING"
        self.get_logger().info(f'{state} vacuum gripper...')
        publisher = self._attach_pub if enable else self._detach_pub
        # The detachable-joint plugin publishes its state after processing a
        # command. Repeating once also covers a late ROS/Gazebo bridge match.
        previous = self.gripper_attached
        self.gripper_attached = None
        for _ in range(2):
            publisher.publish(Empty())
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.gripper_attached is enable:
                    return True
        self.get_logger().error(
            f'No gripper confirmation for {state.lower()} (previous state: {previous})')
        return False

    def release_initial_joint(self):
        """Detach at startup; a no-op detach has no state event to acknowledge."""
        for _ in range(2):
            self._detach_pub.publish(Empty())
            deadline = time.monotonic() + 0.7
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
        if self.gripper_attached is True:
            self.get_logger().error('Cube remains attached after initial detach')
            return False
        return True

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
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error('Arm trajectory did not finish in time!')
            goal_handle.cancel_goal_async()
            return False
        result = result_future.result()
        if (result.status != GoalStatus.STATUS_SUCCEEDED or
                result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL):
            self.get_logger().error(
                f'Arm trajectory failed (status {result.status}, '
                f'error {result.result.error_code}: {result.result.error_string})')
            return False
        return True

    def localize(self):
        """Spread AMCL particles and turn in place until the pose converges."""
        if not self._global_localization.wait_for_service(timeout_sec=60.0):
            self.get_logger().error('AMCL global localization service is unavailable')
            return False
        future = self._global_localization.call_async(EmptyService.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done() or future.result() is None:
            self.get_logger().error('AMCL global localization request failed')
            return False
        self.localization_initialized = True
        self.amcl_covariance = None
        deadline = time.monotonic() + 15.0
        while self.odom_yaw is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.odom_yaw is None:
            self.get_logger().error('No wheel odometry received for localization')
            return False

        self.rotation_progress = 0.0
        deadline = time.monotonic() + 180.0
        self.get_logger().info('Rotating in place for AMCL localization')
        try:
            while (rclpy.ok() and time.monotonic() < deadline and
                   self.rotation_progress < 3.0 * 2.0 * math.pi):
                command = TwistStamped()
                command.header.stamp = self.get_clock().now().to_msg()
                command.header.frame_id = 'base_link'
                command.twist.angular.z = 0.4
                self._velocity_pub.publish(command)
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.rotation_progress >= 2.0 * math.pi and self.localized():
                    self.get_logger().info(
                        f'AMCL converged after {self.rotation_progress:.1f} rad')
                    return True
        finally:
            stop = TwistStamped()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.header.frame_id = 'base_link'
            self._velocity_pub.publish(stop)

        if self.amcl_covariance is not None:
            self.get_logger().error(
                f'AMCL did not converge: covariance '
                f'{self.amcl_covariance[0]:.3f}, {self.amcl_covariance[7]:.3f}, '
                f'{self.amcl_covariance[35]:.3f}; '
                f'rotation {self.rotation_progress:.1f} rad')
        else:
            self.get_logger().error('AMCL did not publish a pose during rotation')
        return False


    # =========================================================================
    # THE MAIN MISSION SEQUENCE
    # =========================================================================

    def run_mission(self):
        self.get_logger().info('Starting mission.')

        # Wait for the actual dock state before evaluating the Move precondition.
        deadline = time.monotonic() + 120.0
        while self.is_docked is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        if self.is_docked is None:
            self.get_logger().error('No /dock_status received')
            return False

        home_base = load_home_base()
        pick_pose = load_shelf(self.source_box)
        drop_pose = load_shelf(self.drop_box)

        def move(name, pose):
            # Backward chaining: satisfy each precondition before Move. The
            # selectors check current state first and repair it if necessary.
            prerequisites = [
                py_trees.composites.Selector(
                    name=f'{name}: Undocked', memory=False, children=[
                        Check('Check undocked', lambda: self.is_docked is False),
                        MissionAction('Undock', self.undock_robot),
                    ]),
                py_trees.composites.Selector(
                    name=f'{name}: Arm safe', memory=False, children=[
                        Check('Check arm safe', self.arm_is_safe),
                        MissionAction('Stow arm', lambda:
                                      self.move_arm_to_joint_angles(SAFE_JOINTS)),
                    ]),
            ]
            if self.grade == 'a':
                prerequisites.append(py_trees.composites.Selector(
                    name=f'{name}: Localized', memory=False, children=[
                        Check('Check AMCL', self.localized),
                        MissionAction('Localize', self.localize),
                    ]))
            return py_trees.composites.Sequence(
                name=name, memory=True, children=prerequisites + [
                    MissionAction('Navigate', lambda: self.go_to_pose(pose), attempts=3),
                ])

        root = py_trees.composites.Sequence(name='Warehouse mission', memory=True,
            children=[
                MissionAction('Release initial joint', self.release_initial_joint),
                move('Move to source', pick_pose),
                MissionAction('Lower arm to cube', lambda:
                              self.move_arm_to_joint_angles(PICK_JOINTS)),
                MissionAction('Pick cube', lambda: self.toggle_vacuum(True)),
                MissionAction('Stow carried cube', lambda:
                              self.move_arm_to_joint_angles(SAFE_JOINTS)),
                move('Move to target', drop_pose),
                MissionAction('Lower arm to target', lambda:
                              self.move_arm_to_joint_angles(PLACE_JOINTS)),
                MissionAction('Place cube', lambda: self.toggle_vacuum(False)),
                MissionAction('Stow arm', lambda:
                              self.move_arm_to_joint_angles(SAFE_JOINTS)),
                move('Return to base', home_base),
            ])
        tree = py_trees.trees.BehaviourTree(root)
        while rclpy.ok():
            tree.tick()
            if root.status != py_trees.common.Status.RUNNING:
                succeeded = root.status == py_trees.common.Status.SUCCESS
                self.get_logger().info('Mission complete' if succeeded else 'Mission failed')
                return succeeded
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    succeeded = False

    try:
        succeeded = node.run_mission()
    except KeyboardInterrupt:
        node.get_logger().info("Mission interrupted by user.")
    except Exception as e:
        node.get_logger().fatal(f"Mission failed: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if succeeded else 1

if __name__ == '__main__':
    main()
