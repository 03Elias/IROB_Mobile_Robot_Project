"""
mission_node.py — Student entry point.
"""

import os
import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionClient

import yaml
import math
from pathlib import Path
from geometry_msgs.msg import PoseStamped

from geometry_msgs.msg import Twist, TwistStamped
from ament_index_python.packages import get_package_share_directory
from irobot_create_msgs.action import Undock
from std_msgs.msg import Empty

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

#build and run the mission behaviour tree
import py_trees
#set communication rules for sensor subscriptions
from rclpy.qos import qos_profile_sensor_data
#receive AMCL poses and covariance
from geometry_msgs.msg import PoseWithCovarianceStamped
#receive the robot's docking status
from irobot_create_msgs.msg import DockStatus
#call the global-localization service
from std_srvs.srv import Empty as EmptyService
#check the final status of ROS actions
from action_msgs.msg import GoalStatus
#send navigation goals to Nav2
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

#agles for joins
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
        self._undock_client = ActionClient(self, Undock, '/undock')
        self._arm_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/lite6_traj_controller/follow_joint_trajectory'
        )

        #get the dock messages
        self.is_docked = None
        self.create_subscription(
            DockStatus, '/dock_status', self._dock_status_callback,
            qos_profile_sensor_data)

        #get the pose messages
        self._amcl_pose = None
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose',
            self._amcl_pose_callback,
            qos_profile_sensor_data)

        #create a nav2 action cleint
        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        #create service that distributes the particle for the amcl
        self._localization = self.create_client(EmptyService, '/reinitialize_global_localization')
        #create a publishes for robot to be able to turn when scanning
        self._turn_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

    #when dock_status publishes a message the callback is called
    #and the newest value of dock_status is stored
    def _dock_status_callback(self, msg):
        self.is_docked = msg.is_docked

    #when amcl_pose publishes a message the callback is called
    #and the newest value of amcl_pose is stored
    def _amcl_pose_callback(self, msg):
        self._amcl_pose = msg

    def localize(self):
        if not self._localization.wait_for_service(timeout_sec=10.0):
            return False

        self._amcl_pose = None
        future = self._localization.call_async(EmptyService.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            return False

        self.get_logger().info('AMCL searching for its position')
        cmd = TwistStamped()
        last_stamp = None
        try:
            while True:
                cmd.header.stamp = self.get_clock().now().to_msg()

                #we make and publish command to rotate 0.4 rad/s
                cmd.twist.angular.z = 0.4
                self._turn_pub.publish(cmd)

                #we process messages for 0.1 sec so that _amcl_pose updates
                rclpy.spin_once(self, timeout_sec=0.1)
                pose = self._amcl_pose

                if pose is not None and pose.header.stamp != last_stamp:
                    last_stamp = pose.header.stamp
                    #get the covarience 6x6 matrix which represent uncertainties
                    covariance = pose.pose.covariance
                    #this number was figured out by logging all the convariances and
                    #looking for which was the lowest one reached
                    #covarience[0], [7], and [35] are x, y, and yaw
                    if (covariance[0] < 20.0 and covariance[7] < 20.0
                            and covariance[35] < 0.5):
                        break

            self.get_logger().info('AMCL position found')
            return True
        finally:
            #stop rotating the robot, including when localization is interrupted
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.twist.angular.z = 0.0
            self._turn_pub.publish(cmd)

    def undock_robot(self):
        #we give the docking status callback time to update
        rclpy.spin_once(self, timeout_sec=1.0)
        if self.is_docked is False:
            return True

        #wait for up to 10 sec for the action server (ROS node) to become available
        if not self._undock_client.wait_for_server(timeout_sec=10.0):
            return False

        #create a goal and wait for the action server to accept it
        #goal_future is empty first and then has the answer whether the goal was accepted
        goal_future = self._undock_client.send_goal_async(Undock.Goal())
        rclpy.spin_until_future_complete(self,goal_future)
        #the goal handle is the specific undocking request
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            rclpy.spin_once(self, timeout_sec=0.5)
            return self.is_docked is False

        #get the result of the request and the status of the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        #if the status is successful return True
        if (result.status == GoalStatus.STATUS_SUCCEEDED):
            self.is_docked = False
            return True
        else:
            return False

    def go_to_pose(self, pose_stamped):
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(f"Navigating to x: {pose_stamped.pose.position.x}, y: {pose_stamped.pose.position.y}")

        #create a navigation goal and give it the destination pose
        goal = NavigateToPose.Goal()
        goal.pose = pose_stamped

        #send the goal and wait for the action server to accept it
        #goal_future is empty first and then has the answer whether the goal was accepted
        goal_future = self._nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        #the goal handle is the specific navigation request
        goal_handle = goal_future.result()

        #return False if the navigation goal was rejected
        if not goal_handle.accepted:
            return False

        #get the result of the request and the status of the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        #if the status is successful return True
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
        # Added
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
        dock_base = load_home_base()
        pick_pose = load_shelf(self.source_box)
        drop_pose = load_shelf(self.drop_box)

        self.arm_is_safe = False
        self.localized = False
        #stores the destination that have been reachesd
        reached = set()

        #create bt leaf from a function
        def leaf(name, function, log=False):
            def update(behaviour):
                #log which leaf is running
                if log:
                    self.get_logger().info(f'{name} running')
                #run the function and make the bt status true or false
                if function():
                    return py_trees.common.Status.SUCCESS
                return py_trees.common.Status.FAILURE

            #make a behaviour from function
            behaviour = py_trees.meta.create_behaviour_from_function(update)
            return behaviour(name)

        #create an action leaf
        def action_node(name, function):
            return py_trees.decorators.Retry(
                name=f'Retry {name}',
                child=leaf(name, function, log=True),
                num_failures=10)

        #create a codition
        def condition(name, check, action_name, action):
            return py_trees.composites.Selector(
                name=name,
                #makes the selector check the condition every time it is ticked
                memory=False,
                #creates two children, checks condition, if it is false performs the action
                children=[
                    leaf(f'{name}?', check),
                    action_node(action_name, action),
                ])

        #move arm to safe pose
        def move_arm(angles, safe):
            self.arm_is_safe = False
            success = self.move_arm_to_joint_angles(angles)
            if success:
                self.arm_is_safe = safe
            return success

        #localize
        def run_localization():
            self.localized = self.localize()
            return self.localized

        #engage vacuum
        def vacuum(enable):
            self.toggle_vacuum(enable)
            return True

        def arm_is_safe():
            return self.arm_is_safe

        def move_arm_safe():
            return move_arm(SAFE_JOINTS, True)

        def is_undocked():
            return self.is_docked is False

        def is_localized():
            return self.localized

        def move_arm_to_pick():
            return move_arm(PICK_JOINTS, False)

        def pick():
            return vacuum(True)

        def move_arm_to_drop():
            return move_arm(PLACE_JOINTS, False)

        def drop():
            return vacuum(False)

        #subtree for moving to pick, drop and dock
        def move(name, pose):

            #navigate to pose
            def navigate():
                success = self.go_to_pose(pose)
                if success:
                    reached.add(name)
                return success

            def destination_reached():
                return name in reached

            #firstly calls for example arm_is_safe, if is not then move_arm_safe
            move_children = [
                condition(
                    'Arm In Safe Pos',
                    arm_is_safe,
                    'Move Arm to Safe Pos',
                    move_arm_safe),
                condition(
                    'Undocked',
                    is_undocked,
                    'Undock',
                    self.undock_robot),
                condition(
                    'Localized',
                    is_localized,
                    'Localization',
                    run_localization),
                #creates an action leaf that is executed only if conditions are true
                action_node(f'Move to {name}', navigate),
            ]

            #return a subtree for moving to pick, drop, dock
            return py_trees.composites.Selector(
                name=f'{name} Reached',
                memory=False,
                #first child check if destination has been reached, the second one runs the move sequence
                children=[
                    leaf(f'At {name}?', destination_reached),
                    py_trees.composites.Sequence(
                        name='Move', memory=True, children=move_children),
                ])

        #create the root node
        mission = py_trees.composites.Sequence(
            name='Mission',
            memory=True,
            children=[
                move('Pick', pick_pose),
                action_node('Arm to Pick', move_arm_to_pick),
                action_node('Pick', pick),
                move('Drop', drop_pose),
                action_node('Arm to Drop', move_arm_to_drop),
                action_node('Drop', drop),
                move('Dock', dock_base),
            ])

        #create a bt
        tree = py_trees.trees.BehaviourTree(mission)
        #run until the mission either succeeds or fails
        while mission.status not in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
            #starts at the root and travels down
            tree.tick()
            if mission.status == py_trees.common.Status.RUNNING:
                time.sleep(0.1)

        if mission.status == py_trees.common.Status.SUCCESS:
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
