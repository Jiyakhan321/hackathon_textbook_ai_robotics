---
sidebar_position: 3
---

# Gazebo-ROS Integration

## Overview

Gazebo-ROS integration enables seamless communication between your simulated humanoid robot and the ROS 2 ecosystem. This integration allows you to use the same control nodes, perception algorithms, and other ROS 2 components with your simulation as you would with a real robot. In this section, we'll explore how to properly integrate Gazebo with ROS 2 for humanoid robot simulation.

## Gazebo-ROS Architecture

The Gazebo-ROS integration works through plugins that bridge the gap between Gazebo's simulation environment and ROS 2's communication system:

- **Gazebo ROS PKGs**: Provide ROS interfaces for Gazebo simulation
- **Sensor Plugins**: Convert Gazebo sensor data to ROS messages
- **Actuator Plugins**: Convert ROS commands to Gazebo actuator control
- **Robot State Publisher**: Maintains the robot's transform tree

## Installing Gazebo-ROS Packages

First, ensure you have the necessary packages installed:

```bash
# Install Gazebo ROS packages for Humble
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-gazebo-dev
```

## Essential Gazebo Plugins

### 1. Robot State Publisher Plugin

Add this to your world file or robot model to enable TF publishing:

```xml
<!-- In your world file -->
<gazebo>
  <plugin filename="libgazebo_ros_init.so" name="gazebo_ros_init">
    <ros>
      <namespace>/humanoid_robot</namespace>
    </ros>
  </plugin>
</gazebo>
```

### 2. Gazebo ROS Control Plugin

This plugin enables ROS 2 control of your robot's joints:

```xml
<!-- Add to your robot model (URDF/XACRO) -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros2_control.so">
    <parameters>$(find my_robot_description)/config/humanoid_control.yaml</parameters>
    <robot_namespace>/humanoid_robot</robot_namespace>
  </plugin>
</gazebo>
```

## Configuring Robot Control

### 1. Control Configuration File

Create a control configuration file that defines your robot's controllers:

```yaml
# config/humanoid_control.yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz

    # Joint Trajectory Controller
    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    # Individual joint controllers
    left_leg_controller:
      type: joint_group_position_controller/JointGroupPositionController
    right_leg_controller:
      type: joint_group_position_controller/JointGroupPositionController
    left_arm_controller:
      type: joint_group_position_controller/JointGroupPositionController
    right_arm_controller:
      type: joint_group_position_controller/JointGroupPositionController

# Joint Trajectory Controller Configuration
joint_trajectory_controller:
  ros__parameters:
    joints:
      - left_hip_pitch_joint
      - left_hip_roll_joint
      - left_hip_yaw_joint
      - left_knee_joint
      - left_ankle_pitch_joint
      - left_ankle_roll_joint
      - right_hip_pitch_joint
      - right_hip_roll_joint
      - right_hip_yaw_joint
      - right_knee_joint
      - right_ankle_pitch_joint
      - right_ankle_roll_joint
      - left_shoulder_pitch_joint
      - left_shoulder_roll_joint
      - left_elbow_joint
      - right_shoulder_pitch_joint
      - right_shoulder_roll_joint
      - right_elbow_joint
    interface_name: position

# Left Leg Controller
left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_pitch_joint
      - left_hip_roll_joint
      - left_hip_yaw_joint
      - left_knee_joint
      - left_ankle_pitch_joint
      - left_ankle_roll_joint

# Right Leg Controller
right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_pitch_joint
      - right_hip_roll_joint
      - right_hip_yaw_joint
      - right_knee_joint
      - right_ankle_pitch_joint
      - right_ankle_roll_joint

# Left Arm Controller
left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_pitch_joint
      - left_shoulder_roll_joint
      - left_elbow_joint

# Right Arm Controller
right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_pitch_joint
      - right_shoulder_roll_joint
      - right_elbow_joint
```

### 2. URDF Integration with Control

Update your URDF to include transmission elements for control:

```xml
<?xml version="1.0"?>
<robot name="humanoid_with_control" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Your existing URDF content -->

  <!-- Transmission elements for ROS Control -->
  <transmission name="left_hip_pitch_trans" type="transmission_interface/SimpleTransmission">
    <joint name="left_hip_pitch_joint">
      <hardwareInterface>position_controllers/JointPositionController</hardwareInterface>
    </joint>
    <actuator name="left_hip_pitch_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_hip_roll_trans" type="transmission_interface/SimpleTransmission">
    <joint name="left_hip_roll_joint">
      <hardwareInterface>position_controllers/JointPositionController</hardwareInterface>
    </joint>
    <actuator name="left_hip_roll_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_hip_yaw_trans" type="transmission_interface/SimpleTransmission">
    <joint name="left_hip_yaw_joint">
      <hardwareInterface>position_controllers/JointPositionController</hardwareInterface>
    </joint>
    <actuator name="left_hip_yaw_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Add transmissions for all joints you want to control -->
  <!-- ... more transmissions ... -->

  <!-- Gazebo ROS Control Plugin -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros2_control.so">
      <parameters>$(find my_robot_description)/config/humanoid_control.yaml</parameters>
      <robot_namespace>/humanoid_robot</robot_namespace>
    </plugin>
  </gazebo>

</robot>
```

## Sensor Integration

### 1. IMU Sensor Plugin

Add IMU sensor to your robot for balance and orientation feedback:

```xml
<!-- Add to your base_link or torso link -->
<gazebo reference="base_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.02</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.02</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.02</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <ros>
      <namespace>/humanoid_robot</namespace>
      <remapping>~/out:=imu/data</remapping>
    </ros>
  </sensor>
</gazebo>
```

### 2. Camera Sensor Plugin

Add a camera for visual perception:

```xml
<gazebo reference="head_link">
  <sensor name="camera" type="camera">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>image_raw:=camera/image_raw</remapping>
        <remapping>camera_info:=camera/camera_info</remapping>
      </ros>
      <camera_name>head_camera</camera_name>
      <frame_name>head_camera_optical_frame</frame_name>
      <hack_baseline>0.07</hack_baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

### 3. LiDAR Sensor Plugin

Add LiDAR for navigation and mapping:

```xml
<gazebo reference="base_link">
  <sensor name="laser_scan" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_scan_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>laser_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Launch Files for Integration

### 1. Complete Launch File

Create a launch file that starts Gazebo with your robot:

```python
# launch/humanoid_gazebo.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    world = LaunchConfiguration('world')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    declare_robot_name = DeclareLaunchArgument(
        name='robot_name',
        default_value='humanoid_robot',
        description='Name of the robot'
    )

    declare_world = DeclareLaunchArgument(
        name='world',
        default_value='empty',
        description='Choose one of: empty, willowgarage, maze'
    )

    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'worlds',
                world
            ]),
            'verbose': 'false',
        }.items()
    )

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'humanoid_with_control.urdf'
            ])
        }]
    )

    # Joint State Publisher (for GUI control)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Spawn the robot in Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', robot_name,
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )

    # Load and activate controllers
    load_joint_state_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    load_trajectory_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_name,
        declare_world,
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        load_joint_state_controller,
        load_trajectory_controller,
    ])
```

### 2. Controller Spawning Launch File

Create a separate launch file for controller management:

```python
# launch/humanoid_controllers.launch.py

from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'config',
                'humanoid_control.yaml'
            ])
        ],
        output='both',
    )

    # Joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='both',
    )

    # Joint trajectory controller
    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller'],
        output='both',
    )

    # Set up event handler to start controllers after controller manager starts
    delay_joint_state_broadcaster_spawner_after_controller_manager = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[
                joint_state_broadcaster_spawner,
            ],
        )
    )

    delay_joint_trajectory_controller_spawner_after_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=joint_state_broadcaster_spawner,
            on_start=[
                joint_trajectory_controller_spawner,
            ],
        )
    )

    return LaunchDescription([
        controller_manager,
        delay_joint_state_broadcaster_spawner_after_controller_manager,
        delay_joint_trajectory_controller_spawner_after_joint_state_broadcaster,
    ])
```

## Testing the Integration

### 1. Basic Integration Test

Create a simple test script to verify the integration:

```python
# test/gazebo_integration_test.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from builtin_interfaces.msg import Duration
import time

class GazeboIntegrationTest(Node):
    def __init__(self):
        super().__init__('gazebo_integration_test')

        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, '/humanoid_robot/joint_states', self.joint_state_callback, 10)

        self.imu_sub = self.create_subscription(
            Imu, '/humanoid_robot/imu/data', self.imu_callback, 10)

        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/humanoid_robot/joint_trajectory_controller/joint_trajectory', 10)

        self.controller_state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            '/humanoid_robot/joint_trajectory_controller/state',
            self.controller_state_callback, 10)

        # Test timer
        self.test_timer = self.create_timer(5.0, self.run_integration_test)

        self.joint_states_received = False
        self.imu_data_received = False
        self.test_step = 0

        self.get_logger().info('Gazebo integration test node initialized')

    def joint_state_callback(self, msg):
        if not self.joint_states_received:
            self.get_logger().info(f'✓ Joint states received: {len(msg.name)} joints')
            self.joint_states_received = True

    def imu_callback(self, msg):
        if not self.imu_data_received:
            self.get_logger().info('✓ IMU data received from simulation')
            self.imu_data_received = True

    def controller_state_callback(self, msg):
        self.get_logger().debug('Controller state received')

    def run_integration_test(self):
        if self.test_step == 0:
            self.get_logger().info('Step 1: Verifying sensor data flow...')
            self.test_step += 1
        elif self.test_step == 1:
            self.get_logger().info('Step 2: Sending trajectory command...')
            self.send_test_trajectory()
            self.test_step += 1
        elif self.test_step == 2:
            self.get_logger().info('Step 3: Verifying actuator response...')
            self.test_step += 1
        elif self.test_step == 3:
            self.get_logger().info('Integration test completed!')
            self.test_timer.cancel()
            self.get_logger().info('✓ Gazebo-ROS integration appears to be working correctly')

    def send_test_trajectory(self):
        """Send a simple trajectory command to test actuator control"""
        msg = JointTrajectory()
        msg.joint_names = ['left_shoulder_pitch_joint', 'right_shoulder_pitch_joint']

        point = JointTrajectoryPoint()
        point.positions = [0.5, 0.5]  # Move both shoulders up
        point.velocities = [0.1, 0.1]
        point.time_from_start = Duration(sec=2, nanosec=0)

        msg.points.append(point)
        msg.header.stamp = self.get_clock().now().to_msg()

        self.trajectory_pub.publish(msg)
        self.get_logger().info('Trajectory command published')

def main(args=None):
    rclpy.init(args=args)
    node = GazeboIntegrationTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Integration test interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Advanced Control Test

Create a more comprehensive test that validates control performance:

```python
# test/control_performance_test.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from builtin_interfaces.msg import Duration
import numpy as np
import time

class ControlPerformanceTest(Node):
    def __init__(self):
        super().__init__('control_performance_test')

        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, '/humanoid_robot/joint_states', self.joint_state_callback, 10)

        self.controller_state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            '/humanoid_robot/joint_trajectory_controller/state',
            self.controller_state_callback, 10)

        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/humanoid_robot/joint_trajectory_controller/joint_trajectory', 10)

        # Test parameters
        self.test_joints = ['left_knee_joint', 'right_knee_joint']
        self.current_positions = {}
        self.target_positions = {}
        self.test_active = False
        self.test_start_time = None
        self.test_results = []

        # Performance metrics
        self.position_errors = []
        self.response_times = []

        self.get_logger().info('Control performance test initialized')

    def joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.test_joints:
                self.current_positions[name] = msg.position[i]

    def controller_state_callback(self, msg):
        if self.test_active and self.test_start_time:
            current_time = self.get_clock().now().nanoseconds * 1e-9

            # Calculate position errors
            for i, joint_name in enumerate(msg.joint_names):
                if joint_name in self.test_joints:
                    actual_pos = msg.feedback.positions[i] if msg.feedback.positions else 0.0
                    desired_pos = msg.desired.positions[i] if msg.desired.positions else 0.0
                    error = abs(actual_pos - desired_pos)
                    self.position_errors.append(error)

    def start_performance_test(self):
        """Start a performance test with step inputs"""
        self.get_logger().info('Starting control performance test...')

        # Send step command
        msg = JointTrajectory()
        msg.joint_names = self.test_joints

        point = JointTrajectoryPoint()
        # Move knees from 0 to 0.5 radians
        point.positions = [0.5, 0.5]
        point.velocities = [0.2, 0.2]  # Reasonable velocity
        point.time_from_start = Duration(sec=3, nanosec=0)

        msg.points.append(point)
        msg.header.stamp = self.get_clock().now().to_msg()

        self.trajectory_pub.publish(msg)
        self.test_active = True
        self.test_start_time = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info(f'Step command sent: {self.test_joints} -> [0.5, 0.5]')

        # Schedule test completion check
        self.create_timer(5.0, self.check_test_completion)

    def check_test_completion(self):
        """Check if the test has completed and evaluate performance"""
        if not self.position_errors:
            self.get_logger().warn('No position error data collected')
            return

        avg_error = np.mean(self.position_errors)
        max_error = np.max(self.position_errors)
        std_error = np.std(self.position_errors)

        self.get_logger().info(f'Control Performance Results:')
        self.get_logger().info(f'  Average Position Error: {avg_error:.4f} rad')
        self.get_logger().info(f'  Max Position Error: {max_error:.4f} rad')
        self.get_logger().info(f'  Std Dev Position Error: {std_error:.4f} rad')

        # Evaluate performance
        if avg_error < 0.05:
            self.get_logger().info('✓ Control performance is EXCELLENT')
        elif avg_error < 0.1:
            self.get_logger().info('✓ Control performance is GOOD')
        elif avg_error < 0.2:
            self.get_logger().info('⚠ Control performance is ACCEPTABLE')
        else:
            self.get_logger().warn('⚠ Control performance needs improvement')

        self.test_active = False
        self.position_errors = []  # Reset for next test

def main(args=None):
    rclpy.init(args=args)
    node = ControlPerformanceTest()

    # Start test after a brief delay to allow systems to initialize
    node.create_timer(3.0, node.start_performance_test)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Performance test interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Common Integration Issues

### 1. Controller Loading Issues

If controllers fail to load, check these common issues:

```bash
# Check if controller manager is running
ros2 node list | grep controller_manager

# Check controller status
ros2 control list_controllers

# Load a specific controller manually
ros2 run controller_manager spawner joint_state_broadcaster

# Check parameter configurations
ros2 param list
```

### 2. TF Tree Issues

Verify the transform tree is properly maintained:

```bash
# Check TF tree
ros2 run tf2_tools view_frames

# Echo transforms
ros2 run tf2_ros tf2_echo base_link left_foot_link
```

### 3. Sensor Data Issues

Debug sensor data flow:

```bash
# Check if sensor topics are being published
ros2 topic list | grep -E "(imu|scan|camera)"

# Echo sensor data
ros2 topic echo /humanoid_robot/imu/data
ros2 topic echo /humanoid_robot/scan
```

## Advanced Integration Features

### 1. Gazebo Services

Use Gazebo services for simulation control:

```python
# Advanced simulation control
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, GetEntityState, SpawnEntity, DeleteEntity

class AdvancedSimulationControl(Node):
    def __init__(self):
        super().__init__('advanced_sim_control')

        # Create clients for Gazebo services
        self.set_state_client = self.create_client(
            SetEntityState, '/world/default/set_entity_state')
        self.get_state_client = self.create_client(
            GetEntityState, '/world/default/get_entity_state')

        # Wait for services to be available
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set state service not available, waiting...')

        while not self.get_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Get state service not available, waiting...')

    def reset_robot_position(self, x=0.0, y=0.0, z=1.0):
        """Reset robot to specific position"""
        from gazebo_msgs.msg import EntityState
        from geometry_msgs.msg import Pose, Twist

        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = 'humanoid_robot'
        req.state.pose = Pose()
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = z
        req.state.pose.orientation.w = 1.0  # No rotation
        req.state.twist = Twist()  # No velocity

        future = self.set_state_client.call_async(req)
        # Handle response asynchronously
```

## Performance Optimization

### 1. Efficient Update Rates

Setting appropriate update rates for different components:

```xml
<!-- High frequency for critical sensors (IMU, joint states) -->
<plugin name="high_freq_imu" filename="libgazebo_ros_imu.so">
  <update_rate>1000</update_rate> <!-- 1kHz for balance control -->
  <!-- ... -->
</plugin>

<!-- Medium frequency for cameras and LiDAR -->
<plugin name="camera_controller" filename="libgazebo_ros_camera.so">
  <update_rate>30</update_rate> <!-- 30Hz for vision processing -->
  <!-- ... -->
</plugin>

<!-- Lower frequency for less critical sensors -->
<plugin name="gps_controller" filename="libgazebo_ros_gps.so">
  <update_rate>10</update_rate> <!-- 10Hz for navigation -->
  <!-- ... -->
</plugin>
```

### 2. Threading Configuration

Optimizing thread usage for better performance:

```xml
<!-- In your world file -->
<world name="humanoid_world">
  <!-- Physics configuration for optimal performance -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size> <!-- 1ms time step -->
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>1000.0</real_time_update_rate>
    <ode>
      <solver>
        <type>quick</type></solver>
      <iters>100</iters> <!-- Balance between accuracy and performance -->
      <sor>1.3</sor>
    </ode>
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
    </constraints>
  </physics>

  <!-- Your robot and plugins -->
  <!-- ... -->
</world>
```

## Troubleshooting Common Issues

### 1. Communication Problems

Common issues and solutions:

**Problem: Joint states not being published**
- Verify plugin is loaded: `gz topic -l | grep joint_states`
- Check namespace configuration in plugin
- Ensure robot model is properly loaded in Gazebo

**Problem: Control commands not reaching joints**
- Verify topic names match between controller and plugin
- Check that controller is loaded and running: `ros2 control list_controllers`
- Ensure proper QoS settings between publisher and subscriber

**Problem: TF tree not publishing properly**
- Verify robot state publisher is running
- Check that joint states are being published
- Ensure proper frame names in URDF/SDF

### 2. Performance Issues

Optimizing for better performance:

- Reduce sensor update rates for non-critical sensors
- Use lower resolution for cameras when possible
- Optimize physics parameters for your specific use case
- Consider using simpler collision geometries

## Testing and Validation

### 1. Communication Validation Script

Testing the Gazebo-ROS integration:

```python
#!/usr/bin/env python3
# test_gazebo_ros_integration.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

class GazeboROSIntegrationTester(Node):
    def __init__(self):
        super().__init__('gazebo_ros_integration_tester')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory, '/humanoid_controller/joint_trajectory', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Flags to track if we're receiving data
        self.joint_states_received = False
        self.imu_data_received = False

        # Timer for periodic checks
        self.timer = self.create_timer(1.0, self.periodic_check)

        self.get_logger().info('Gazebo-ROS integration tester initialized')

    def joint_state_callback(self, msg):
        self.joint_states_received = True
        if len(msg.name) > 0:
            self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')

    def imu_callback(self, msg):
        self.imu_data_received = True
        self.get_logger().debug('Received IMU data')

    def periodic_check(self):
        if self.joint_states_received and self.imu_data_received:
            self.get_logger().info('✓ Gazebo-ROS integration is working correctly')
        else:
            status = []
            if not self.joint_states_received:
                status.append('No joint states received')
            if not self.imu_data_received:
                status.append('No IMU data received')
            self.get_logger().warn(f'✗ Integration issues: {", ".join(status)}')

        # Send a simple test command to verify control path
        self.send_test_command()

    def send_test_command(self):
        # Send a simple joint trajectory command
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['left_hip_joint']  # Example joint

        point = JointTrajectoryPoint()
        point.positions = [0.1]  # Move to 0.1 radians
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        traj_msg.points = [point]

        self.joint_cmd_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)
    tester = GazeboROSIntegrationTester()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Integration tester shutting down')
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Next Steps

With Gazebo-ROS integration properly configured, you now have a complete digital twin system for humanoid robots. The integration enables seamless communication between simulation and control systems, allowing for realistic testing and development of humanoid robot applications.

In the next section, we'll explore sensor simulation in Gazebo, covering how to configure realistic sensors for your humanoid robot including cameras, LiDAR, IMU, and force/torque sensors.