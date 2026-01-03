---
sidebar_position: 2
---

# Creating Specialized ROS 2 Nodes

## Overview

In this section, we'll explore how to create specialized types of ROS 2 nodes for humanoid robot applications. We'll cover different node types, their purposes, and implementation patterns.

## Types of Nodes in Humanoid Robotics

### 1. Sensor Nodes
Sensor nodes are responsible for reading data from various sensors on the humanoid robot.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from std_msgs.msg import Header
import numpy as np

class HumanoidSensorNode(Node):
    def __init__(self):
        super().__init__('humanoid_sensor_node')

        # Publishers for different sensor data
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.laser_pub = self.create_publisher(LaserScan, 'scan', 10)

        # Timer for sensor reading
        self.sensor_timer = self.create_timer(0.01, self.read_sensors)  # 100 Hz

        # Initialize sensor data
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist'
        ]
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

    def read_sensors(self):
        """Simulate reading sensor data from hardware"""
        # Update joint positions with simulated data
        for i in range(len(self.joint_positions)):
            self.joint_positions[i] += 0.01 * np.sin(self.get_clock().now().nanoseconds * 1e-9)

        # Publish joint states
        self.publish_joint_states()

        # Publish IMU data
        self.publish_imu_data()

        # Publish laser scan data
        self.publish_laser_scan()

    def publish_joint_states(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts
        self.joint_pub.publish(msg)

    def publish_imu_data(self):
        msg = Imu()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        # Simulate IMU data
        msg.orientation.w = 1.0
        self.imu_pub.publish(msg)

    def publish_laser_scan(self):
        msg = LaserScan()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_link'
        msg.angle_min = -1.57  # -90 degrees
        msg.angle_max = 1.57   # 90 degrees
        msg.angle_increment = 0.01
        msg.range_min = 0.1
        msg.range_max = 10.0
        msg.ranges = [2.0] * 314  # 314 points
        self.laser_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidSensorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Controller Nodes
Controller nodes implement control algorithms for robot joints and actuators.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np

class HumanoidControllerNode(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        self.trajectory_sub = self.create_subscription(
            JointTrajectory, 'joint_trajectory', self.trajectory_callback, 10)

        # Publishers
        self.controller_state_pub = self.create_publisher(
            JointTrajectoryControllerState, 'controller_state', 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.001, self.control_loop)  # 1 kHz

        # Initialize controller state
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.current_joint_efforts = {}
        self.desired_joint_positions = {}
        self.desired_joint_velocities = {}
        self.desired_joint_efforts = {}

        # PID controller parameters
        self.kp = 100.0  # Proportional gain
        self.ki = 0.1    # Integral gain
        self.kd = 10.0   # Derivative gain

    def joint_state_callback(self, msg):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_joint_efforts[name] = msg.effort[i]

    def trajectory_callback(self, msg):
        """Receive trajectory commands"""
        if len(msg.points) > 0:
            # Get the last point in the trajectory
            last_point = msg.points[-1]
            for i, name in enumerate(msg.joint_names):
                if i < len(last_point.positions):
                    self.desired_joint_positions[name] = last_point.positions[i]
                if i < len(last_point.velocities):
                    self.desired_joint_velocities[name] = last_point.velocities[i]
                if i < len(last_point.effort):
                    self.desired_joint_efforts[name] = last_point.effort[i]

    def control_loop(self):
        """Main control loop implementing PID control"""
        control_commands = {}

        for joint_name in self.current_joint_positions:
            current_pos = self.current_joint_positions[joint_name]
            desired_pos = self.desired_joint_positions.get(joint_name, current_pos)

            # Simple PID control
            error = desired_pos - current_pos
            control_output = self.kp * error

            control_commands[joint_name] = control_output

        # Publish controller state
        self.publish_controller_state()

    def publish_controller_state(self):
        msg = JointTrajectoryControllerState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Fill in joint names
        msg.joint_names = list(self.current_joint_positions.keys())

        # Fill in desired states
        for joint_name in msg.joint_names:
            desired_point = JointTrajectoryPoint()
            desired_point.positions = [self.desired_joint_positions.get(joint_name, 0.0)]
            desired_point.velocities = [self.desired_joint_velocities.get(joint_name, 0.0)]
            desired_point.accelerations = [0.0]  # Simplified
            msg.desired.points.append(desired_point)

        self.controller_state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3. Perception Nodes
Perception nodes process sensor data to extract meaningful information.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
import numpy as np

class HumanoidPerceptionNode(Node):
    def __init__(self):
        super().__init__('humanoid_perception')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)

        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 'pointcloud', self.pointcloud_callback, 10)

        # Publishers
        self.object_pub = self.create_publisher(
            PointStamped, 'detected_object', 10)

        # Timer for processing
        self.process_timer = self.create_timer(0.1, self.process_data)  # 10 Hz

        # Data storage
        self.latest_image = None
        self.latest_pointcloud = None
        self.detected_objects = []

    def image_callback(self, msg):
        """Process incoming image data"""
        self.latest_image = msg
        # In a real implementation, you would convert ROS Image to OpenCV format
        # and perform computer vision processing

    def pointcloud_callback(self, msg):
        """Process incoming point cloud data"""
        self.latest_pointcloud = msg
        # In a real implementation, you would process the point cloud data
        # to detect objects, planes, etc.

    def process_data(self):
        """Main processing function"""
        if self.latest_image is not None:
            # Simulate object detection
            if np.random.random() > 0.8:  # 20% chance of detection
                self.detect_object()

    def detect_object(self):
        """Simulate object detection"""
        # Create a detected object at a random position
        obj = PointStamped()
        obj.header = Header()
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.header.frame_id = 'camera_frame'

        # Random position in front of robot
        obj.point.x = 1.0 + np.random.uniform(-0.2, 0.2)
        obj.point.y = np.random.uniform(-0.5, 0.5)
        obj.point.z = np.random.uniform(0.0, 1.0)

        self.detected_objects.append(obj)
        self.object_pub.publish(obj)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Node Composition

For complex humanoid systems, you can compose multiple nodes together:

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import String

class HumanoidSystemNode(Node):
    def __init__(self):
        super().__init__('humanoid_system')

        # Create callback groups for different components
        self.sensors_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        # Publishers
        self.status_pub = self.create_publisher(String, 'system_status', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Timer for system monitoring
        self.system_timer = self.create_timer(
            1.0, self.system_monitor, callback_group=self.sensors_cb_group)

        # Initialize system components
        self.initialize_components()

    def initialize_components(self):
        """Initialize various system components"""
        self.get_logger().info('Initializing humanoid system components...')

        # Initialize sensors
        self.initialize_sensors()

        # Initialize controllers
        self.initialize_controllers()

        # Initialize perception
        self.initialize_perception()

    def initialize_sensors(self):
        """Initialize sensor components"""
        self.get_logger().info('Sensors initialized')

    def initialize_controllers(self):
        """Initialize controller components"""
        self.get_logger().info('Controllers initialized')

    def initialize_perception(self):
        """Initialize perception components"""
        self.get_logger().info('Perception initialized')

    def system_monitor(self):
        """Monitor system status"""
        msg = String()
        msg.data = f'System running at {self.get_clock().now().nanoseconds * 1e-9:.2f}s'
        self.status_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidSystemNode()

    # Create executor and add the node
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Node Configuration and Parameters

Properly configure nodes using parameters:

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile

class ConfigurableHumanoidNode(Node):
    def __init__(self):
        super().__init__('configurable_humanoid')

        # Declare parameters with descriptions and ranges
        self.declare_parameter(
            'robot_name',
            'default_humanoid',
            ParameterDescriptor(description='Name of the humanoid robot')
        )

        self.declare_parameter(
            'control_frequency',
            100,  # Hz
            ParameterDescriptor(
                description='Control loop frequency',
                integer_range=[10, 1000]
            )
        )

        self.declare_parameter(
            'safety_limits.enabled',
            True,
            ParameterDescriptor(description='Enable safety limits')
        )

        self.declare_parameter(
            'safety_limits.max_velocity',
            2.0,
            ParameterDescriptor(
                description='Maximum joint velocity limit',
                floating_point_range=[0.1, 10.0]
            )
        )

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.safety_enabled = self.get_parameter('safety_limits.enabled').value
        self.max_velocity = self.get_parameter('safety_limits.max_velocity').value

        # Set up QoS profiles based on parameters
        qos_depth = self.get_parameter_or('qos_depth', Parameter('qos_depth', Parameter.Type.INTEGER, 10))

        # Create publisher with configurable QoS
        qos_profile = QoSProfile(depth=qos_depth.value)
        self.status_pub = self.create_publisher(String, 'robot_status', qos_profile)

        # Create timer based on frequency parameter
        timer_period = 1.0 / self.control_frequency
        self.control_timer = self.create_timer(timer_period, self.control_callback)

        self.get_logger().info(
            f'Configured {self.robot_name} with control frequency {self.control_frequency}Hz'
        )

    def control_callback(self):
        """Control callback that respects parameters"""
        if self.safety_enabled:
            # Apply safety checks based on parameters
            if self.check_safety_limits():
                self.execute_control()
            else:
                self.get_logger().warn('Safety limits exceeded, stopping control')
        else:
            self.execute_control()

    def check_safety_limits(self):
        """Check if current state respects safety limits"""
        # Implement safety checks based on parameters
        return True

    def execute_control(self):
        """Execute control logic"""
        self.get_logger().info('Executing control command')

def main(args=None):
    rclpy.init(args=args)
    node = ConfigurableHumanoidNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Node Creation

### 1. Proper Error Handling
```python
def create_robust_publisher(self, msg_type, topic_name, qos_profile=10):
    try:
        publisher = self.create_publisher(msg_type, topic_name, qos_profile)
        self.get_logger().info(f'Successfully created publisher for {topic_name}')
        return publisher
    except Exception as e:
        self.get_logger().error(f'Failed to create publisher for {topic_name}: {e}')
        return None
```

### 2. Resource Management
```python
def destroy_node(self):
    # Clean up all created resources
    if hasattr(self, 'publishers'):
        for pub in self.publishers:
            self.destroy_publisher(pub)

    if hasattr(self, 'subscribers'):
        for sub in self.subscribers:
            self.destroy_subscription(sub)

    if hasattr(self, 'timers'):
        for timer in self.timers:
            self.destroy_timer(timer)

    if hasattr(self, 'clients'):
        for client in self.clients:
            self.destroy_client(client)

    if hasattr(self, 'services'):
        for service in self.services:
            self.destroy_service(service)

    return super().destroy_node()
```

### 3. Logging Best Practices
```python
def log_system_event(self, event_type, message, details=None):
    """Consistent logging for system events"""
    if details:
        log_msg = f'[{event_type}] {message} - Details: {details}'
    else:
        log_msg = f'[{event_type}] {message}'

    if event_type in ['ERROR', 'CRITICAL']:
        self.get_logger().error(log_msg)
    elif event_type in ['WARNING', 'WARN']:
        self.get_logger().warn(log_msg)
    else:
        self.get_logger().info(log_msg)
```

## Next Steps

Now that you understand how to create specialized nodes, let's look at publishers and subscribers in more detail, particularly for humanoid robot applications.