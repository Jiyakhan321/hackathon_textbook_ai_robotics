---
sidebar_position: 3
---

# Publishers and Subscribers in Humanoid Robotics

## Overview

Publishers and subscribers form the backbone of ROS 2 communication, enabling the asynchronous exchange of information between different components of a humanoid robot system. In this section, we'll explore how to effectively use publishers and subscribers for various humanoid robot applications.

## Publisher-Subscriber Pattern in Humanoid Systems

The publisher-subscriber pattern is particularly useful in humanoid robotics because it allows for:
- Decoupling of components (sensors, controllers, perception systems)
- Flexible system architecture
- Real-time data exchange
- Multiple consumers of the same data

## Basic Publisher Implementation

Let's start with a basic joint state publisher for humanoid robots:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create publisher for joint states
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Timer for publishing joint states at regular intervals
        self.timer = self.create_timer(0.05, self.publish_joint_states)  # 20 Hz

        # Initialize joint information
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint'
        ]

        # Initialize positions with zero values
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

        self.get_logger().info('Joint state publisher initialized')

    def publish_joint_states(self):
        """Publish joint state information"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = self.update_joint_positions()
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        self.publisher.publish(msg)
        self.get_logger().debug(f'Published joint states: {len(msg.name)} joints')

    def update_joint_positions(self):
        """Update joint positions based on time for demonstration"""
        current_time = self.get_clock().now().nanoseconds * 1e-9  # Convert to seconds

        # Create a simple oscillating pattern for demonstration
        for i in range(len(self.joint_positions)):
            # Different oscillation patterns for different joints
            amplitude = 0.5 if i % 2 == 0 else 0.3
            frequency = 0.5 + (i % 3) * 0.2
            self.joint_positions[i] = amplitude * math.sin(frequency * current_time + i)

        return self.joint_positions

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()

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

## Advanced Publisher with Multiple Topics

For humanoid robots, you often need to publish to multiple topics simultaneously:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Temperature
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np

class MultiTopicPublisher(Node):
    def __init__(self):
        super().__init__('multi_topic_publisher')

        # Multiple publishers for different data types
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.temperature_pub = self.create_publisher(Temperature, 'temperature', 10)
        self.motor_current_pub = self.create_publisher(Float64MultiArray, 'motor_currents', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Timer for coordinated publishing
        self.publish_timer = self.create_timer(0.01, self.publish_all_data)  # 100 Hz

        # Initialize data
        self.joint_names = [f'joint_{i}' for i in range(12)]
        self.time_offset = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info('Multi-topic publisher initialized')

    def publish_all_data(self):
        """Publish all types of data in a coordinated manner"""
        current_time = self.get_clock().now().nanoseconds * 1e-9

        # Publish joint states
        self.publish_joint_states(current_time)

        # Publish IMU data
        self.publish_imu_data(current_time)

        # Publish temperature data
        self.publish_temperature_data(current_time)

        # Publish motor currents
        self.publish_motor_currents(current_time)

        # Check and publish emergency stop if needed
        self.check_and_publish_emergency_stop(current_time)

    def publish_joint_states(self, current_time):
        """Publish joint state information"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names

        # Generate time-varying joint positions
        positions = [0.5 * np.sin(2 * np.pi * 0.5 * (current_time - self.time_offset) + i)
                     for i in range(len(self.joint_names))]
        msg.position = positions

        velocities = [np.cos(2 * np.pi * 0.5 * (current_time - self.time_offset) + i)
                      for i in range(len(self.joint_names))]
        msg.velocity = velocities

        efforts = [0.1 * np.sin(4 * np.pi * 0.5 * (current_time - self.time_offset) + i)
                   for i in range(len(self.joint_names))]
        msg.effort = efforts

        self.joint_pub.publish(msg)

    def publish_imu_data(self, current_time):
        """Publish IMU sensor data"""
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'

        # Simulate IMU orientation (simplified)
        msg.orientation.w = np.cos(0.1 * current_time)
        msg.orientation.x = np.sin(0.1 * current_time) * 0.1
        msg.orientation.y = np.sin(0.1 * current_time) * 0.05
        msg.orientation.z = 0.0

        # Simulate angular velocity
        msg.angular_velocity.x = 0.1 * np.cos(0.1 * current_time)
        msg.angular_velocity.y = 0.05 * np.sin(0.1 * current_time)
        msg.angular_velocity.z = 0.02 * np.sin(0.2 * current_time)

        # Simulate linear acceleration
        msg.linear_acceleration.x = 9.81 * np.sin(0.1 * current_time)
        msg.linear_acceleration.y = 0.1 * np.cos(0.1 * current_time)
        msg.linear_acceleration.z = 9.81 * np.cos(0.1 * current_time)

        self.imu_pub.publish(msg)

    def publish_temperature_data(self, current_time):
        """Publish temperature sensor data"""
        msg = Temperature()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'thermal_sensor'
        msg.temperature = 25.0 + 5.0 * np.sin(0.05 * current_time)  # Varying temperature
        msg.variance = 0.1  # Small variance

        self.temperature_pub.publish(msg)

    def publish_motor_currents(self, current_time):
        """Publish motor current information"""
        msg = Float64MultiArray()
        # Simulate motor currents for each joint
        currents = [0.5 + 0.2 * np.sin(0.2 * current_time + i)
                    for i in range(len(self.joint_names))]
        msg.data = currents

        self.motor_current_pub.publish(msg)

    def check_and_publish_emergency_stop(self, current_time):
        """Check conditions and publish emergency stop if needed"""
        # Simulate emergency stop based on some condition
        emergency = False  # In real system, check actual conditions

        msg = Bool()
        msg.data = emergency
        self.emergency_stop_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiTopicPublisher()

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

## Basic Subscriber Implementation

Now let's look at how to implement subscribers that can handle data from humanoid robot systems:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')

        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Store latest joint state
        self.latest_joint_state = None
        self.joint_state_received = False

        # Timer to process data periodically
        self.process_timer = self.create_timer(1.0, self.process_joint_data)

        self.get_logger().info('Joint state subscriber initialized')

    def joint_state_callback(self, msg):
        """Callback for receiving joint state messages"""
        self.latest_joint_state = msg
        self.joint_state_received = True

        # Log received data
        self.get_logger().debug(f'Received joint state with {len(msg.name)} joints')

    def process_joint_data(self):
        """Process joint data periodically"""
        if self.joint_state_received and self.latest_joint_state:
            # Analyze joint positions
            avg_position = np.mean(self.latest_joint_state.position)
            max_position = np.max(np.abs(self.latest_joint_state.position))

            self.get_logger().info(
                f'Joint state analysis - Avg pos: {avg_position:.3f}, Max abs pos: {max_position:.3f}'
            )

def main(args=None):
    rclpy.init(args=args)
    node = JointStateSubscriber()

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

## Advanced Multi-Topic Subscriber

For humanoid robots, you often need to subscribe to multiple topics and correlate the data:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
from collections import deque
import time

class HumanoidDataProcessor(Node):
    def __init__(self):
        super().__init__('humanoid_data_processor')

        # Multiple subscriptions
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        self.motor_current_sub = self.create_subscription(
            Float64MultiArray, 'motor_currents', self.motor_current_callback, 10)

        self.emergency_stop_sub = self.create_subscription(
            Bool, 'emergency_stop', self.emergency_stop_callback, 10)

        # Data storage with time stamps
        self.joint_buffer = deque(maxlen=100)  # Store last 100 joint states
        self.imu_buffer = deque(maxlen=100)    # Store last 100 IMU readings
        self.current_buffer = deque(maxlen=100) # Store last 100 current readings

        # State variables
        self.emergency_stop_active = False
        self.last_joint_update = 0
        self.last_imu_update = 0

        # Processing timer
        self.process_timer = self.create_timer(0.1, self.process_sensor_data)  # 10 Hz

        self.get_logger().info('Humanoid data processor initialized')

    def joint_callback(self, msg):
        """Handle joint state messages"""
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        self.joint_buffer.append({
            'timestamp': timestamp,
            'msg': msg
        })
        self.last_joint_update = timestamp

    def imu_callback(self, msg):
        """Handle IMU messages"""
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        self.imu_buffer.append({
            'timestamp': timestamp,
            'msg': msg
        })
        self.last_imu_update = timestamp

    def motor_current_callback(self, msg):
        """Handle motor current messages"""
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        self.current_buffer.append({
            'timestamp': timestamp,
            'data': msg.data
        })

    def emergency_stop_callback(self, msg):
        """Handle emergency stop messages"""
        self.emergency_stop_active = msg.data
        if self.emergency_stop_active:
            self.get_logger().warn('EMERGENCY STOP ACTIVATED!')

    def process_sensor_data(self):
        """Process all sensor data and extract meaningful information"""
        if not self.joint_buffer or not self.imu_buffer:
            return

        # Get latest data
        latest_joint = self.joint_buffer[-1]
        latest_imu = self.imu_buffer[-1]

        # Calculate joint velocity estimates from position changes
        if len(self.joint_buffer) > 1:
            prev_joint = self.joint_buffer[-2]
            dt = latest_joint['timestamp'] - prev_joint['timestamp']
            if dt > 0:
                position_diff = np.array(latest_joint['msg'].position) - np.array(prev_joint['msg'].position)
                estimated_velocities = position_diff / dt

                # Log max velocity
                max_vel = np.max(np.abs(estimated_velocities))
                if max_vel > 10.0:  # Threshold for logging
                    self.get_logger().info(f'High joint velocity detected: {max_vel:.3f} rad/s')

        # Analyze IMU data for stability
        imu_msg = latest_imu['msg']
        linear_accel = np.sqrt(
            imu_msg.linear_acceleration.x**2 +
            imu_msg.linear_acceleration.y**2 +
            imu_msg.linear_acceleration.z**2
        )

        # Check if robot is stable based on IMU
        if abs(linear_accel - 9.81) > 2.0:  # Deviation from gravity
            self.get_logger().info(f'Potential instability detected: accel={linear_accel:.2f}')

        # Analyze motor currents for anomalies
        if self.current_buffer:
            latest_currents = self.current_buffer[-1]['data']
            avg_current = np.mean(latest_currents)
            max_current = np.max(latest_currents)

            if max_current > 5.0:  # High current threshold
                self.get_logger().warn(f'High motor current detected: {max_current:.2f}A')

        # Check for data staleness
        current_time = self.get_clock().now().nanoseconds * 1e-9
        if current_time - self.last_joint_update > 1.0:
            self.get_logger().warn('Joint state data is stale (>1 second)')
        if current_time - self.last_imu_update > 1.0:
            self.get_logger().warn('IMU data is stale (>1 second)')

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidDataProcessor()

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

## Quality of Service (QoS) Considerations

For humanoid robotics applications, QoS settings are crucial for reliable communication:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class QoSHumanoidNode(Node):
    def __init__(self):
        super().__init__('qos_humanoid_node')

        # Define different QoS profiles for different data types

        # Joint states: need reliability and some history
        joint_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # IMU data: need all messages for control
        imu_qos = QoSProfile(
            depth=50,  # Keep more IMU messages
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_ALL  # Keep all IMU messages for control
        )

        # Status messages: can be best effort
        status_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Create publishers with appropriate QoS
        self.joint_pub = self.create_publisher(JointState, 'joint_states', joint_qos)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', imu_qos)
        self.status_pub = self.create_publisher(String, 'status', status_qos)

        # Create subscriptions with matching QoS
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, joint_qos)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, imu_qos)

        self.get_logger().info('QoS-configured humanoid node initialized')

    def joint_callback(self, msg):
        """Handle joint state messages with appropriate QoS"""
        self.get_logger().debug(f'Received joint state with QoS')

    def imu_callback(self, msg):
        """Handle IMU messages with appropriate QoS"""
        self.get_logger().debug(f'Received IMU data with QoS')

def main(args=None):
    rclpy.init(args=args)
    node = QoSHumanoidNode()

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

## Publisher-Subscriber Best Practices

### 1. Proper Resource Management
```python
class BestPracticeNode(Node):
    def __init__(self):
        super().__init__('best_practice_node')

        # Store references to all publishers/subscribers
        self.publishers = {}
        self.subscribers = {}

        # Create publishers
        self.publishers['joint_states'] = self.create_publisher(JointState, 'joint_states', 10)
        self.publishers['imu_data'] = self.create_publisher(Imu, 'imu/data', 10)

        # Create subscribers
        self.subscribers['commands'] = self.create_subscription(
            JointTrajectory, 'joint_trajectory', self.command_callback, 10)

    def destroy_node(self):
        """Properly clean up all resources"""
        for pub in self.publishers.values():
            self.destroy_publisher(pub)

        for sub in self.subscribers.values():
            self.destroy_subscription(sub)

        return super().destroy_node()
```

### 2. Data Validation
```python
def validate_joint_state(self, msg):
    """Validate joint state message before processing"""
    if len(msg.name) != len(msg.position):
        self.get_logger().error('Joint names and positions length mismatch')
        return False

    if len(msg.position) != len(msg.velocity) and len(msg.velocity) > 0:
        self.get_logger().warn('Position-velocity length mismatch')

    # Check for invalid values
    for pos in msg.position:
        if not (-100 < pos < 100):  # Reasonable joint limit
            self.get_logger().warn(f'Unusual joint position: {pos}')
            return False

    return True
```

### 3. Error Handling
```python
def safe_publish(self, publisher, msg):
    """Safely publish a message with error handling"""
    try:
        publisher.publish(msg)
        return True
    except Exception as e:
        self.get_logger().error(f'Failed to publish message: {e}')
        return False
```

## Next Steps

Now that you understand publishers and subscribers, let's move on to URDF modeling for humanoid robots, which defines the physical structure that these communication systems will control.