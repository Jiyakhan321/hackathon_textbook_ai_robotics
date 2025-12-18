---
sidebar_position: 1
---

# rclpy Basics: Python Client Library for ROS 2

## Introduction to rclpy

rclpy is the Python client library for ROS 2. It provides a Python API to interact with the ROS 2 middleware, allowing you to create nodes, publish and subscribe to topics, provide and call services, and more.

## Installation and Setup

rclpy is typically installed as part of the ROS 2 distribution. Make sure you have sourced your ROS 2 environment:

```bash
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
```

## Basic Node Structure

Here's the minimal structure of a ROS 2 node using rclpy:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node created')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()

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

## Creating Publishers

Publishers send messages to topics. Here's how to create a publisher:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')

        # Create a publisher
        self.publisher = self.create_publisher(
            String,           # Message type
            'topic_name',     # Topic name
            10                # QoS queue size
        )

        # Create a timer to publish periodically
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()

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

## Creating Subscribers

Subscribers receive messages from topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')

        # Create a subscription
        self.subscription = self.create_subscription(
            String,           # Message type
            'topic_name',     # Topic name
            self.listener_callback,  # Callback function
            10                # QoS queue size
        )

        # Prevent unused variable warning
        self.subscription  # type: ignore

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = SubscriberNode()

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

## Creating Services

Services provide request-response communication:

### Service Server
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServerNode(Node):
    def __init__(self):
        super().__init__('service_server_node')

        # Create a service
        self.srv = self.create_service(
            AddTwoInts,           # Service type
            'add_two_ints',       # Service name
            self.add_two_ints_callback  # Callback function
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ServiceServerNode()

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

### Service Client
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceClientNode(Node):
    def __init__(self):
        super().__init__('service_client_node')

        # Create a client
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.request = AddTwoInts.Request()

    def send_request(self, a, b):
        self.request.a = a
        self.request.b = b
        self.future = self.cli.call_async(self.request)
        return self.future

def main(args=None):
    rclpy.init(args=args)
    node = ServiceClientNode()

    # Send request
    future = node.send_request(1, 2)

    try:
        rclpy.spin_until_future_complete(node, future)
        response = future.result()
        node.get_logger().info(f'Result: {response.sum}')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Working with Parameters

Parameters allow configuration of nodes:

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('joint_limits', [1.57, 1.57, 1.57])

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.joint_limits = self.get_parameter('joint_limits').value

        self.get_logger().info(f'Robot: {self.robot_name}, Max velocity: {self.max_velocity}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

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

## Quality of Service (QoS) Settings

QoS settings control how messages are delivered:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class QoSNode(Node):
    def __init__(self):
        super().__init__('qos_node')

        # Create QoS profile for reliable delivery
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.publisher = self.create_publisher(String, 'reliable_topic', qos_profile)
        self.subscription = self.create_subscription(
            String, 'reliable_topic', self.listener_callback, qos_profile
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = QoSNode()

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

## Working with Time

ROS 2 provides various time utilities:

```python
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from rclpy.time import Time as RclpyTime

class TimeNode(Node):
    def __init__(self):
        super().__init__('time_node')

        # Get current ROS time
        current_time = self.get_clock().now()
        self.get_logger().info(f'Current time: {current_time}')

        # Create timer
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # Get time when callback is executed
        callback_time = self.get_clock().now()
        self.get_logger().info(f'Timer callback at: {callback_time}')

def main(args=None):
    rclpy.init(args=args)
    node = TimeNode()

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

## Best Practices

### 1. Proper Resource Management
```python
import rclpy
from rclpy.node import Node

class BestPracticeNode(Node):
    def __init__(self):
        super().__init__('best_practice_node')
        self.publishers = []
        self.subscribers = []
        self.timers = []

        # Create resources
        pub = self.create_publisher(String, 'topic', 10)
        self.publishers.append(pub)

        timer = self.create_timer(1.0, self.timer_callback)
        self.timers.append(timer)

    def timer_callback(self):
        self.get_logger().info('Timer callback')

    def destroy_node(self):
        # Clean up resources explicitly
        for pub in self.publishers:
            self.destroy_publisher(pub)
        for sub in self.subscribers:
            self.destroy_subscription(sub)
        for timer in self.timers:
            self.destroy_timer(timer)

        return super().destroy_node()
```

### 2. Error Handling
```python
def safe_publisher(node, msg_type, topic_name, queue_size=10):
    try:
        publisher = node.create_publisher(msg_type, topic_name, queue_size)
        return publisher
    except Exception as e:
        node.get_logger().error(f'Failed to create publisher: {e}')
        return None
```

### 3. Logging
```python
def log_with_context(node, level, message, context=None):
    if context:
        message = f'[{context}] {message}'
    node.get_logger().log(level, message)
```

## Common Patterns for Humanoid Robots

### Joint State Publisher
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.05, self.publish_joint_states)  # 20 Hz

        # Initialize joint names for humanoid
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint'
        ]

        # Initialize joint positions
        self.joint_positions = [0.0] * len(self.joint_names)

    def publish_joint_states(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions

        # Set timestamp
        time = self.get_clock().now().to_msg()
        msg.header.stamp = time
        msg.header.frame_id = 'base_link'

        self.publisher.publish(msg)

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

## Next Steps

Now that you understand rclpy basics, let's look at how to create specific types of nodes for humanoid robot control.