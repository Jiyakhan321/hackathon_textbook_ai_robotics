---
sidebar_position: 1
---

# Nodes, Topics, and Services

## Understanding ROS 2 Architecture

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's not an operating system but rather a collection of tools, libraries, and conventions that aim to simplify the task of creating complex robotic applications.

## Core Concepts

### Nodes
A node is a process that performs computation. In ROS 2, nodes are designed to be as lightweight as possible. Multiple nodes are typically used to implement a robot application, each performing a specific task.

### Topics
Topics enable asynchronous message passing between nodes. Nodes can publish messages to a topic or subscribe to a topic to receive messages. This creates a publish-subscribe communication pattern.

### Services
Services provide synchronous request-response communication between nodes. A service client sends a request to a service server and waits for a response.

## Creating Your First ROS 2 Node

Let's create a simple ROS 2 node that publishes a message:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Understanding Topics

Topics are named buses over which nodes exchange messages. The name is a unique identifier within the ROS graph. Topics use a publish-subscribe communication pattern where multiple nodes can publish to the same topic and multiple nodes can subscribe to the same topic.

### Publisher-Subscriber Pattern
- Publishers send data to a topic without knowledge of subscribers
- Subscribers receive data from a topic without knowledge of publishers
- This decouples the components and allows for flexible system design

## Services for Request-Response

Services provide a request-response communication pattern. When a client calls a service, it sends a request and waits for a response from the service server.

Example service client:
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' %
        (1, 2, response.sum))
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

1. **Node Design**: Keep nodes focused on a single responsibility
2. **Topic Naming**: Use descriptive, consistent names with forward slashes for hierarchy
3. **Message Types**: Use standard message types when possible to ensure compatibility
4. **Error Handling**: Implement proper error handling and logging in your nodes
5. **Resource Management**: Always clean up resources when nodes are destroyed

## Next Steps

Now that you understand the core concepts of ROS 2 architecture, let's dive deeper into message types and how they enable communication between nodes.