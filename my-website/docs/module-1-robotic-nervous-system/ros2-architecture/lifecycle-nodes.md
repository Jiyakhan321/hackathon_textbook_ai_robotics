---
sidebar_position: 3
---

# Lifecycle Nodes

## Introduction to Lifecycle Nodes

Lifecycle nodes provide a structured way to manage the state of ROS 2 nodes. Unlike regular nodes that start and run continuously, lifecycle nodes have well-defined states and transitions, making them ideal for complex robotic systems where components need to be initialized, activated, deactivated, and cleaned up in a controlled manner.

## Lifecycle Node States

A lifecycle node can be in one of the following states:

1. **Unconfigured** - Initial state after creation
2. **Inactive** - After configuration but before activation
3. **Active** - Running and participating in computation
4. **Finalized** - After cleanup, cannot return to other states
5. **Error Processing** - When an error occurs during state transition

## State Transition Diagram

```
[Unconfigured] <--+---> [Inactive] -----> [Active]
     ^                |         |            |
     |                |         |            |
     |                v         v            |
     +--------- [Finalized] <- [Error Process]
```

## Creating a Lifecycle Node

Here's an example of a lifecycle node for humanoid robot control:

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import State
from rclpy.qos import QoSProfile

class HumanoidLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('humanoid_lifecycle_node')
        self.get_logger().info('Lifecycle node initialized in unconfigured state')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring humanoid controller...')

        # Create publishers, subscribers, and parameters
        self.joint_publisher = self.create_publisher(
            'sensor_msgs/msg/JointState',
            'joint_states',
            QoSProfile(depth=10)
        )

        # Initialize hardware interfaces
        self.initialize_hardware()

        self.get_logger().info('Configuration complete')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating humanoid controller...')

        # Activate publishers and subscribers
        self.joint_publisher.on_activate()

        # Start control loops
        self.start_control_loops()

        self.get_logger().info('Activation complete')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating humanoid controller...')

        # Deactivate publishers and subscribers
        self.joint_publisher.on_deactivate()

        # Stop control loops
        self.stop_control_loops()

        self.get_logger().info('Deactivation complete')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up humanoid controller...')

        # Destroy publishers, subscribers, and parameters
        self.destroy_publisher(self.joint_publisher)

        # Clean up hardware interfaces
        self.cleanup_hardware()

        self.get_logger().info('Cleanup complete')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down humanoid controller...')
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Error occurred in humanoid controller')
        return TransitionCallbackReturn.SUCCESS

    def initialize_hardware(self):
        # Initialize robot hardware interfaces
        self.get_logger().info('Initializing hardware interfaces...')
        # Add your hardware initialization code here

    def start_control_loops(self):
        # Start control loops for humanoid robot
        self.get_logger().info('Starting control loops...')
        # Add your control loop start code here

    def stop_control_loops(self):
        # Stop control loops for humanoid robot
        self.get_logger().info('Stopping control loops...')
        # Add your control loop stop code here

    def cleanup_hardware(self):
        # Clean up hardware interfaces
        self.get_logger().info('Cleaning up hardware interfaces...')
        # Add your hardware cleanup code here

def main(args=None):
    rclpy.init(args=args)

    node = HumanoidLifecycleNode()

    # Transition through states manually for demonstration
    node.trigger_configure()
    node.trigger_activate()

    # Run for a while
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.trigger_deactivate()
        node.trigger_cleanup()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Using Lifecycle Manager

ROS 2 provides a lifecycle manager to control multiple lifecycle nodes:

```python
import rclpy
from rclpy.lifecycle import LifecycleManager
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

class HumanoidSystemManager(Node):
    def __init__(self):
        super().__init__('humanoid_system_manager')

        # Initialize lifecycle manager
        self.manager = LifecycleManager('/humanoid_system_manager')

        # Set the node names that the manager will control
        self.manager.set_parameters([
            ('node_names', ['humanoid_controller', 'sensor_processor', 'motion_planner'])
        ])

    def initialize_system(self):
        # Configure all nodes
        self.manager.configure()

        # Activate all nodes
        self.manager.activate()

    def shutdown_system(self):
        # Deactivate all nodes
        self.manager.deactivate()

        # Cleanup all nodes
        self.manager.cleanup()

def main(args=None):
    rclpy.init(args=args)

    manager_node = HumanoidSystemManager()

    try:
        manager_node.initialize_system()
        rclpy.spin(manager_node)
    except KeyboardInterrupt:
        pass
    finally:
        manager_node.shutdown_system()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Benefits of Lifecycle Nodes

### 1. Controlled Startup
- Ensures components are initialized in the correct order
- Allows for dependency management between nodes
- Provides clear feedback on startup progress

### 2. Safe Operation
- Prevents premature activation of components
- Enables graceful handling of initialization failures
- Supports runtime reconfiguration

### 3. Resource Management
- Proper allocation and deallocation of resources
- Controlled activation/deactivation of expensive operations
- Better memory management

### 4. Debugging and Monitoring
- Clear state information for debugging
- Better observability of system status
- Easier fault isolation

## Best Practices

1. **State Validation**: Always validate preconditions in state transition callbacks
2. **Error Handling**: Implement proper error handling in all transition callbacks
3. **Resource Cleanup**: Ensure all resources are properly cleaned up in cleanup transitions
4. **Dependency Management**: Use lifecycle managers for coordinating multiple nodes
5. **Logging**: Provide clear logging messages for each state transition
6. **Timeout Handling**: Implement appropriate timeouts for state transitions
7. **Graceful Degradation**: Design nodes to handle partial failures gracefully

## Common Patterns

### Pattern 1: Sensor Node
- Configure: Initialize sensor hardware and parameters
- Activate: Start data acquisition
- Deactivate: Stop data acquisition
- Cleanup: Close sensor connections

### Pattern 2: Controller Node
- Configure: Load control parameters and initialize controllers
- Activate: Start control loops
- Deactivate: Stop control loops and set safe positions
- Cleanup: Release control resources

### Pattern 3: Processing Node
- Configure: Initialize processing algorithms and parameters
- Activate: Start processing pipeline
- Deactivate: Pause processing
- Cleanup: Release processing resources

## Next Steps

Lifecycle nodes are essential for building robust robotic systems. In the next section, we'll explore how to create Python agents using rclpy to implement these concepts in practice.