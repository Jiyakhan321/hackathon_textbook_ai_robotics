---
sidebar_position: 2
---

# Message Types and Definitions

## Understanding ROS 2 Messages

Messages are the data structures that are passed between nodes in ROS 2. They define the format of data that can be exchanged through topics, services, and actions. Messages are defined using a special interface definition language (IDL) and are compiled into language-specific code.

## Standard Message Types

ROS 2 provides a rich set of standard message types in the `std_msgs` package:

### Basic Types
- `std_msgs/msg/Bool` - Boolean value
- `std_msgs/msg/Int32` - 32-bit integer
- `std_msgs/msg/Float64` - 64-bit floating point
- `std_msgs/msg/String` - String of characters
- `std_msgs/msg/Header` - Standard metadata for messages

### Sensor Types
- `sensor_msgs/msg/JointState` - State of joints
- `sensor_msgs/msg/LaserScan` - Laser range finder data
- `sensor_msgs/msg/Image` - Image data
- `sensor_msgs/msg/CameraInfo` - Camera calibration information
- `sensor_msgs/msg/Imu` - Inertial measurement unit data

### Geometry Types
- `geometry_msgs/msg/Twist` - Linear and angular velocities
- `geometry_msgs/msg/Pose` - Position and orientation
- `geometry_msgs/msg/Point` - 3D point
- `geometry_msgs/msg/Quaternion` - 4D quaternion

## Creating Custom Messages

For humanoid robots, you'll often need custom message types. Custom messages are defined in `.msg` files:

Example custom message for humanoid joint states:
```
# HumanoidJointState.msg
string[] joint_names
float64[] positions
float64[] velocities
float64[] efforts
uint8[] joint_types
---
# Additional metadata
builtin_interfaces/Time timestamp
string robot_name
```

To create a custom message:
1. Create a `msg` directory in your package
2. Define your message in a `.msg` file
3. Update `CMakeLists.txt` to include the message
4. Build the package with `colcon build`

## Message Definition Syntax

### Basic Types
- `bool` - Boolean (true/false)
- `byte` - 8-bit unsigned integer
- `char` - 8-bit unsigned integer
- `float32` - 32-bit floating point
- `float64` - 64-bit floating point
- `int8`, `int16`, `int32`, `int64` - Signed integers
- `uint8`, `uint16`, `uint32`, `uint64` - Unsigned integers
- `string` - Variable-length string
- `wstring` - Wide variable-length string

### Arrays
- `type[]` - Unbounded array
- `type[n]` - Fixed-size array of n elements

### Constants
You can define constants in message definitions:
```
int32 RED=1
int32 GREEN=2
int32 BLUE=3
int32 color
```

## Working with Messages in Python

### Using Standard Messages
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class MessageExample(Node):
    def __init__(self):
        super().__init__('message_example')
        self.publisher = self.create_publisher(String, 'chatter', 10)

    def publish_message(self):
        msg = String()
        msg.data = 'Hello from ROS 2'
        self.publisher.publish(msg)
```

### Creating Joint State Messages for Humanoid Robots
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)

    def publish_joint_state(self):
        msg = JointState()
        msg.name = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint'
        ]
        msg.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)

        # Set timestamp
        time = self.get_clock().now().to_msg()
        msg.header.stamp = time
        msg.header.frame_id = 'base_link'

        self.joint_publisher.publish(msg)
```

## Message Serialization and Performance

Messages are serialized when transmitted between nodes. For humanoid robots with many joints and sensors, consider:

1. **Efficient Data Types**: Use appropriate data types (e.g., float32 instead of float64 when precision allows)
2. **Message Frequency**: Optimize publishing rates for your application
3. **Data Compression**: For large data like images, consider compression
4. **Message Size**: Be mindful of message size to avoid network congestion

## Best Practices

1. **Use Standard Types**: Leverage existing standard message types when possible
2. **Descriptive Names**: Use clear, descriptive names for custom message fields
3. **Documentation**: Document the meaning and units of custom message fields
4. **Backward Compatibility**: Design messages to be forward and backward compatible
5. **Validation**: Validate message content in your nodes to ensure data integrity

## Next Steps

Now that you understand message types, let's explore lifecycle nodes which provide more sophisticated node management capabilities.