---
title: Module 1 Project
sidebar_position: 1
---


# Module 1 Project: Building Your First Humanoid Robot Control System

## Project Overview

In this capstone project for Module 1, you'll integrate all the concepts learned to create a complete humanoid robot control system using ROS 2. You'll design a robot model, implement communication between different components, and create a control interface.

## Project Objectives

By completing this project, you will demonstrate your ability to:
- Create a complete humanoid robot URDF model
- Implement publisher-subscriber communication patterns
- Design and implement ROS 2 services for robot control
- Integrate multiple ROS 2 nodes into a cohesive system
- Validate and test your robot system

## Project Requirements

### 1. Robot Model (URDF)
Create a complete humanoid robot model with:
- Minimum 12 actuated joints (6 per leg, 4 per arm, 2 for torso/neck)
- Proper inertial properties for each link
- Visual and collision geometries
- Gazebo integration tags

### 2. Communication System
Implement the following ROS 2 communication patterns:
- **Publishers**: Joint state publisher with realistic movement patterns
- **Subscribers**: Joint state subscriber that validates and logs data
- **Services**: Robot control service with multiple movement patterns

### 3. Control Interface
Create a user-friendly control system that allows:
- Manual control of individual joints
- Execution of predefined movements (wave, walk, sit, stand)
- Smooth trajectory execution
- Safety limits enforcement

## Project Structure

Your project should include the following components:

```
humanoid_robot_project/
├── urdf/
│   └── humanoid_model.urdf
├── launch/
│   └── humanoid_system.launch.py
├── config/
│   └── humanoid_control.yaml
├── scripts/
│   ├── robot_model_publisher.py
│   ├── joint_state_subscriber.py
│   ├── control_service_server.py
│   ├── control_service_client.py
│   └── validation_script.py
├── rviz/
│   └── humanoid_view.rviz
└── test/
    └── integration_test.py
```

## Implementation Steps

### Step 1: Design Your Robot Model (Week 1)

Create your humanoid robot URDF with realistic dimensions and joint limits:

```xml
<!-- Example starter for your humanoid_model.urdf -->
<?xml version="1.0"?>
<robot name="student_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.4 0.4 0.4 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base/Pelvis Link -->
  <link name="base_link">
    <inertial>
      <mass value="12.0"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia ixx="0.25" ixy="0" ixz="0" iyy="0.25" iyz="0" izz="0.2"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.25 0.1"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.25 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Add your complete robot structure here -->
  <!-- Include all necessary joints, links, and properties -->

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/student_humanoid</robotNamespace>
    </plugin>
  </gazebo>

</robot>
```

### Step 2: Create the Robot State Publisher (Week 1)

Create a publisher that simulates realistic robot movements:

```python
# robot_model_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class RobotModelPublisher(Node):
    def __init__(self):
        super().__init__('robot_model_publisher')

        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.02, self.publish_joint_states)  # 50 Hz

        # Define your robot's joint names
        self.joint_names = [
            # Fill in your robot's joint names
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint',
            'torso_joint', 'neck_joint'
        ]

        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

        self.time_counter = 0.0

        self.get_logger().info(f'Robot Model Publisher started with {len(self.joint_names)} joints')

    def publish_joint_states(self):
        """Publish joint state messages with realistic movement patterns"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Update joint positions based on time
        self.time_counter += 0.02

        # Implement realistic movement patterns
        self.update_joint_positions()

        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        self.publisher.publish(msg)

    def update_joint_positions(self):
        """Update joint positions with realistic patterns"""
        # Example: Breathing motion for torso
        torso_motion = 0.05 * math.sin(0.2 * self.time_counter)

        # Example: Slight sway for balance
        balance_sway = 0.02 * math.sin(0.3 * self.time_counter)

        # Update your joints based on the patterns you design
        # This is where you implement your robot's unique characteristics

        # For example, if you have a torso joint:
        if 'torso_joint' in self.joint_names:
            idx = self.joint_names.index('torso_joint')
            self.joint_positions[idx] = torso_motion

        # Add your own patterns for each joint

def main(args=None):
    rclpy.init(args=args)

    publisher = RobotModelPublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Shutting down robot model publisher')
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Implement the Joint State Subscriber (Week 2)

Create a subscriber that validates and processes joint states:

```python
# joint_state_subscriber.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')

        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.previous_positions = None
        self.velocity_calculated = []

        self.get_logger().info('Joint State Subscriber initialized')

    def joint_state_callback(self, msg):
        """Process incoming joint state messages"""
        self.get_logger().debug(f'Received joint state with {len(msg.name)} joints')

        # Validate message
        if not self.validate_joint_state(msg):
            self.get_logger().warn('Invalid joint state received')
            return

        # Calculate velocities if not provided
        if not msg.velocity and self.previous_positions:
            calculated_velocities = self.calculate_velocities(msg)
            self.get_logger().info(f'Max calculated velocity: {max(calculated_velocities):.3f}')

        # Store positions for next velocity calculation
        self.previous_positions = list(msg.position)

        # Log statistics
        self.log_joint_statistics(msg)

    def validate_joint_state(self, msg):
        """Validate the joint state message"""
        if len(msg.name) != len(msg.position):
            self.get_logger().error('Joint names and positions length mismatch')
            return False

        if msg.velocity and len(msg.name) != len(msg.velocity):
            self.get_logger().error('Joint names and velocities length mismatch')
            return False

        if msg.effort and len(msg.name) != len(msg.effort):
            self.get_logger().error('Joint names and efforts length mismatch')
            return False

        # Check for reasonable position values
        for pos in msg.position:
            if abs(pos) > 100:  # 100 radians is unreasonable
                self.get_logger().warn(f'Unreasonable joint position: {pos}')

        return True

    def calculate_velocities(self, msg):
        """Calculate velocities from position changes"""
        if not self.previous_positions or len(self.previous_positions) != len(msg.position):
            return [0.0] * len(msg.position)

        # Calculate velocities (assuming constant time interval)
        time_interval = 0.02  # From publisher timer
        velocities = []

        for i in range(len(msg.position)):
            velocity = (msg.position[i] - self.previous_positions[i]) / time_interval
            velocities.append(velocity)

        return velocities

    def log_joint_statistics(self, msg):
        """Log statistics about the joint states"""
        if msg.position:
            avg_pos = np.mean(msg.position)
            max_pos = np.max(np.abs(msg.position))

            self.get_logger().info(
                f'Joint stats - Avg pos: {avg_pos:.3f}, Max abs pos: {max_pos:.3f}'
            )

def main(args=None):
    rclpy.init(args=args)

    subscriber = JointStateSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        subscriber.get_logger().info('Shutting down joint state subscriber')
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Create the Control Service (Week 2-3)

Implement a service for robot control with multiple movement patterns:

```python
# control_service_server.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time

# Define your service messages here or import them
# For now, we'll create simple class-based message structures

class SetJointPositionsRequest:
    def __init__(self):
        self.joint_names = []
        self.positions = []
        self.duration = 0.0

class SetJointPositionsResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.achieved_positions = []

class ExecuteMovementRequest:
    def __init__(self):
        self.movement_type = ""
        self.speed = 1.0

class ExecuteMovementResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.executed_movement = ""

class ControlServiceServer(Node):
    def __init__(self):
        super().__init__('control_service_server')

        # Publisher for joint states
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Create your services here
        # For this example, we'll simulate the service functionality
        self.service_timer = self.create_timer(0.02, self.service_timer_callback)

        # Robot state
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint',
            'torso_joint', 'neck_joint'
        ]

        self.current_positions = [0.0] * len(self.joint_names)
        self.target_positions = [0.0] * len(self.joint_names)
        self.moving = False
        self.movement_start_time = 0.0
        self.movement_duration = 0.0

        # Command queue for movements
        self.command_queue = []

        self.get_logger().info('Control Service Server initialized')

    def service_timer_callback(self):
        """Main timer callback to handle movements"""
        if self.moving:
            # Handle smooth movement interpolation
            current_time = self.get_clock().now().nanoseconds * 1e-9
            elapsed = current_time - self.movement_start_time

            if elapsed >= self.movement_duration:
                # Movement completed
                self.current_positions = self.target_positions.copy()
                self.moving = False
                self.get_logger().info('Movement completed')
            else:
                # Interpolate positions
                progress = elapsed / self.movement_duration
                # Use smooth interpolation
                smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))

                for i in range(len(self.current_positions)):
                    start_pos = self.start_positions[i] if hasattr(self, 'start_positions') else 0.0
                    self.current_positions[i] = (
                        start_pos + smooth_progress * (self.target_positions[i] - start_pos)
                    )

        # Execute commands from queue
        if self.command_queue:
            cmd = self.command_queue.pop(0)
            self.execute_command(cmd)

        # Publish current joint states
        self.publish_joint_states()

    def execute_command(self, command):
        """Execute a command from the queue"""
        if command['type'] == 'set_positions':
            self.move_to_positions(command['joint_names'], command['positions'], command['duration'])
        elif command['type'] == 'movement':
            self.execute_movement_pattern(command['movement_type'], command['speed'])

    def move_to_positions(self, joint_names, positions, duration):
        """Move specified joints to target positions"""
        if len(joint_names) != len(positions):
            self.get_logger().error('Joint names and positions length mismatch')
            return False

        # Update target positions
        for i, joint_name in enumerate(joint_names):
            if joint_name in self.joint_names:
                idx = self.joint_names.index(joint_name)
                self.target_positions[idx] = positions[i]
            else:
                self.get_logger().warn(f'Joint {joint_name} not found')

        # Start movement
        self.moving = True
        self.movement_start_time = self.get_clock().now().nanoseconds * 1e-9
        self.movement_duration = duration
        self.start_positions = self.current_positions.copy()

        self.get_logger().info(f'Starting movement to {len(joint_names)} positions over {duration}s')
        return True

    def execute_movement_pattern(self, movement_type, speed):
        """Execute a predefined movement pattern"""
        self.get_logger().info(f'Executing movement: {movement_type}')

        if movement_type == 'wave':
            # Wave right arm
            self.target_positions[self.joint_names.index('right_shoulder_joint')] = 1.0 / speed
            self.target_positions[self.joint_names.index('right_elbow_joint')] = 1.0 / speed
            self.movement_duration = 2.0 / speed
        elif movement_type == 'hello':
            # Wave with left arm
            self.target_positions[self.joint_names.index('left_shoulder_joint')] = 1.0 / speed
            self.target_positions[self.joint_names.index('left_elbow_joint')] = 1.0 / speed
            self.movement_duration = 2.0 / speed
        elif movement_type == 'stand':
            # Return to standing position
            for i in range(len(self.target_positions)):
                self.target_positions[i] = 0.0
            self.movement_duration = 1.5 / speed
        else:
            self.get_logger().warn(f'Unknown movement type: {movement_type}')
            return False

        # Start movement
        self.moving = True
        self.movement_start_time = self.get_clock().now().nanoseconds * 1e-9
        self.start_positions = self.current_positions.copy()

        return True

    def add_command_to_queue(self, command_type, **kwargs):
        """Add a command to the execution queue"""
        command = {'type': command_type}
        command.update(kwargs)
        self.command_queue.append(command)

    def publish_joint_states(self):
        """Publish the current joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names
        msg.position = self.current_positions
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)

        self.joint_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    server = ControlServiceServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Shutting down control service server')
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Create the Control Client (Week 3)

Implement a client for the control service:

```python
# control_service_client.py
import rclpy
from rclpy.node import Node
import time

class ControlServiceClient(Node):
    def __init__(self):
        super().__init__('control_service_client')

        # For this example, we'll communicate directly with our server node
        # In a real implementation, you would create actual service clients
        self.get_logger().info('Control Service Client initialized')

    def send_movement_command(self, server_node, movement_type, speed=1.0):
        """Send a movement command to the server"""
        self.get_logger().info(f'Sending movement command: {movement_type} at speed {speed}')

        command = {
            'type': 'movement',
            'movement_type': movement_type,
            'speed': speed
        }

        server_node.add_command_to_queue(command['type'],
                                       movement_type=command['movement_type'],
                                       speed=command['speed'])

    def send_position_command(self, server_node, joint_names, positions, duration=2.0):
        """Send a position command to the server"""
        self.get_logger().info(f'Sending position command for {len(joint_names)} joints')

        command = {
            'type': 'set_positions',
            'joint_names': joint_names,
            'positions': positions,
            'duration': duration
        }

        server_node.add_command_to_queue(command['type'],
                                       joint_names=command['joint_names'],
                                       positions=command['positions'],
                                       duration=command['duration'])

def main(args=None):
    rclpy.init(args=args)

    # Create server and client
    server = ControlServiceServer()
    client = ControlServiceClient()

    # Run both in separate threads or use a multi-node executor
    # For simplicity, we'll demonstrate the commands here

    def demo_sequence():
        client.get_logger().info('Starting demo sequence...')

        time.sleep(2)  # Wait for systems to initialize

        # Sequence of movements
        client.send_movement_command(server, 'stand', speed=1.0)
        time.sleep(3)

        client.send_movement_command(server, 'wave', speed=1.0)
        time.sleep(3)

        client.send_movement_command(server, 'hello', speed=0.8)
        time.sleep(3)

        # Move specific joints
        client.send_position_command(
            server,
            ['left_shoulder_joint', 'left_elbow_joint'],
            [0.5, 0.5],
            duration=2.0
        )
        time.sleep(3)

        client.send_movement_command(server, 'stand', speed=1.0)
        time.sleep(2)

        client.get_logger().info('Demo sequence completed')

    # Start the demo sequence in a separate thread
    import threading
    demo_thread = threading.Thread(target=demo_sequence)
    demo_thread.start()

    try:
        # Run the server
        rclpy.spin(server)
    except KeyboardInterrupt:
        client.get_logger().info('Shutting down control service client')
    finally:
        server.destroy_node()
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 6: Create Validation and Testing Script (Week 3)

```python
# validation_script.py
import xml.etree.ElementTree as ET
from urdf_parser_py.urdf import URDF
import subprocess
import sys

def validate_urdf_model(urdf_path):
    """Validate the URDF model for humanoid robot requirements"""
    print(f"Validating URDF model: {urdf_path}")

    try:
        robot = URDF.from_xml_file(urdf_path)

        print(f"✓ Robot name: {robot.name}")
        print(f"✓ Number of links: {len(robot.links)}")
        print(f"✓ Number of joints: {len(robot.joints)}")

        # Check for basic humanoid requirements
        humanoid_joints = {
            'left_leg': ['hip', 'knee', 'ankle'],
            'right_leg': ['hip', 'knee', 'ankle'],
            'left_arm': ['shoulder', 'elbow'],
            'right_arm': ['shoulder', 'elbow'],
            'torso': ['joint']  # At least one torso joint
        }

        # Count joints by type
        joint_types = {}
        for joint in robot.joints:
            if joint.type not in joint_types:
                joint_types[joint.type] = 0
            joint_types[joint.type] += 1

        print(f"✓ Joint types: {joint_types}")

        # Validate mass properties
        total_mass = sum(link.inertial.mass for link in robot.links if link.inertial)
        print(f"✓ Total robot mass: {total_mass:.2f} kg")

        if total_mass < 1.0:
            print("⚠ Warning: Robot mass seems too low (< 1kg)")

        # Check for visual and collision geometries
        links_without_visual = [link.name for link in robot.links if not link.visual]
        links_without_collision = [link.name for link in robot.links if not link.collision]

        if links_without_visual:
            print(f"⚠ Links without visual geometry: {links_without_visual}")

        if links_without_collision:
            print(f"⚠ Links without collision geometry: {links_without_collision}")

        # Check joint limits
        limited_joints = [j for j in robot.joints if j.limit]
        unlimited_joints = [j for j in robot.joints if not j.limit and j.type != 'fixed']

        print(f"✓ Joints with limits: {len(limited_joints)}")
        print(f"✓ Joints without limits: {len(unlimited_joints)}")

        if unlimited_joints and len(unlimited_joints) > len(robot.joints) * 0.1:  # More than 10% without limits
            print(f"⚠ Many joints without limits: {len(unlimited_joints)}")

        print("\n✅ URDF validation completed")
        return True

    except Exception as e:
        print(f"❌ URDF validation failed: {e}")
        return False

def run_system_integration_test():
    """Run integration tests for the entire system"""
    print("\nRunning system integration tests...")

    # Test 1: Check if ROS 2 is available
    try:
        result = subprocess.run(['ros2', 'topic', 'list'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ ROS 2 system available")
        else:
            print("❌ ROS 2 system not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ ROS 2 not installed or not in PATH")
        return False

    # Test 2: Check for required dependencies
    required_packages = [
        'rclpy',
        'sensor_msgs',
        'std_msgs'
    ]

    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print(f"❌ Missing Python packages: {missing_packages}")
        return False
    else:
        print("✓ All required Python packages available")

    print("✅ System integration tests passed")
    return True

def generate_report(robot_name, urdf_path):
    """Generate a validation report"""
    print(f"\n{'='*60}")
    print(f"ROBOT VALIDATION REPORT: {robot_name}")
    print(f"{'='*60}")

    # Validate URDF
    urdf_valid = validate_urdf_model(urdf_path)

    # Run system tests
    system_valid = run_system_integration_test()

    print(f"\nFINAL RESULTS:")
    print(f"URDF Model: {'✅ PASS' if urdf_valid else '❌ FAIL'}")
    print(f"System Integration: {'✅ PASS' if system_valid else '❌ FAIL'}")

    overall_pass = urdf_valid and system_valid
    print(f"Overall Status: {'✅ PASS' if overall_pass else '❌ FAIL'}")

    return overall_pass

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validation_script.py <urdf_file_path>")
        sys.exit(1)

    urdf_file = sys.argv[1]
    robot_name = "Student_Humanoid_Robot"

    success = generate_report(robot_name, urdf_file)
    sys.exit(0 if success else 1)
```

## Project Deliverables

### 1. Documentation (20%)
- Complete URDF model with proper documentation
- Code comments and inline documentation
- README file explaining the system architecture
- Validation results and testing logs

### 2. Implementation (50%)
- Working URDF model with realistic kinematics
- Functional publisher-subscriber system
- Working service-client for robot control
- Proper error handling and validation

### 3. Testing and Validation (20%)
- Validation script that checks URDF requirements
- Integration testing between components
- Demonstration of movement patterns
- Performance and safety checks

### 4. Presentation (10%)
- Demonstration of the working system
- Explanation of design choices
- Discussion of challenges faced and solutions

## Evaluation Criteria

### Technical Requirements (70%)
- URDF model completeness and accuracy (20%)
- Proper ROS 2 communication patterns (20%)
- Service implementation and functionality (15%)
- Code quality and documentation (15%)

### System Integration (20%)
- Successful integration of all components
- Proper error handling
- System validation results

### Creativity and Innovation (10%)
- Unique features or improvements
- Creative movement patterns
- Innovative control approaches

## Submission Requirements

1. **Source Code**: All Python scripts, URDF files, and configuration files
2. **Documentation**: README with setup instructions and technical documentation
3. **Validation Results**: Output from your validation script
4. **Video Demonstration**: A 5-minute video showing your system in action
5. **Technical Report**: 2-3 page report explaining your design decisions

## Grading Rubric

- **Excellent (90-100%)**: All requirements met with innovative features, excellent documentation, and robust implementation
- **Good (80-89%)**: All requirements met with good implementation and documentation
- **Satisfactory (70-79%)**: Basic requirements met but with some issues or missing features
- **Needs Improvement (60-69%)**: Partial completion with significant issues
- **Unsatisfactory (60%)**: Incomplete or non-functional system
/m
## Helpful Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [Robotics Stack Exchange](https://robotics.stackexchange.com/)
- [OpenRDF Documentation](https://www.openrobots.org/wiki/orocos/)

## Tips for Success

1. **Start Early**: This is a comprehensive project that requires time for testing and debugging
2. **Test Incrementally**: Implement and test each component before integrating
3. **Validate Often**: Use the validation script frequently during development
4. **Document Progress**: Keep notes on design decisions and challenges overcome
5. **Seek Help**: Don't hesitate to ask for help when stuck on technical issues

## Next Steps

Upon successful completion of this project, you will be ready to advance to Module 2: The Digital Twin, where you'll learn to create simulation environments for your humanoid robot in Gazebo and Unity.