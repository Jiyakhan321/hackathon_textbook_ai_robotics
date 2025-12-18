---
sidebar_position: 1
---

# Exercise 1: Simple Publisher Node for Humanoid Robot

## Overview

In this exercise, you'll create a simple ROS 2 publisher node that publishes joint state information for a humanoid robot. This will help you understand the basics of creating ROS 2 nodes and publishing messages to topics.

## Learning Objectives

By completing this exercise, you will:
- Create a basic ROS 2 publisher node
- Understand how to publish sensor_msgs/JointState messages
- Learn to work with time stamps and message headers
- Practice using timers for periodic publishing

## Prerequisites

Before starting this exercise, ensure you have:
- ROS 2 Humble Hawksbill installed
- Basic Python programming knowledge
- Understanding of ROS 2 concepts (nodes, topics, messages)

## Step 1: Setting Up the Package

First, create a new ROS 2 package for your exercises:

```bash
# Create a new workspace if you don't have one
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws/src

# Create the package
ros2 pkg create --build-type ament_python humanoid_publisher_examples
cd humanoid_publisher_examples
```

Your package structure should look like:
```
humanoid_publisher_examples/
├── package.xml
├── setup.py
├── setup.cfg
└── humanoid_publisher_examples/
    └── __init__.py
```

## Step 2: Creating the Publisher Node

Create a new Python file in the package directory:

```python
# humanoid_publisher_examples/simple_joint_publisher.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class SimpleJointPublisher(Node):
    def __init__(self):
        super().__init__('simple_joint_publisher')

        # Create publisher for joint states
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Create a timer to publish at regular intervals (50 Hz)
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize joint names for a simple humanoid (torso, 2 arms, 2 legs)
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        # Initialize joint positions with zeros
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

        # Initialize a time counter for oscillating movements
        self.time_counter = 0.0

        self.get_logger().info('Simple Joint Publisher node initialized')

    def timer_callback(self):
        """Callback function that publishes joint state messages"""
        # Create a new JointState message
        msg = JointState()

        # Set the header with timestamp and frame ID
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Set the joint names
        msg.name = self.joint_names

        # Update joint positions with simple oscillating patterns
        self.update_joint_positions()

        # Set the positions, velocities, and efforts
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        # Publish the message
        self.publisher.publish(msg)

        # Log a message every 100 publications
        if int(self.time_counter * 50) % 100 == 0:
            self.get_logger().info(f'Published joint states: {len(msg.name)} joints')

    def update_joint_positions(self):
        """Update joint positions with simple oscillating patterns"""
        # Increment time counter (using timer period)
        self.time_counter += 0.02

        # Create different oscillating patterns for different joints
        for i, joint_name in enumerate(self.joint_names):
            if 'hip' in joint_name:
                # Hip joints: slow oscillation
                self.joint_positions[i] = 0.3 * math.sin(0.5 * self.time_counter + i)
            elif 'knee' in joint_name:
                # Knee joints: follow hip with phase offset
                self.joint_positions[i] = 0.2 * math.sin(0.5 * self.time_counter + i + 1)
            elif 'ankle' in joint_name:
                # Ankle joints: faster oscillation
                self.joint_positions[i] = 0.15 * math.sin(0.8 * self.time_counter + i + 2)
            elif 'shoulder' in joint_name:
                # Shoulder joints: independent oscillation
                self.joint_positions[i] = 0.4 * math.sin(0.4 * self.time_counter + i * 0.5)
            elif 'elbow' in joint_name:
                # Elbow joints: follow shoulder
                self.joint_positions[i] = 0.25 * math.sin(0.4 * self.time_counter + i * 0.5 + 1)

def main(args=None):
    """Main function to initialize and run the node"""
    rclpy.init(args=args)

    # Create the publisher node
    simple_publisher = SimpleJointPublisher()

    try:
        # Start spinning the node
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        simple_publisher.get_logger().info('Interrupted by user')
    finally:
        # Clean up
        simple_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 3: Creating the Executable Script

Create an executable script to run your node:

```python
# humanoid_publisher_examples/simple_publisher_main.py

from simple_joint_publisher import main

if __name__ == '__main__':
    main()
```

## Step 4: Updating setup.py

Update your `setup.py` file to include the entry point for your executable:

```python
from setuptools import setup

package_name = 'humanoid_publisher_examples'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Simple publisher examples for humanoid robots',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_joint_publisher = humanoid_publisher_examples.simple_joint_publisher:main',
        ],
    },
)
```

## Step 5: Building and Running the Package

Build your package:

```bash
cd ~/humanoid_ws
colcon build --packages-select humanoid_publisher_examples
source install/setup.bash
```

## Step 6: Running the Publisher

Run your publisher node:

```bash
ros2 run humanoid_publisher_examples simple_joint_publisher
```

## Step 7: Verifying the Publisher

In a new terminal, verify that your publisher is working:

```bash
# Check if the topic exists
ros2 topic list | grep joint_states

# Echo the published messages
ros2 topic echo /joint_states

# Check the message type
ros2 topic type /joint_states
```

## Step 8: Visualizing with RViz

To visualize the joint states in RViz:

1. Make sure you have a URDF file for your humanoid robot
2. Run the robot state publisher:
```bash
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="path/to/your/robot.urdf"
```
3. Run your joint publisher in another terminal
4. Launch RViz:
```bash
ros2 run rviz2 rviz2
```
5. Add a RobotModel display and set the topic to your robot description
6. Add a TF display to visualize the transforms

## Step 9: Enhanced Publisher with Error Handling

Now let's create an enhanced version with better error handling and additional features:

```python
# humanoid_publisher_examples/enhanced_joint_publisher.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import numpy as np

class EnhancedJointPublisher(Node):
    def __init__(self):
        super().__init__('enhanced_joint_publisher')

        # Create publisher for joint states
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Create a timer to publish at regular intervals
        timer_period = 0.02  # 50 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Define joint configuration
        self.joint_config = {
            'left_leg': ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint'],
            'right_leg': ['right_hip_joint', 'right_knee_joint', 'right_ankle_joint'],
            'left_arm': ['left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint'],
            'right_arm': ['right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint']
        }

        # Flatten joint names list
        self.joint_names = []
        for joints in self.joint_config.values():
            self.joint_names.extend(joints)

        # Initialize joint states
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

        # Movement pattern parameters
        self.time_counter = 0.0
        self.pattern_type = 'walking'  # 'walking', 'waving', 'idle'

        self.get_logger().info(f'Enhanced Joint Publisher initialized with {len(self.joint_names)} joints')

    def timer_callback(self):
        """Enhanced callback with multiple movement patterns"""
        try:
            # Create and populate the JointState message
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'

            # Update joint positions based on selected pattern
            self.update_joint_positions()

            # Update message data
            msg.name = self.joint_names
            msg.position = self.joint_positions.copy()
            msg.velocity = self.joint_velocities
            msg.effort = self.joint_efforts

            # Publish the message
            self.publisher.publish(msg)

            # Log periodically
            if int(self.time_counter * 50) % 250 == 0:
                self.get_logger().info(f'Published joint states - Pattern: {self.pattern_type}')

        except Exception as e:
            self.get_logger().error(f'Error in timer_callback: {e}')

    def update_joint_positions(self):
        """Update joint positions based on movement pattern"""
        self.time_counter += 0.02

        # Select movement pattern
        if self.pattern_type == 'walking':
            self._update_walking_pattern()
        elif self.pattern_type == 'waving':
            self._update_waving_pattern()
        else:  # idle
            self._update_idle_pattern()

    def _update_walking_pattern(self):
        """Simulate a simple walking pattern"""
        # Left leg (alternating with right)
        left_phase = math.sin(2 * math.pi * 0.5 * self.time_counter)
        right_phase = math.sin(2 * math.pi * 0.5 * self.time_counter + math.pi)  # Opposite phase

        # Hip joints
        self._set_joints_by_name(['left_hip_joint'], 0.3 * left_phase)
        self._set_joints_by_name(['right_hip_joint'], 0.3 * right_phase)

        # Knee joints (follow hip with different amplitude)
        self._set_joints_by_name(['left_knee_joint'], 0.5 * left_phase)
        self._set_joints_by_name(['right_knee_joint'], 0.5 * right_phase)

        # Ankle joints (smaller movement)
        self._set_joints_by_name(['left_ankle_joint'], 0.1 * left_phase)
        self._set_joints_by_name(['right_ankle_joint'], 0.1 * right_phase)

        # Arms swinging opposite to legs
        self._set_joints_by_name(['left_shoulder_joint'], 0.2 * right_phase)
        self._set_joints_by_name(['right_shoulder_joint'], 0.2 * left_phase)
        self._set_joints_by_name(['left_elbow_joint'], 0.15 * right_phase)
        self._set_joints_by_name(['right_elbow_joint'], 0.15 * left_phase)

    def _update_waving_pattern(self):
        """Simulate waving motion with right arm"""
        # Keep legs in neutral position
        leg_joints = ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
                     'right_hip_joint', 'right_knee_joint', 'right_ankle_joint']
        for joint in leg_joints:
            self._set_joints_by_name([joint], 0.0)

        # Waving motion with right arm
        wave_phase = math.sin(2 * math.pi * 0.8 * self.time_counter)
        self._set_joints_by_name(['right_shoulder_joint'], 0.5 + 0.3 * wave_phase)
        self._set_joints_by_name(['right_elbow_joint'], 0.4 * wave_phase)

        # Left arm in neutral position
        self._set_joints_by_name(['left_shoulder_joint'], 0.2)
        self._set_joints_by_name(['left_elbow_joint'], -0.5)

    def _update_idle_pattern(self):
        """Keep all joints in neutral position with small oscillation"""
        for i in range(len(self.joint_positions)):
            # Small random oscillation to keep it interesting
            self.joint_positions[i] = 0.05 * math.sin(0.3 * self.time_counter + i * 0.1)

    def _set_joints_by_name(self, joint_names, value):
        """Helper method to set specific joint positions"""
        for joint_name in joint_names:
            if joint_name in self.joint_names:
                index = self.joint_names.index(joint_name)
                self.joint_positions[index] = value

def main(args=None):
    """Main function to initialize and run the enhanced node"""
    rclpy.init(args=args)

    try:
        enhanced_publisher = EnhancedJointPublisher()
        enhanced_publisher.get_logger().info('Starting enhanced joint publisher...')

        rclpy.spin(enhanced_publisher)

    except KeyboardInterrupt:
        print('\nShutting down enhanced joint publisher...')
    finally:
        enhanced_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 10: Running the Enhanced Publisher

Update your `setup.py` to include the new executable:

```python
# Add to entry_points in setup.py
entry_points={
    'console_scripts': [
        'simple_joint_publisher = humanoid_publisher_examples.simple_joint_publisher:main',
        'enhanced_joint_publisher = humanoid_publisher_examples.enhanced_joint_publisher:main',
    ],
},
```

Rebuild and run the enhanced publisher:

```bash
cd ~/humanoid_ws
colcon build --packages-select humanoid_publisher_examples
source install/setup.bash

# Run the enhanced publisher
ros2 run humanoid_publisher_examples enhanced_joint_publisher
```

## Step 11: Testing and Verification

Create a simple test script to verify your publisher:

```python
# humanoid_publisher_examples/test_publisher.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')

        # Create subscription to joint states
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.received_first_message = False
        self.get_logger().info('Joint state subscriber initialized')

    def joint_state_callback(self, msg):
        """Callback to verify joint state messages"""
        if not self.received_first_message:
            self.get_logger().info(f'Received first joint state message with {len(msg.name)} joints')
            self.get_logger().info(f'Joint names: {msg.name}')
            self.received_first_message = True

        # Log periodically
        if len(msg.name) > 0:
            avg_position = sum(msg.position) / len(msg.position) if msg.position else 0
            self.get_logger().debug(f'Average joint position: {avg_position:.3f}')

def main(args=None):
    rclpy.init(args=args)

    subscriber = JointStateSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        subscriber.get_logger().info('Subscriber interrupted')
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Tips

1. **Topic not found**: Make sure your publisher node is running
2. **No messages received**: Check that the topic names match exactly
3. **Permission errors**: Ensure your Python files have execute permissions
4. **Import errors**: Make sure your package is properly built and sourced

## Next Steps

After completing this exercise, you should have a solid understanding of:
- Creating ROS 2 publisher nodes
- Working with JointState messages
- Using timers for periodic publishing
- Implementing different movement patterns

In the next exercise, we'll explore creating a subscriber node that can receive and process these joint state messages to control a simulated humanoid robot.