---
sidebar_position: 3
---

# Visualizing and Validating URDF Models

## Overview

Creating a URDF model is only the first step. To ensure your humanoid robot model works correctly, you need to visualize it, validate its structure, and test its functionality. This section covers the tools and techniques for visualizing and validating URDF models.

## URDF Validation Tools

### 1. Command Line Validation

The first step in validation is checking the URDF syntax:

```bash
# Install URDF validation tools
sudo apt-get install ros-humble-urdfdom-py

# Validate URDF syntax
check_urdf /path/to/robot.urdf

# This will output:
# * Successfully Parsed file
# * Robot name: humanoid_robot
# * 27 links
# * 26 joints
# * 26 fixed joints
# * 0 floating joints
# * 0 prismatic joints
# * 26 revolute joints
```

### 2. Python Validation Script

You can also validate URDF programmatically:

```python
#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from urdf_parser_py.urdf import URDF
import sys

def validate_urdf(urdf_path):
    """Validate URDF file programmatically"""
    try:
        # Parse URDF file
        robot = URDF.from_xml_file(urdf_path)

        print(f"Robot name: {robot.name}")
        print(f"Number of links: {len(robot.links)}")
        print(f"Number of joints: {len(robot.joints)}")

        # Check for common issues
        issues = []

        # Check for links without visual/collision geometry
        for link in robot.links:
            if not link.visual and not link.collision:
                issues.append(f"Link {link.name} has no visual or collision geometry")

        # Check for links without inertial properties
        for link in robot.links:
            if not link.inertial:
                issues.append(f"Link {link.name} has no inertial properties")

        # Check joint limits
        for joint in robot.joints:
            if joint.limit:
                if joint.limit.lower >= joint.limit.upper:
                    issues.append(f"Joint {joint.name} has invalid limits: {joint.limit.lower} >= {joint.limit.upper}")

        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("URDF validation passed!")

        return len(issues) == 0

    except Exception as e:
        print(f"Error validating URDF: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_urdf.py <urdf_file>")
        sys.exit(1)

    urdf_file = sys.argv[1]
    validate_urdf(urdf_file)
```

## Visualization Tools

### 1. RViz Visualization

RViz is the primary visualization tool for ROS. Here's how to set it up for URDF visualization:

```bash
# Launch RViz with robot model
ros2 run rviz2 rviz2

# In RViz, add displays:
# - RobotModel: Set "Description Source" to "Topic" and "Topic" to "/robot_description"
# - TF: To see the transform tree
# - JointState: To visualize joint values
```

Create a launch file to automate this:

```python
# robot_visualization.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "model",
            default_value="humanoid.urdf",
            description="URDF file to visualize",
        )
    )

    # Get URDF path
    model_path = LaunchConfiguration("model")

    # Robot State Publisher node
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": model_path}],
    )

    # Joint State Publisher GUI (for manual joint control)
    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
    )

    # RViz node
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("my_robot_description"), "rviz", "view_robot.rviz"])],
    )

    return LaunchDescription(
        declared_arguments
        + [
            robot_state_publisher,
            joint_state_publisher_gui,
            rviz,
        ]
    )
```

### 2. Command Line Visualization

For quick visualization without creating launch files:

```bash
# Method 1: Using xacro if your URDF is in xacro format
ros2 run xacro xacro /path/to/robot.xacro > /tmp/robot.urdf
ros2 run rviz2 rviz2

# Method 2: Direct URDF visualization
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="/path/to/robot.urdf"
```

## Interactive Joint Control

Use the Joint State Publisher GUI to interactively control joint positions:

```xml
<!-- In your URDF, make sure to include default joint values -->
<joint name="left_elbow_joint" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_lower_arm"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0.0" upper="2.5" effort="80" velocity="1.2"/>
  <!-- Joint State Publisher will use the limit values for slider ranges -->
</joint>
```

## Creating a Visualization Package

Let's create a complete visualization setup:

### Package Structure
```
robot_visualization/
├── CMakeLists.txt
├── package.xml
├── launch/
│   └── display.launch.py
├── rviz/
│   └── view_robot.rviz
└── urdf/
    └── humanoid.urdf
```

### Launch File
```python
# launch/display.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    urdf_model_path = LaunchConfiguration('model')
    gui_config = LaunchConfiguration('gui')

    # Declare launch arguments
    model_arg = DeclareLaunchArgument(
        name='model',
        default_value='urdf/humanoid.urdf',
        description='URDF file path relative to this package'
    )

    gui_arg = DeclareLaunchArgument(
        name='gui',
        default_value='true',
        description='Flag to enable joint state publisher gui'
    )

    # Robot State Publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare('robot_visualization'),
                LaunchConfiguration('model')
            ])
        }]
    )

    # Joint State Publisher node
    joint_state_publisher_node = Node(
        condition=IfCondition(gui_config),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui'
    )

    # Joint State Publisher (non-GUI version as backup)
    joint_state_publisher_node_backup = Node(
        condition=IfCondition('not ' + gui_config),
        package='joint_state_publisher',
        executable='joint_state_publisher'
    )

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('robot_visualization'),
            'rviz',
            'view_robot.rviz'
        ])]
    )

    return LaunchDescription([
        model_arg,
        gui_arg,
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_node_backup,
        rviz_node
    ])
```

### RViz Configuration
```yaml
# rviz/view_robot.rviz
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /TF1
      Splitter Ratio: 0.5
    Tree Height: 617
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 0.5
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
    - Class: rviz_default_plugins/SetGoal
    - Class: rviz_default_plugins/PublishPoint
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2.5
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.5
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002f4fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000002f4000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa000025a900000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d006501000000000000045000000000000000000000023f000002f400000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Selection:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: false
  Width: 1200
  X: 72
  Y: 60
```

## Testing URDF in Gazebo Simulation

To test your URDF in Gazebo simulation:

```xml
<!-- Add Gazebo-specific tags to your URDF -->
<robot name="humanoid_robot">
  <!-- ... your existing URDF ... -->

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Material definitions for Gazebo -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>
</robot>
```

## URDF Debugging Techniques

### 1. Checking Transform Tree

Use the TF tree to verify your kinematic structure:

```bash
# View the transform tree
ros2 run tf2_tools view_frames

# View specific transforms
ros2 run tf2_ros tf2_echo base_link left_foot_link

# Visualize TF in RViz
# Add TF display and check that all links are connected properly
```

### 2. Joint State Monitoring

Monitor joint states to ensure they're being published correctly:

```bash
# Echo joint states topic
ros2 topic echo /joint_states

# Use rqt to visualize joint states
ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller
```

### 3. URDF Parser Debugging

Create a debug script to analyze your URDF:

```python
#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from urdf_parser_py.urdf import URDF
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_urdf(urdf_path):
    """Analyze URDF structure and properties"""
    robot = URDF.from_xml_file(urdf_path)

    print(f"Robot: {robot.name}")
    print(f"Links: {len(robot.links)}")
    print(f"Joints: {len(robot.joints)}")

    # Create kinematic tree
    G = nx.DiGraph()

    # Add nodes (links)
    for link in robot.links:
        G.add_node(link.name)

    # Add edges (joints)
    for joint in robot.joints:
        G.add_edge(joint.parent, joint.child)

    # Analyze tree structure
    print(f"Graph nodes: {list(G.nodes())}")
    print(f"Graph edges: {list(G.edges())}")

    # Find root link (node with no incoming edges)
    root_links = [node for node in G.nodes() if G.in_degree(node) == 0]
    print(f"Root links: {root_links}")

    # Find leaf links (nodes with no outgoing edges)
    leaf_links = [node for node in G.nodes() if G.out_degree(node) == 0]
    print(f"Leaf links: {leaf_links}")

    # Calculate path lengths from root to leaves
    if root_links:
        root = root_links[0]
        for leaf in leaf_links:
            try:
                path = nx.shortest_path(G, root, leaf)
                print(f"Path from {root} to {leaf}: {' -> '.join(path)} (length: {len(path)-1})")
            except nx.NetworkXNoPath:
                print(f"No path from {root} to {leaf}")

    # Check joint properties
    joint_types = defaultdict(int)
    for joint in robot.joints:
        joint_types[joint.type] += 1

    print(f"Joint types: {dict(joint_types)}")

    # Check for missing properties
    missing_visual = []
    missing_collision = []
    missing_inertial = []

    for link in robot.links:
        if not link.visual:
            missing_visual.append(link.name)
        if not link.collision:
            missing_collision.append(link.name)
        if not link.inertial:
            missing_inertial.append(link.name)

    if missing_visual:
        print(f"Links without visual: {missing_visual}")
    if missing_collision:
        print(f"Links without collision: {missing_collision}")
    if missing_inertial:
        print(f"Links without inertial: {missing_inertial}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_urdf.py <urdf_file>")
        sys.exit(1)

    analyze_urdf(sys.argv[1])
```

## Advanced Visualization: Creating Joint Trajectory Demonstrations

Create a script to demonstrate joint movements:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math
import time

class JointTrajectoryDemo(Node):
    def __init__(self):
        super().__init__('joint_trajectory_demo')

        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Timer for publishing joint states
        self.timer = self.create_timer(0.05, self.publish_joint_states)  # 20 Hz

        # Initialize joint names (should match your URDF)
        self.joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]

        # Initialize positions
        self.positions = [0.0] * len(self.joint_names)

        # Movement parameters
        self.time_offset = time.time()

        self.get_logger().info('Joint trajectory demo initialized')

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Create oscillating movements for demonstration
        current_time = time.time()

        for i, joint_name in enumerate(self.joint_names):
            # Different movement patterns for different joints
            if 'hip' in joint_name:
                # Hip joints move with larger amplitude
                self.positions[i] = 0.5 * math.sin(0.5 * (current_time - self.time_offset) + i)
            elif 'knee' in joint_name:
                # Knee joints follow hip movement
                self.positions[i] = 0.3 * math.sin(0.5 * (current_time - self.time_offset) + i + 1)
            elif 'shoulder' in joint_name:
                # Shoulder joints move independently
                self.positions[i] = 0.4 * math.sin(0.3 * (current_time - self.time_offset) + i * 0.5)
            elif 'elbow' in joint_name:
                # Elbow joints follow shoulder
                self.positions[i] = 0.3 * math.sin(0.3 * (current_time - self.time_offset) + i * 0.5 + 1)
            else:
                # Other joints move with smaller amplitude
                self.positions[i] = 0.2 * math.sin(0.7 * (current_time - self.time_offset) + i * 0.3)

        msg.position = self.positions
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)

        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryDemo()

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

## Common URDF Issues and Solutions

### 1. Floating Point Precision
```xml
<!-- Good: Use reasonable precision -->
<origin xyz="0.0 0.0 0.1" rpy="0.0 0.0 0.0"/>

<!-- Avoid: Excessive precision -->
<origin xyz="0.00000000000000001 0.0 0.1" rpy="0.0 0.0 0.0"/>
```

### 2. Mass and Inertia Issues
```xml
<!-- Good: Realistic mass properties -->
<inertial>
  <mass value="5.0"/>
  <origin xyz="0 0 -0.15" rpy="0 0 0"/>
  <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
</inertial>

<!-- Avoid: Zero or negative values -->
<inertial>
  <mass value="0"/>  <!-- This will cause issues in simulation -->
  <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
</inertial>
```

### 3. Joint Limit Issues
```xml
<!-- Good: Realistic joint limits -->
<joint name="left_elbow_joint" type="revolute">
  <limit lower="0.0" upper="2.5" effort="80" velocity="1.2"/>
</joint>

<!-- Avoid: Limits that exceed mechanical capabilities -->
<joint name="left_elbow_joint" type="revolute">
  <limit lower="-3.14" upper="3.14" effort="80" velocity="1.2"/>  <!-- Elbows don't rotate 360° -->
</joint>
```

## Best Practices for URDF Visualization

1. **Start Simple**: Begin with a basic model and add complexity gradually
2. **Use Hierarchical Structure**: Organize joints and links logically
3. **Validate Early**: Check your URDF frequently during development
4. **Test in Simulation**: Always test your model in simulation before hardware
5. **Document Assumptions**: Comment on coordinate frames and joint directions
6. **Use Standard Formats**: Follow ROS/URDF conventions for compatibility
7. **Performance Considerations**: Use simpler collision geometries for simulation

## Next Steps

Now that you can visualize and validate your URDF models, let's move on to creating practical exercises that will help you apply these concepts to real humanoid robot scenarios.