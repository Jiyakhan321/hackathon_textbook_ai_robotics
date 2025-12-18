---
sidebar_position: 1
---

# URDF Syntax and Structure

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. It defines the physical and visual properties of a robot, including links, joints, inertial properties, and visual/collision geometries. For humanoid robots, URDF is essential for simulation, visualization, and control.

## Basic URDF Structure

A basic URDF file has the following structure:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0.1 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Link Elements

Links represent rigid bodies in the robot. Each link contains:

### Inertial Properties
```xml
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
</inertial>
```

- `mass`: Mass of the link in kilograms
- `origin`: Center of mass location and orientation offset
- `inertia`: 3x3 inertia matrix values (ixx, ixy, ixz, iyy, iyz, izz)

### Visual Properties
```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.2 0.2 0.2"/>
  </geometry>
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>
</visual>
```

- `origin`: Visual mesh origin relative to link frame
- `geometry`: Shape of the visual representation
- `material`: Color and appearance properties

### Collision Properties
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.2 0.2 0.2"/>
  </geometry>
</collision>
```

- `origin`: Collision mesh origin relative to link frame
- `geometry`: Shape of the collision representation

## Joint Elements

Joints define the connection between links and their motion characteristics:

### Joint Types
- `revolute`: Rotational joint with limited range
- `continuous`: Rotational joint without limits
- `prismatic`: Linear sliding joint with limits
- `fixed`: No movement between links
- `floating`: 6 DOF movement (rarely used)
- `planar`: Movement on a plane (rarely used)

### Joint Definition Example
```xml
<joint name="hip_joint" type="revolute">
  <parent link="torso_link"/>
  <child link="thigh_link"/>
  <origin xyz="0 0 -0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

- `name`: Unique joint identifier
- `type`: Joint type (revolute, continuous, etc.)
- `parent`: Parent link name
- `child`: Child link name
- `origin`: Joint location and orientation relative to parent
- `axis`: Axis of rotation or translation
- `limit`: Joint limits (for revolute and prismatic joints)
- `dynamics`: Joint dynamics parameters

## Geometry Types

URDF supports several geometry types:

### Primitive Shapes
```xml
<!-- Box -->
<geometry>
  <box size="0.1 0.2 0.3"/>
</geometry>

<!-- Cylinder -->
<geometry>
  <cylinder radius="0.05" length="0.1"/>
</geometry>

<!-- Sphere -->
<geometry>
  <sphere radius="0.05"/>
</geometry>
```

### Mesh Files
```xml
<geometry>
  <mesh filename="package://humanoid_description/meshes/hip.dae" scale="1 1 1"/>
</geometry>
```

## Material Definitions

Materials define the visual appearance of links:

```xml
<material name="red">
  <color rgba="1 0 0 1"/>
</material>

<material name="green">
  <color rgba="0 1 0 1"/>
</material>

<material name="blue">
  <color rgba="0 0 1 1"/>
</material>

<material name="black">
  <color rgba="0 0 0 1"/>
</material>

<material name="white">
  <color rgba="1 1 1 1"/>
</material>
```

## Complete Humanoid Robot URDF Example

Here's a simplified humanoid robot model:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
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
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link (pelvis) -->
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.15"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso_link"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>

  <link name="torso_link">
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.25 0.4"/>
      </geometry>
      <material name="grey"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.25 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1.0"/>
  </joint>

  <link name="head_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh_link"/>
    <origin xyz="0 0.1 -0.075" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_thigh_link">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh_link"/>
    <child link="left_shin_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.35" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_shin_link">
    <inertial>
      <mass value="2.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin_link"/>
    <child link="left_foot_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.78" upper="0.78" effort="50" velocity="1.0"/>
  </joint>

  <link name="left_foot_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Leg (similar to left, mirrored) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh_link"/>
    <origin xyz="0 -0.1 -0.075" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_thigh_link">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh_link"/>
    <child link="right_shin_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.35" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_shin_link">
    <inertial>
      <mass value="2.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_shin_link"/>
    <child link="right_foot_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.78" upper="0.78" effort="50" velocity="1.0"/>
  </joint>

  <link name="right_foot_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso_link"/>
    <child link="left_upper_arm_link"/>
    <origin xyz="0.15 0.1 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm_link">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm_link"/>
    <child link="left_lower_arm_link"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1.0"/>
  </joint>

  <link name="left_lower_arm_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.15"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wrist_joint" type="revolute">
    <parent link="left_lower_arm_link"/>
    <child link="left_hand_link"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.78" upper="0.78" effort="20" velocity="1.0"/>
  </joint>

  <link name="left_hand_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0.025 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0.025 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0.025 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Arm (similar to left, mirrored) -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso_link"/>
    <child link="right_upper_arm_link"/>
    <origin xyz="0.15 -0.1 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <link name="right_upper_arm_link">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm_link"/>
    <child link="right_lower_arm_link"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1.0"/>
  </joint>

  <link name="right_lower_arm_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.15"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_wrist_joint" type="revolute">
    <parent link="right_lower_arm_link"/>
    <child link="right_hand_link"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.78" upper="0.78" effort="20" velocity="1.0"/>
  </joint>

  <link name="right_hand_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0.025 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0.025 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0.025 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
  </link>
</robot>
```

## Xacro for Complex URDFs

For complex humanoid robots, Xacro (XML Macros) is often used to simplify URDF creation:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="mass_leg" value="5.0" />
  <xacro:property name="mass_arm" value="2.0" />
  <xacro:property name="mass_torso" value="10.0" />

  <!-- Macro for leg segments -->
  <xacro:macro name="leg_segment" params="name parent xyz mass radius length">
    <joint name="${name}_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${name}_link"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    </joint>

    <link name="${name}_link">
      <inertial>
        <mass value="${mass}"/>
        <origin xyz="0 0 -${length/2}" rpy="0 0 0"/>
        <inertia ixx="${mass*length*length/12}" ixy="0" ixz="0"
                 iyy="${mass*length*length/12}" iyz="0"
                 izz="${mass*radius*radius/2}"/>
      </inertial>

      <visual>
        <origin xyz="0 0 -${length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${length}"/>
        </geometry>
        <material name="blue"/>
      </visual>

      <collision>
        <origin xyz="0 0 -${length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${length}"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="${mass_torso}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Use the macro to create leg segments -->
  <xacro:leg_segment name="left_thigh" parent="base_link"
                     xyz="0 0.1 -0.1" mass="${mass_leg}" radius="0.05" length="0.3"/>

</robot>
```

## URDF Validation and Tools

### Checking URDF validity:
```bash
# Install check tools
sudo apt-get install ros-humble-urdfdom-py

# Validate URDF
check_urdf /path/to/robot.urdf
```

### Visualizing URDF:
```bash
# Launch RViz with robot model
ros2 launch urdf_tutorial display.launch.py model:=/path/to/robot.urdf
```

## Best Practices

1. **Use consistent naming**: Follow a clear convention for link and joint names
2. **Proper mass properties**: Accurate inertial properties are crucial for simulation
3. **Collision vs visual**: Use simpler geometries for collision to improve performance
4. **Joint limits**: Always specify appropriate joint limits to prevent damage
5. **Xacro for complexity**: Use Xacro macros for repetitive structures like limbs
6. **Documentation**: Comment your URDF files to explain the structure

## Next Steps

Now that you understand URDF syntax, let's explore how to create specific humanoid robot structures with proper kinematic chains and biomechanical considerations.