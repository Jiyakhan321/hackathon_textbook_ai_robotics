---
sidebar_position: 2
---

# Humanoid Robot Structure and Kinematics

## Overview

Designing a humanoid robot structure requires careful consideration of biomechanics, kinematics, and control requirements. In this section, we'll explore how to structure a humanoid robot model in URDF to achieve human-like movement and functionality.

## Humanoid Robot Anatomy

A typical humanoid robot consists of several key components that mirror human anatomy:

### 1. Torso and Head
- Pelvis (base link)
- Torso/trunk
- Neck and head

### 2. Upper Extremities
- Shoulders
- Upper arms
- Forearms
- Hands

### 3. Lower Extremities
- Hips
- Thighs
- Shins
- Feet

## Kinematic Chains in Humanoid Robots

Humanoid robots typically have multiple kinematic chains:

### Standalone Chains
- Left arm chain (from torso to left hand)
- Right arm chain (from torso to right hand)
- Left leg chain (from pelvis to left foot)
- Right leg chain (from pelvis to right foot)

### Constrained Chains
- Head chain (from torso to head)
- Spine (multi-joint connection between pelvis and torso)

## Designing Joint Ranges for Humanoid Robots

Humanoid joints should have appropriate ranges of motion based on human capabilities:

### Lower Body Joints
```xml
<!-- Hip joints - typically 3 DOF: flexion/extension, abduction/adduction, internal/external rotation -->
<joint name="left_hip_flexion" type="revolute">
  <limit lower="-2.0" upper="0.5" effort="200" velocity="2.0"/>
</joint>

<joint name="left_hip_abduction" type="revolute">
  <limit lower="-0.5" upper="0.5" effort="150" velocity="1.5"/>
</joint>

<joint name="left_hip_rotation" type="revolute">
  <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
</joint>

<!-- Knee joint - primarily flexion/extension -->
<joint name="left_knee_joint" type="revolute">
  <limit lower="0.0" upper="2.35" effort="200" velocity="2.0"/>  <!-- 0 to 135 degrees -->
</joint>

<!-- Ankle joints - 2 DOF: dorsiflexion/plantarflexion, inversion/eversion -->
<joint name="left_ankle_flexion" type="revolute">
  <limit lower="-0.5" upper="0.5" effort="80" velocity="1.0"/>
</joint>

<joint name="left_ankle_side" type="revolute">
  <limit lower="-0.3" upper="0.3" effort="60" velocity="0.8"/>
</joint>
```

### Upper Body Joints
```xml
<!-- Shoulder joints - 3 DOF: flexion/extension, abduction/adduction, internal/external rotation -->
<joint name="left_shoulder_pitch" type="revolute">
  <limit lower="-2.0" upper="2.0" effort="100" velocity="1.5"/>
</joint>

<joint name="left_shoulder_roll" type="revolute">
  <limit lower="0.0" upper="3.14" effort="80" velocity="1.2"/>
</joint>

<joint name="left_shoulder_yaw" type="revolute">
  <limit lower="-1.57" upper="1.57" effort="60" velocity="1.0"/>
</joint>

<!-- Elbow joint - primarily flexion/extension -->
<joint name="left_elbow_joint" type="revolute">
  <limit lower="0.0" upper="2.5" effort="80" velocity="1.5"/>  <!-- 0 to 143 degrees -->
</joint>

<!-- Wrist joints - 2-3 DOF depending on complexity -->
<joint name="left_wrist_pitch" type="revolute">
  <limit lower="-0.8" upper="0.8" effort="30" velocity="1.0"/>
</joint>

<joint name="left_wrist_yaw" type="revolute">
  <limit lower="-0.5" upper="0.5" effort="20" velocity="0.8"/>
</joint>
```

## Complete Humanoid URDF with Proper Kinematics

Here's a more realistic humanoid robot model with proper kinematic structure:

```xml
<?xml version="1.0"?>
<robot name="advanced_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

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
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Pelvis (base link) -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia ixx="0.15" ixy="0" ixz="0" iyy="0.15" iyz="0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.25 0.1"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.25 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso with multiple segments for better kinematics -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso_lower"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="torso_lower">
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.2 0.3"/>
      </geometry>
      <material name="grey"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.2 0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="torso_upper_joint" type="revolute">
    <parent link="torso_lower"/>
    <child link="torso_upper"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Pitch rotation -->
    <limit lower="-0.3" upper="0.3" effort="100" velocity="0.5"/>
  </joint>

  <link name="torso_upper">
    <inertial>
      <mass value="6.0"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <inertia ixx="0.15" ixy="0" ixz="0" iyy="0.15" iyz="0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.2 0.3"/>
      </geometry>
      <material name="grey"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.2 0.3"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso_upper"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Neck pitch -->
    <limit lower="-0.5" upper="0.5" effort="20" velocity="1.0"/>
  </joint>

  <link name="head_link">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="skin"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Leg Chain -->
  <joint name="left_hip_yaw_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip_yaw_link"/>
    <origin xyz="0 0.1 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.3" upper="0.3" effort="200" velocity="1.0"/>
  </joint>

  <link name="left_hip_yaw_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_hip_roll_joint" type="revolute">
    <parent link="left_hip_yaw_link"/>
    <child link="left_hip_roll_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="250" velocity="1.0"/>
  </joint>

  <link name="left_hip_roll_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_hip_pitch_joint" type="revolute">
    <parent link="left_hip_roll_link"/>
    <child link="left_thigh_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="0.5" effort="250" velocity="1.5"/>
  </joint>

  <link name="left_thigh_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.08" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.08" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh_link"/>
    <child link="left_shin_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="2.35" effort="250" velocity="1.5"/>
  </joint>

  <link name="left_shin_link">
    <inertial>
      <mass value="4.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.08" ixy="0" ixz="0" iyy="0.08" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.07" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.07" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle_pitch_joint" type="revolute">
    <parent link="left_shin_link"/>
    <child link="left_ankle_roll_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_ankle_roll_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_ankle_roll_joint" type="revolute">
    <parent link="left_ankle_roll_link"/>
    <child link="left_foot_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.3" upper="0.3" effort="80" velocity="0.8"/>
  </joint>

  <link name="left_foot_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0.1 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0.1 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.1 0.1"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0.1 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Leg (mirrored) -->
  <joint name="right_hip_yaw_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip_yaw_link"/>
    <origin xyz="0 -0.1 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.3" upper="0.3" effort="200" velocity="1.0"/>
  </joint>

  <link name="right_hip_yaw_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_hip_roll_joint" type="revolute">
    <parent link="right_hip_yaw_link"/>
    <child link="right_hip_roll_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="250" velocity="1.0"/>
  </joint>

  <link name="right_hip_roll_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_hip_pitch_joint" type="revolute">
    <parent link="right_hip_roll_link"/>
    <child link="right_thigh_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="0.5" effort="250" velocity="1.5"/>
  </joint>

  <link name="right_thigh_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.08" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.08" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh_link"/>
    <child link="right_shin_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="2.35" effort="250" velocity="1.5"/>
  </joint>

  <link name="right_shin_link">
    <inertial>
      <mass value="4.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.08" ixy="0" ixz="0" iyy="0.08" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.07" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.07" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle_pitch_joint" type="revolute">
    <parent link="right_shin_link"/>
    <child link="right_ankle_roll_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_ankle_roll_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_ankle_roll_joint" type="revolute">
    <parent link="right_ankle_roll_link"/>
    <child link="right_foot_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.3" upper="0.3" effort="80" velocity="0.8"/>
  </joint>

  <link name="right_foot_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0.1 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0.1 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.1 0.1"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0.1 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Arm Chain -->
  <joint name="left_shoulder_yaw_joint" type="revolute">
    <parent link="torso_upper"/>
    <child link="left_shoulder_yaw_link"/>
    <origin xyz="0.15 0.12 0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_shoulder_yaw_link">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_pitch_joint" type="revolute">
    <parent link="left_shoulder_yaw_link"/>
    <child link="left_shoulder_pitch_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="100" velocity="1.2"/>
  </joint>

  <link name="left_shoulder_pitch_link">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_roll_joint" type="revolute">
    <parent link="left_shoulder_pitch_link"/>
    <child link="left_upper_arm_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="3.14" effort="80" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.2"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm_link"/>
    <child link="left_lower_arm_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="2.5" effort="80" velocity="1.2"/>
  </joint>

  <link name="left_lower_arm_link">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.003"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.15"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wrist_pitch_joint" type="revolute">
    <parent link="left_lower_arm_link"/>
    <child link="left_wrist_yaw_link"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.8" upper="0.8" effort="30" velocity="0.8"/>
  </joint>

  <link name="left_wrist_yaw_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>

  <joint name="left_wrist_yaw_joint" type="revolute">
    <parent link="left_wrist_yaw_link"/>
    <child link="left_hand_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="20" velocity="0.6"/>
  </joint>

  <link name="left_hand_link">
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.12 0.08 0.05"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.12 0.08 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Arm (mirrored) -->
  <joint name="right_shoulder_yaw_joint" type="revolute">
    <parent link="torso_upper"/>
    <child link="right_shoulder_yaw_link"/>
    <origin xyz="0.15 -0.12 0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_shoulder_yaw_link">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_pitch_joint" type="revolute">
    <parent link="right_shoulder_yaw_link"/>
    <child link="right_shoulder_pitch_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="100" velocity="1.2"/>
  </joint>

  <link name="right_shoulder_pitch_link">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_roll_joint" type="revolute">
    <parent link="right_shoulder_pitch_link"/>
    <child link="right_upper_arm_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="3.14" effort="80" velocity="1.0"/>
  </joint>

  <link name="right_upper_arm_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.2"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm_link"/>
    <child link="right_lower_arm_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="2.5" effort="80" velocity="1.2"/>
  </joint>

  <link name="right_lower_arm_link">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.003"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.15"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_wrist_pitch_joint" type="revolute">
    <parent link="right_lower_arm_link"/>
    <child link="right_wrist_yaw_link"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.8" upper="0.8" effort="30" velocity="0.8"/>
  </joint>

  <link name="right_wrist_yaw_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>

  <joint name="right_wrist_yaw_joint" type="revolute">
    <parent link="right_wrist_yaw_link"/>
    <child link="right_hand_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="20" velocity="0.6"/>
  </joint>

  <link name="right_hand_link">
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.12 0.08 0.05"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.12 0.08 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Transmission elements for ROS control -->
  <transmission name="left_hip_pitch_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_pitch_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_pitch_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Additional transmissions would be defined for all joints -->
  <!-- ... -->

</robot>
```

## Kinematic Considerations for Control

### 1. Forward Kinematics
The URDF structure enables forward kinematics - calculating the position and orientation of end effectors (hands, feet) based on joint angles.

### 2. Inverse Kinematics
For complex movements, inverse kinematics solutions are needed to determine joint angles for desired end effector positions.

### 3. Center of Mass
The distribution of mass in the URDF affects balance and stability:
```xml
<!-- Example of mass distribution considerations -->
<link name="torso_upper">
  <inertial>
    <mass value="6.0"/>  <!-- Heavier upper body affects balance -->
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <inertia ixx="0.15" ixy="0" ixz="0" iyy="0.15" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

## Biomechanical Accuracy

For realistic humanoid movement:

1. **Proportional scaling**: Maintain human-like proportions
2. **Joint coupling**: Consider how joints work together (e.g., shoulder girdle)
3. **Range limitations**: Set realistic joint limits based on human anatomy
4. **Mass distribution**: Accurate mass properties for stable movement

## Simulation Considerations

### 1. Collision Detection
Use simplified geometries for collision to improve simulation performance:
```xml
<!-- Complex visual shape -->
<visual>
  <geometry>
    <mesh filename="hand.dae"/>
  </geometry>
</visual>

<!-- Simple collision shape -->
<collision>
  <geometry>
    <box size="0.12 0.08 0.05"/>
  </geometry>
</collision>
```

### 2. Inertial Properties
Accurate inertial properties are crucial for realistic physics simulation:
- Mass values should reflect actual robot weight
- Inertia tensors should represent the actual mass distribution
- Center of mass should be accurately positioned

## Best Practices for Humanoid URDFs

1. **Modular Design**: Organize the robot into logical modules (torso, arms, legs)
2. **Consistent Naming**: Use clear, consistent naming conventions
3. **Realistic Limits**: Set joint limits based on mechanical and safety constraints
4. **Validation**: Regularly validate the URDF using ROS tools
5. **Documentation**: Comment complex sections and explain design choices
6. **Testing**: Test the model in simulation before hardware implementation

## Next Steps

Now that we understand how to structure a humanoid robot in URDF, let's look at how to visualize and validate these models in simulation environments.