---
sidebar_position: 3
---

# Exercise 3: Creating a Complete Humanoid Robot URDF

## Overview

In this exercise, you'll create a complete URDF (Unified Robot Description Format) model for a humanoid robot. This exercise combines the concepts learned in previous modules to build a realistic robot model that can be used in simulation and visualization tools.

## Learning Objectives

By completing this exercise, you will:
- Create a complete humanoid robot URDF with proper kinematic structure
- Define realistic joint limits and physical properties
- Add visual and collision geometries for each link
- Include Gazebo-specific plugins and materials
- Validate and test your URDF model

## Prerequisites

Before starting this exercise, ensure you have:
- ROS 2 Humble Hawksbill installed
- Basic understanding of URDF syntax
- Knowledge of 3D coordinate systems and transformations
- Completion of previous exercises (publishers and services)

## Step 1: Planning the Humanoid Robot Structure

Before creating the URDF, let's plan our humanoid robot:

- **Degrees of Freedom**: 24 DOF (12 for legs, 8 for arms, 4 for torso/head)
- **Segmented structure**: Pelvis, torso, head, 2 arms, 2 legs
- **Realistic proportions**: Based on adult human dimensions
- **Actuated joints**: All major joints will be controllable

## Step 2: Creating the Base URDF File

Create a comprehensive URDF file for your humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="advanced_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include any xacro files if needed -->
  <!-- <xacro:include filename="$(find my_robot_description)/urdf/materials.xacro" /> -->

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
  <material name="silver">
    <color rgba="0.75 0.75 0.75 1.0"/>
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

  <!-- Constants for dimensions -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_mass" value="10.0" /> <!-- Pelvis mass in kg -->
  <xacro:property name="torso_mass" value="15.0" />
  <xacro:property name="head_mass" value="4.0" />
  <xacro:property name="upper_leg_mass" value="6.0" />
  <xacro:property name="lower_leg_mass" value="4.0" />
  <xacro:property name="foot_mass" value="2.0" />
  <xacro:property name="upper_arm_mass" value="3.0" />
  <xacro:property name="lower_arm_mass" value="2.0" />
  <xacro:property name="hand_mass" value="1.0" />

  <!-- Base/Pelvis Link -->
  <link name="base_link">
    <inertial>
      <mass value="${base_mass}"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia
        ixx="0.2" ixy="0" ixz="0"
        iyy="0.2" iyz="0"
        izz="0.15"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.28 0.25 0.1"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.28 0.25 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso (lower) -->
  <joint name="torso_lower_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso_lower"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="torso_lower">
    <inertial>
      <mass value="${torso_mass * 0.4}"/>  <!-- 40% of torso mass -->
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <inertia
        ixx="0.15" ixy="0" ixz="0"
        iyy="0.15" iyz="0"
        izz="0.08"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.22 0.3"/>
      </geometry>
      <material name="grey"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.22 0.3"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso (upper) -->
  <joint name="torso_upper_joint" type="revolute">
    <parent link="torso_lower"/>
    <child link="torso_upper"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Pitch (forward/back) -->
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="100" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="torso_upper">
    <inertial>
      <mass value="${torso_mass * 0.6}"/>  <!-- 60% of torso mass -->
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <inertia
        ixx="0.12" ixy="0" ixz="0"
        iyy="0.12" iyz="0"
        izz="0.06"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.22 0.3"/>
      </geometry>
      <material name="grey"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.22 0.3"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso_upper"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Neck pitch -->
    <limit lower="${-M_PI/3}" upper="${M_PI/3}" effort="20" velocity="1.5"/>
    <dynamics damping="0.1" friction="0.05"/>
  </joint>

  <link name="head_link">
    <inertial>
      <mass value="${head_mass}"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia
        ixx="0.02" ixy="0" ixz="0"
        iyy="0.02" iyz="0"
        izz="0.02"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="skin_color"/>
    </visual>

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Hip (3 DOF) -->
  <joint name="left_hip_yaw_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip_yaw_link"/>
    <origin xyz="0 0.12 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>  <!-- Yaw (rotation) -->
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="200" velocity="1.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="left_hip_yaw_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_hip_roll_joint" type="revolute">
    <parent link="left_hip_yaw_link"/>
    <child link="left_hip_roll_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- Roll (abduction/adduction) -->
    <limit lower="${-M_PI/4}" upper="${M_PI/2}" effort="250" velocity="1.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="left_hip_roll_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_hip_pitch_joint" type="revolute">
    <parent link="left_hip_roll_link"/>
    <child link="left_thigh_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Pitch (flexion/extension) -->
    <limit lower="${-M_PI/2}" upper="${M_PI/4}" effort="250" velocity="1.5"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <!-- Left Thigh -->
  <link name="left_thigh_link">
    <inertial>
      <mass value="${upper_leg_mass}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia
        ixx="0.1" ixy="0" ixz="0"
        iyy="0.1" iyz="0"
        izz="0.02"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.08" length="0.35"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.08" length="0.35"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Knee -->
  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh_link"/>
    <child link="left_shin_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Pitch -->
    <limit lower="0.0" upper="${M_PI*0.9}" effort="250" velocity="1.5"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="left_shin_link">
    <inertial>
      <mass value="${lower_leg_mass}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia
        ixx="0.08" ixy="0" ixz="0"
        iyy="0.08" iyz="0"
        izz="0.015"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.07" length="0.35"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.07" length="0.35"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Ankle -->
  <joint name="left_ankle_pitch_joint" type="revolute">
    <parent link="left_shin_link"/>
    <child link="left_ankle_roll_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Pitch (dorsiflexion/plantarflexion) -->
    <limit lower="${-M_PI/3}" upper="${M_PI/3}" effort="100" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="left_ankle_roll_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_ankle_roll_joint" type="revolute">
    <parent link="left_ankle_roll_link"/>
    <child link="left_foot_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- Roll (inversion/eversion) -->
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="80" velocity="0.8"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <!-- Left Foot -->
  <link name="left_foot_link">
    <inertial>
      <mass value="${foot_mass}"/>
      <origin xyz="0.08 0 -0.02" rpy="0 0 0"/>
      <inertia
        ixx="0.01" ixy="0" ixz="0"
        iyy="0.02" iyz="0"
        izz="0.02"/>
    </inertial>

    <visual>
      <origin xyz="0.08 0 -0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.12 0.08"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0.08 0 -0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.12 0.08"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Leg (mirror of left) -->
  <joint name="right_hip_yaw_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip_yaw_link"/>
    <origin xyz="0 -0.12 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="200" velocity="1.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="right_hip_yaw_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_hip_roll_joint" type="revolute">
    <parent link="right_hip_yaw_link"/>
    <child link="right_hip_roll_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/4}" effort="250" velocity="1.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="right_hip_roll_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_hip_pitch_joint" type="revolute">
    <parent link="right_hip_roll_link"/>
    <child link="right_thigh_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/4}" effort="250" velocity="1.5"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="right_thigh_link">
    <inertial>
      <mass value="${upper_leg_mass}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia
        ixx="0.1" ixy="0" ixz="0"
        iyy="0.1" iyz="0"
        izz="0.02"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.08" length="0.35"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.08" length="0.35"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh_link"/>
    <child link="right_shin_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0.0" upper="${M_PI*0.9}" effort="250" velocity="1.5"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <link name="right_shin_link">
    <inertial>
      <mass value="${lower_leg_mass}"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia
        ixx="0.08" ixy="0" ixz="0"
        iyy="0.08" iyz="0"
        izz="0.015"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.07" length="0.35"/>
      </geometry>
      <material name="blue"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.07" length="0.35"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle_pitch_joint" type="revolute">
    <parent link="right_shin_link"/>
    <child link="right_ankle_roll_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/3}" upper="${M_PI/3}" effort="100" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="right_ankle_roll_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_ankle_roll_joint" type="revolute">
    <parent link="right_ankle_roll_link"/>
    <child link="right_foot_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI/6}" upper="${M_PI/6}" effort="80" velocity="0.8"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <link name="right_foot_link">
    <inertial>
      <mass value="${foot_mass}"/>
      <origin xyz="0.08 0 -0.02" rpy="0 0 0"/>
      <inertia
        ixx="0.01" ixy="0" ixz="0"
        iyy="0.02" iyz="0"
        izz="0.02"/>
    </inertial>

    <visual>
      <origin xyz="0.08 0 -0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.12 0.08"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0.08 0 -0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.25 0.12 0.08"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Shoulder (3 DOF) -->
  <joint name="left_shoulder_pitch_joint" type="revolute">
    <parent link="torso_upper"/>
    <child link="left_shoulder_pitch_link"/>
    <origin xyz="0.16 0.08 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Pitch -->
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1.5"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="left_shoulder_pitch_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_roll_joint" type="revolute">
    <parent link="left_shoulder_pitch_link"/>
    <child link="left_upper_arm_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>  <!-- Roll -->
    <limit lower="0" upper="${M_PI*0.8}" effort="80" velocity="1.2"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <!-- Left Upper Arm -->
  <link name="left_upper_arm_link">
    <inertial>
      <mass value="${upper_arm_mass}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia
        ixx="0.02" ixy="0" ixz="0"
        iyy="0.02" iyz="0"
        izz="0.005"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.25"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.25"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Elbow -->
  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm_link"/>
    <child link="left_lower_arm_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Pitch -->
    <limit lower="0" upper="${M_PI*0.9}" effort="80" velocity="1.2"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <!-- Left Lower Arm -->
  <link name="left_lower_arm_link">
    <inertial>
      <mass value="${lower_arm_mass}"/>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <inertia
        ixx="0.01" ixy="0" ixz="0"
        iyy="0.01" iyz="0"
        izz="0.003"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Wrist -->
  <joint name="left_wrist_pitch_joint" type="revolute">
    <parent link="left_lower_arm_link"/>
    <child link="left_wrist_yaw_link"/>
    <origin xyz="0 0 -0.24" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>  <!-- Pitch -->
    <limit lower="${-M_PI/3}" upper="${M_PI/3}" effort="30" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.02"/>
  </joint>

  <link name="left_wrist_yaw_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_wrist_yaw_joint" type="revolute">
    <parent link="left_wrist_yaw_link"/>
    <child link="left_hand_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>  <!-- Yaw -->
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="20" velocity="0.8"/>
    <dynamics damping="0.05" friction="0.01"/>
  </joint>

  <!-- Left Hand -->
  <link name="left_hand_link">
    <inertial>
      <mass value="${hand_mass}"/>
      <origin xyz="0.06 0 0" rpy="0 0 0"/>
      <inertia
        ixx="0.002" ixy="0" ixz="0"
        iyy="0.003" iyz="0"
        izz="0.003"/>
    </inertial>

    <visual>
      <origin xyz="0.06 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.12 0.08 0.06"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0.06 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.12 0.08 0.06"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Arm (mirror of left) -->
  <joint name="right_shoulder_pitch_joint" type="revolute">
    <parent link="torso_upper"/>
    <child link="right_shoulder_pitch_link"/>
    <origin xyz="0.16 -0.08 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1.5"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="right_shoulder_pitch_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_roll_joint" type="revolute">
    <parent link="right_shoulder_pitch_link"/>
    <child link="right_upper_arm_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-M_PI*0.8}" upper="0" effort="80" velocity="1.2"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <link name="right_upper_arm_link">
    <inertial>
      <mass value="${upper_arm_mass}"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia
        ixx="0.02" ixy="0" ixz="0"
        iyy="0.02" iyz="0"
        izz="0.005"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.25"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.06" length="0.25"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm_link"/>
    <child link="right_lower_arm_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="${M_PI*0.9}" effort="80" velocity="1.2"/>
    <dynamics damping="0.3" friction="0.05"/>
  </joint>

  <link name="right_lower_arm_link">
    <inertial>
      <mass value="${lower_arm_mass}"/>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <inertia
        ixx="0.01" ixy="0" ixz="0"
        iyy="0.01" iyz="0"
        izz="0.003"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
      <material name="green"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.12" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_wrist_pitch_joint" type="revolute">
    <parent link="right_lower_arm_link"/>
    <child link="right_wrist_yaw_link"/>
    <origin xyz="0 0 -0.24" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/3}" upper="${M_PI/3}" effort="30" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.02"/>
  </joint>

  <link name="right_wrist_yaw_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_wrist_yaw_joint" type="revolute">
    <parent link="right_wrist_yaw_link"/>
    <child link="right_hand_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="20" velocity="0.8"/>
    <dynamics damping="0.05" friction="0.01"/>
  </joint>

  <link name="right_hand_link">
    <inertial>
      <mass value="${hand_mass}"/>
      <origin xyz="0.06 0 0" rpy="0 0 0"/>
      <inertia
        ixx="0.002" ixy="0" ixz="0"
        iyy="0.003" iyz="0"
        izz="0.003"/>
    </inertial>

    <visual>
      <origin xyz="0.06 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.12 0.08 0.06"/>
      </geometry>
      <material name="white"/>
    </visual>

    <collision>
      <origin xyz="0.06 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.12 0.08 0.06"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Gazebo materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <gazebo reference="torso_lower">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="torso_upper">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="head_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <!-- Additional Gazebo configurations for each link -->
  <gazebo reference="left_thigh_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="right_thigh_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_shin_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="right_shin_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_foot_link">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="right_foot_link">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="left_upper_arm_link">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="right_upper_arm_link">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="left_lower_arm_link">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="right_lower_arm_link">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="left_hand_link">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="right_hand_link">
    <material>Gazebo/White</material>
  </gazebo>

</robot>
```

## Step 3: Creating a Validation Script

Create a Python script to validate your URDF:

```python
#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from urdf_parser_py.urdf import URDF
import sys
import math

def validate_urdf_structure(urdf_path):
    """Validate the URDF structure and properties"""
    try:
        # Parse the URDF file
        robot = URDF.from_xml_file(urdf_path)

        print(f"Robot name: {robot.name}")
        print(f"Number of links: {len(robot.links)}")
        print(f"Number of joints: {len(robot.joints)}")

        # Check for common issues
        issues = []

        # 1. Check if all joints have valid parent and child links
        for joint in robot.joints:
            if joint.parent not in [link.name for link in robot.links]:
                issues.append(f"Joint {joint.name} has invalid parent link: {joint.parent}")
            if joint.child not in [link.name for link in robot.links]:
                issues.append(f"Joint {joint.name} has invalid child link: {joint.child}")

        # 2. Check for links without visual or collision geometry
        for link in robot.links:
            has_visual = link.visual is not None
            has_collision = link.collision is not None
            if not has_visual and not has_collision:
                issues.append(f"Link {link.name} has neither visual nor collision geometry")

        # 3. Check for links without inertial properties
        for link in robot.links:
            has_inertial = link.inertial is not None
            if not has_inertial:
                issues.append(f"Link {link.name} has no inertial properties")
            else:
                # Check if mass is positive
                if link.inertial.mass <= 0:
                    issues.append(f"Link {link.name} has non-positive mass: {link.inertial.mass}")

        # 4. Check joint limits
        for joint in robot.joints:
            if joint.limit:
                if joint.limit.lower >= joint.limit.upper:
                    issues.append(f"Joint {joint.name} has invalid limits: lower({joint.limit.lower}) >= upper({joint.limit.upper})")

        # 5. Check for potential kinematic loops
        # This is a simplified check - in a proper humanoid, each link (except the base) should have exactly one parent
        children_count = {}
        for joint in robot.joints:
            if joint.child in children_count:
                children_count[joint.child] += 1
            else:
                children_count[joint.child] = 1

        for link_name, count in children_count.items():
            if count > 1:
                issues.append(f"Link {link_name} has multiple parents (kinematic loop): {count} parents")

        # 6. Validate humanoid-specific structure
        expected_joints = {
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_elbow_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_joint'
        }

        actual_joints = {joint.name for joint in robot.joints}
        missing_joints = expected_joints - actual_joints
        if missing_joints:
            issues.append(f"Missing expected humanoid joints: {missing_joints}")

        # Print results
        if issues:
            print("\n❌ Issues found:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            return False
        else:
            print("\n✅ URDF validation passed!")
            return True

    except Exception as e:
        print(f"❌ Error validating URDF: {e}")
        return False

def print_humanoid_summary(urdf_path):
    """Print a summary of the humanoid robot structure"""
    try:
        robot = URDF.from_xml_file(urdf_path)

        print(f"\n🤖 Humanoid Robot Summary for: {robot.name}")
        print("=" * 50)

        # Count different types of joints
        joint_types = {}
        for joint in robot.joints:
            jtype = joint.type
            joint_types[jtype] = joint_types.get(jtype, 0) + 1

        print(f"Joint Types: {joint_types}")

        # Identify body parts
        left_joints = [j.name for j in robot.joints if j.name.startswith('left_')]
        right_joints = [j.name for j in robot.joints if j.name.startswith('right_')]
        other_joints = [j.name for j in robot.joints if not j.name.startswith(('left_', 'right_'))]

        print(f"Left side joints: {len(left_joints)}")
        print(f"Right side joints: {len(right_joints)}")
        print(f"Center joints: {len(other_joints)}")

        # Calculate total mass
        total_mass = sum(link.inertial.mass for link in robot.links if link.inertial)
        print(f"Total robot mass: {total_mass:.2f} kg")

        # Check for actuated joints (revolute joints)
        actuated_joints = [j.name for j in robot.joints if j.type == 'revolute']
        print(f"Actuated joints: {len(actuated_joints)}")

    except Exception as e:
        print(f"Error creating summary: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_urdf.py <urdf_file>")
        sys.exit(1)

    urdf_file = sys.argv[1]

    print("Validating Humanoid Robot URDF...")
    is_valid = validate_urdf_structure(urdf_file)

    if is_valid:
        print_humanoid_summary(urdf_file)
```

## Step 4: Creating a Visualization Launch File

Create a launch file to visualize your robot:

```python
# humanoid_publisher_examples/launch/view_humanoid.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    urdf_model_path = LaunchConfiguration('model')
    use_gui = LaunchConfiguration('gui')

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
                FindPackageShare('humanoid_publisher_examples'),
                LaunchConfiguration('model')
            ])
        }]
    )

    # Joint State Publisher node
    joint_state_publisher_node = Node(
        condition=IfCondition(use_gui),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui'
    )

    # Joint State Publisher (non-GUI version as backup)
    joint_state_publisher_node_backup = Node(
        condition=IfCondition('not ' + use_gui),
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
            FindPackageShare('humanoid_publisher_examples'),
            'rviz',
            'view_humanoid.rviz'
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

## Step 5: Creating an RViz Configuration

Create an RViz configuration file to properly visualize your humanoid:

```yaml
# humanoid_publisher_examples/rviz/view_humanoid.rviz
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
        - /Grid1
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
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: true
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
      Distance: 3
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0.5
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

## Step 6: Testing Your URDF

1. **Validate the URDF syntax**:
```bash
# Check URDF syntax
check_urdf /path/to/your/humanoid.urdf

# Use your validation script
python3 validate_urdf.py /path/to/your/humanoid.urdf
```

2. **Visualize in RViz**:
```bash
# Create the URDF file in your package
mkdir -p ~/humanoid_ws/src/humanoid_publisher_examples/urdf
# Save the URDF content to humanoid.urdf

# Build and source
cd ~/humanoid_ws
colcon build --packages-select humanoid_publisher_examples
source install/setup.bash

# Launch visualization
ros2 launch humanoid_publisher_examples view_humanoid.launch.py
```

## Step 7: Integration with Previous Exercises

Now let's create a script that publishes joint states for your new robot model:

```python
# humanoid_publisher_examples/humanoid_demo_publisher.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class HumanoidDemoPublisher(Node):
    def __init__(self):
        super().__init__('humanoid_demo_publisher')

        # Create publisher for joint states
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Timer for publishing (50 Hz)
        self.timer = self.create_timer(0.02, self.timer_callback)

        # Define all humanoid joints
        self.joint_names = [
            # Torso
            'torso_upper_joint',

            # Neck
            'neck_joint',

            # Left leg (6 DOF)
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint', 'left_ankle_roll_joint',

            # Right leg (6 DOF)
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint', 'right_ankle_roll_joint',

            # Left arm (6 DOF)
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_elbow_joint',
            'left_wrist_pitch_joint', 'left_wrist_yaw_joint',

            # Right arm (6 DOF)
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_joint',
            'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]

        # Initialize all joint positions
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

        # Time counter for animations
        self.time_counter = 0.0

        self.get_logger().info(f'Humanoid Demo Publisher initialized with {len(self.joint_names)} joints')

    def timer_callback(self):
        """Publish joint states with various movement patterns"""
        # Update joint positions based on time
        self.time_counter += 0.02

        # Walking gait simulation for legs
        walk_phase = math.sin(2 * math.pi * 0.5 * self.time_counter)

        # Left leg
        self.joint_positions[self.joint_names.index('left_hip_pitch_joint')] = 0.2 * walk_phase
        self.joint_positions[self.joint_names.index('left_knee_joint')] = 0.4 * walk_phase
        self.joint_positions[self.joint_names.index('left_ankle_pitch_joint')] = -0.2 * walk_phase

        # Right leg (opposite phase)
        self.joint_positions[self.joint_names.index('right_hip_pitch_joint')] = 0.2 * -walk_phase
        self.joint_positions[self.joint_names.index('right_knee_joint')] = 0.4 * -walk_phase
        self.joint_positions[self.joint_names.index('right_ankle_pitch_joint')] = -0.2 * -walk_phase

        # Arm swinging opposite to legs
        self.joint_positions[self.joint_names.index('left_shoulder_pitch_joint')] = 0.1 * -walk_phase
        self.joint_positions[self.joint_names.index('right_shoulder_pitch_joint')] = 0.1 * walk_phase

        # Add small oscillations to other joints
        for i, joint_name in enumerate(self.joint_names):
            if joint_name not in [
                'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint',
                'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint',
                'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint'
            ]:
                # Small random movement for other joints
                self.joint_positions[i] = 0.05 * math.sin(0.3 * self.time_counter + i * 0.2)

        # Create and publish the message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        self.publisher.publish(msg)

        # Log periodically
        if int(self.time_counter * 50) % 100 == 0:
            self.get_logger().info('Published humanoid joint states')

def main(args=None):
    rclpy.init(args=args)

    try:
        demo_publisher = HumanoidDemoPublisher()
        demo_publisher.get_logger().info('Starting humanoid demo publisher...')

        rclpy.spin(demo_publisher)

    except KeyboardInterrupt:
        print('\nShutting down humanoid demo publisher...')
    finally:
        demo_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 8: Adding the Demo Publisher to setup.py

Add the new executable to your `setup.py`:

```python
# Add to entry_points in setup.py
entry_points={
    'console_scripts': [
        'simple_joint_publisher = humanoid_publisher_examples.simple_joint_publisher:main',
        'enhanced_joint_publisher = humanoid_publisher_examples.enhanced_joint_publisher:main',
        'joint_control_server = humanoid_publisher_examples.joint_control_server:main',
        'robust_control_server = humanoid_publisher_examples.robust_control_server:main',
        'joint_control_client = humanoid_publisher_examples.joint_control_client:main',
        'manual_control_client = humanoid_publisher_examples.manual_control_client:main',
        'humanoid_demo_publisher = humanoid_publisher_examples.humanoid_demo_publisher:main',
    ],
},
```

## Step 9: Running the Complete System

1. **Build the package**:
```bash
cd ~/humanoid_ws
colcon build --packages-select humanoid_publisher_examples
source install/setup.bash
```

2. **Run the visualization with demo publisher**:
```bash
# Terminal 1: Launch RViz with the robot model
ros2 launch humanoid_publisher_examples view_humanoid.launch.py

# Terminal 2: Run the demo publisher
ros2 run humanoid_publisher_examples humanoid_demo_publisher
```

## Troubleshooting Tips

1. **URDF Validation Errors**: Use `check_urdf` to identify syntax errors
2. **Missing Joint States**: Make sure the joint names in your publisher match the URDF
3. **Visualization Issues**: Check that the robot description parameter is properly set
4. **Gazebo Integration**: Ensure Gazebo plugins are properly configured

## Next Steps

After completing this exercise, you should have:
- A complete humanoid robot URDF model
- Validation tools to check your model
- Visualization setup for development
- Integration with your control systems

In the next module, we'll explore how to simulate this humanoid robot in Gazebo and connect it with ROS 2 controllers.