---
sidebar_position: 2
---

# Physics Modeling for Humanoid Robots

## Overview

Physics modeling is crucial for creating realistic humanoid robot simulations in Gazebo. The physics engine determines how your robot interacts with the environment, how forces are applied, and how movements are constrained. This section covers the essential physics concepts and configurations needed for accurate humanoid simulation.

## Understanding Gazebo Physics Engines

Gazebo supports several physics engines, but for humanoid robots, Open Dynamics Engine (ODE) is most commonly used due to its stability and performance characteristics.

### ODE Physics Configuration

The physics configuration significantly affects humanoid robot simulation:

```xml
<physics type="ode">
  <!-- Time step settings -->
  <max_step_size>0.001</max_step_size>      <!-- 1ms time step (1000 Hz) -->
  <real_time_factor>1.0</real_time_factor>  <!-- Real-time simulation -->
  <real_time_update_rate>1000.0</real_time_update_rate>  <!-- Update rate -->

  <!-- ODE-specific parameters -->
  <ode>
    <!-- Solver settings -->
    <solver>
      <type>quick</type>     <!-- QuickStep solver for better performance -->
      <iters>100</iters>     <!-- Solver iterations (higher = more accurate but slower) -->
      <sor>1.3</sor>         <!-- Successive Over-Relaxation parameter -->
    </solver>

    <!-- Constraint settings for stable contacts -->
    <constraints>
      <cfm>0.000001</cfm>                    <!-- Constraint Force Mixing -->
      <erp>0.2</erp>                        <!-- Error Reduction Parameter -->
      <contact_max_correcting_vel>100</contact_max_correcting_vel>  <!-- Max correction velocity -->
      <contact_surface_layer>0.001</contact_surface_layer>          <!-- Contact surface layer -->
    </constraints>
  </ode>
</physics>
```

## Key Physics Parameters for Humanoid Robots

### 1. Time Step Considerations

For humanoid robots, the time step is critical for stable control:

```xml
<!-- For precise control, use smaller time steps -->
<max_step_size>0.001</max_step_size>  <!-- 1ms - good for control systems -->
<!-- OR -->
<max_step_size>0.0005</max_step_size>  <!-- 0.5ms - more precise but slower -->

<!-- For less critical simulations -->
<max_step_size>0.01</max_step_size>   <!-- 10ms - faster but less precise -->
```

### 2. Solver Configuration

The solver configuration affects how contacts and constraints are resolved:

```xml
<solver>
  <!-- For humanoid robots with many joints -->
  <type>quick</type>
  <iters>100</iters>        <!-- Start with 100, increase if unstable -->
  <sor>1.3</sor>            <!-- Usually between 1.0-1.5 -->

  <!-- Alternative for more stability (slower) -->
  <!--
  <iters>200</iters>
  <sor>1.0</sor>
  -->
</solver>
```

### 3. Constraint Parameters

These parameters are crucial for stable foot contacts:

```xml
<constraints>
  <!-- CFM (Constraint Force Mixing) - lower values = stiffer constraints -->
  <cfm>0.000001</cfm>       <!-- Very low for stable contacts -->

  <!-- ERP (Error Reduction Parameter) - how fast errors are corrected -->
  <erp>0.2</erp>            <!-- 0.1-0.8 typically, 0.2 is good start -->

  <!-- Maximum velocity for contact correction -->
  <contact_max_correcting_vel>100</contact_max_correcting_vel>

  <!-- Surface layer thickness to prevent deep penetration -->
  <contact_surface_layer>0.001</contact_surface_layer>  <!-- 1mm -->
</constraints>
```

## Link Physics Properties

Each link in your humanoid robot needs proper physics properties:

### 1. Inertial Properties

Accurate inertial properties are essential for realistic simulation:

```xml
<link name="left_thigh_link">
  <inertial>
    <!-- Mass in kg -->
    <mass>5.0</mass>

    <!-- Origin offset (usually center of mass) -->
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>

    <!-- Inertia matrix (calculated based on geometry) -->
    <inertia
      ixx="0.1" ixy="0.0" ixz="0.0"
      iyy="0.1" iyz="0.0"
      izz="0.02"/>
  </inertial>

  <!-- Visual properties -->
  <visual name="visual">
    <geometry>
      <capsule radius="0.08" length="0.35"/>
    </geometry>
  </visual>

  <!-- Collision properties -->
  <collision name="collision">
    <geometry>
      <capsule radius="0.08" length="0.35"/>
    </geometry>
  </collision>
</link>
```

### 2. Surface Properties for Contacts

Surface properties affect how your robot interacts with the environment:

```xml
<link name="left_foot_link">
  <inertial>
    <mass>2.0</mass>
    <origin xyz="0.08 0 -0.02" rpy="0 0 0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
  </inertial>

  <collision name="collision">
    <geometry>
      <box size="0.25 0.12 0.08"/>
    </geometry>

    <!-- Surface properties for foot-ground contact -->
    <surface>
      <friction>
        <ode>
          <mu>0.8</mu>      <!-- Coefficient of friction -->
          <mu2>0.8</mu2>    <!-- Secondary friction coefficient -->
          <fdir1>0 0 0</fdir1>  <!-- Friction direction -->
        </ode>
      </friction>

      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>  <!-- Bounciness -->
        <threshold>100000</threshold>  <!-- Velocity threshold for bounce -->
      </bounce>

      <contact>
        <ode>
          <soft_cfm>0.000001</soft_cfm>      <!-- Soft constraint force mixing -->
          <soft_erp>0.2</soft_erp>           <!-- Soft error reduction parameter -->
          <kp>1000000000000.0</kp>          <!-- Contact stiffness -->
          <kd>1.0</kd>                      <!-- Contact damping -->
          <max_vel>100.0</max_vel>          <!-- Maximum contact correction velocity -->
          <min_depth>0.001</min_depth>      <!-- Minimum contact depth -->
        </ode>
      </contact>
    </surface>
  </collision>

  <visual name="visual">
    <geometry>
      <box size="0.25 0.12 0.08"/>
    </geometry>
  </visual>
</link>
```

## Joint Physics Configuration

Joints need proper dynamics for realistic movement:

### 1. Revolute Joint with Dynamics

```xml
<joint name="left_knee_joint" type="revolute">
  <parent link="left_thigh_link"/>
  <child link="left_shin_link"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Knee rotation axis -->

  <!-- Joint limits -->
  <limit lower="0.0" upper="2.35" effort="250" velocity="1.5"/>

  <!-- Joint dynamics -->
  <dynamics damping="1.0" friction="0.2"/>
</joint>
```

### 2. Advanced Joint Dynamics

For more realistic joint behavior:

```xml
<joint name="left_shoulder_joint" type="revolute">
  <parent link="torso_upper"/>
  <child link="left_upper_arm_link"/>
  <origin xyz="0.16 0.08 0.25" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="2.0" effort="100" velocity="1.5"/>

  <!-- Detailed dynamics for realistic movement -->
  <dynamics
    damping="0.5"           <!-- Viscous damping coefficient -->
    friction="0.1"          <!-- Static friction coefficient -->
    spring_reference="0.0"  <!-- Spring reference angle -->
    spring_stiffness="0.0"  <!-- Spring stiffness -->
  />
</joint>
```

## Physics Optimization for Humanoid Robots

### 1. Balancing Accuracy and Performance

```xml
<!-- Optimized physics for humanoid simulation -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>

  <ode>
    <solver>
      <!-- Good balance for humanoid: 50-100 iterations -->
      <type>quick</type>
      <iters>80</iters>
      <sor>1.2</sor>
    </solver>

    <constraints>
      <!-- Stable values for humanoid contacts -->
      <cfm>0.00001</cfm>    <!-- Slightly higher for performance -->
      <erp>0.2</erp>
      <contact_max_correcting_vel>10</contact_max_correcting_vel>  <!-- Lower for stability -->
      <contact_surface_layer>0.002</contact_surface_layer>         <!-- Slightly thicker -->
    </constraints>
  </ode>
</physics>
```

### 2. Per-Link Physics Optimization

For complex humanoid robots, you might need different physics settings for different parts:

```xml
<!-- Light, fast-moving parts (hands, feet) -->
<link name="left_hand_link">
  <inertial>
    <mass>0.8</mass>  <!-- Lower mass for faster response -->
    <origin xyz="0.06 0 0" rpy="0 0 0"/>
    <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
  </inertial>

  <collision name="collision">
    <geometry>
      <box size="0.12 0.08 0.06"/>
    </geometry>
    <surface>
      <contact>
        <ode>
          <soft_cfm>0.00001</soft_cfm>
          <soft_erp>0.1</soft_erp>  <!-- Lower ERP for precision -->
        </ode>
      </contact>
    </surface>
  </collision>
</link>

<!-- Heavy, stable parts (torso, pelvis) -->
<link name="base_link">
  <inertial>
    <mass>12.0</mass>  <!-- Higher mass for stability -->
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <inertia ixx="0.25" ixy="0" ixz="0" iyy="0.25" iyz="0" izz="0.2"/>
  </inertial>

  <collision name="collision">
    <geometry>
      <box size="0.28 0.25 0.1"/>
    </geometry>
    <surface>
      <contact>
        <ode>
          <soft_cfm>0.0001</soft_cfm>  <!-- Higher CFM for stability -->
          <soft_erp>0.3</soft_erp>     <!-- Higher ERP for stability -->
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

## Advanced Physics Concepts

### 1. Multi-Body Dynamics

For complex humanoid robots with many interconnected parts:

```xml
<!-- Example of a complex chain -->
<model name="humanoid_with_physics">
  <!-- Physics properties for the entire model can be set here -->
  <static>false</static>  <!-- Model is dynamic -->

  <!-- Links with appropriate physics as shown above -->
  <!-- ... -->

  <!-- Self-collision can be enabled if needed -->
  <self_collide>false</self_collide>  <!-- Usually false for humanoid -->

  <!-- Enable/disable gravity for the entire model -->
  <gravity>true</gravity>
</model>
```

### 2. Contact Stabilization

For stable walking and standing:

```xml
<!-- In your world file -->
<world name="humanoid_stable_world">
  <physics type="ode">
    <!-- Stable physics configuration -->
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>

    <ode>
      <solver>
        <type>quick</type>
        <iters>100</iters>  <!-- Higher for stability -->
        <sor>1.0</sor>      <!-- Lower SOR for more accurate solution -->
      </solver>

      <constraints>
        <cfm>0.000001</cfm>  <!-- Very low for stiff contacts -->
        <erp>0.1</erp>      <!-- Lower ERP for less aggressive error correction -->
        <contact_max_correcting_vel>5</contact_max_correcting_vel>  <!-- Lower for stability -->
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>

  <!-- Your robot and environment models -->
</world>
```

## Debugging Physics Issues

### 1. Common Physics Problems

**Problem: Robot falls through ground**
- Check if ground plane is static
- Verify collision geometry exists
- Check physics parameters (CFM/ERP)

**Problem: Robot jitters or shakes**
- Increase solver iterations
- Adjust CFM/ERP values
- Check joint limits and dynamics

**Problem: Robot moves too slowly or not at all**
- Check effort limits in joints
- Verify controller commands are being sent
- Check friction values

### 2. Physics Debug Visualization

Add this to your world file for physics debugging:

```xml
<!-- In your world file, add physics visualization -->
<world name="debug_world">
  <!-- ... physics settings ... -->

  <!-- Enable contact visualization -->
  <gui>
    <camera name="user_camera">
      <pose>5 -5 2 0 0.5 1.5708</pose>
    </camera>
  </gui>

  <!-- Physics engine settings -->
  <physics type="ode">
    <!-- Your physics settings -->
  </physics>
</world>
```

## Performance Considerations

### 1. Optimizing for Real-time Simulation

```xml
<!-- For real-time control applications -->
<physics type="ode">
  <max_step_size>0.002</max_step_size>  <!-- Larger time step for performance -->
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>500.0</real_time_update_rate>  <!-- Lower update rate -->

  <ode>
    <solver>
      <type>quick</type>
      <iters>50</iters>  <!-- Lower iterations for speed -->
      <sor>1.3</sor>
    </solver>

    <constraints>
      <cfm>0.0001</cfm>  <!-- Higher CFM for performance -->
      <erp>0.3</erp>    <!-- Higher ERP for performance -->
    </constraints>
  </ode>
</physics>
```

### 2. Optimizing for Accuracy

```xml
<!-- For high-precision simulation -->
<physics type="ode">
  <max_step_size>0.0005</max_step_size>  <!-- Very small time step -->
  <real_time_factor>0.5</real_time_factor>  <!-- Allow slower than real-time -->
  <real_time_update_rate>2000.0</real_time_update_rate>

  <ode>
    <solver>
      <type>quick</type>
      <iters>200</iters>  <!-- High iterations for accuracy -->
      <sor>1.0</sor>
    </solver>

    <constraints>
      <cfm>0.0000001</cfm>  <!-- Very low CFM for stiffness -->
      <erp>0.1</erp>       <!-- Low ERP for precision -->
    </constraints>
  </ode>
</physics>
```

## Testing Physics Configurations

Create a simple test script to validate your physics setup:

```python
#!/usr/bin/env python3
# physics_test.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import PointStamped
import time

class PhysicsTestNode(Node):
    def __init__(self):
        super().__init__('physics_test_node')

        # Subscribe to joint states to monitor robot stability
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        # Subscribe to IMU data for balance monitoring
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Timer for periodic checks
        self.timer = self.create_timer(1.0, self.periodic_check)

        self.joint_states = None
        self.imu_data = None
        self.check_count = 0

        self.get_logger().info('Physics test node initialized')

    def joint_callback(self, msg):
        self.joint_states = msg

    def imu_callback(self, msg):
        self.imu_data = msg

    def periodic_check(self):
        self.check_count += 1

        if self.joint_states:
            # Check for reasonable joint positions (not flying away)
            max_pos = max(abs(p) for p in self.joint_states.position)
            if max_pos > 10.0:  # Unreasonable position
                self.get_logger().warn(f'Large joint position detected: {max_pos}')

            # Check for reasonable velocities
            if self.joint_states.velocity:
                max_vel = max(abs(v) for v in self.joint_states.velocity)
                if max_vel > 100.0:  # Unreasonable velocity
                    self.get_logger().warn(f'Large joint velocity detected: {max_vel}')

        if self.imu_data:
            # Check if robot is upright (simplified)
            z_orientation = self.imu_data.orientation.z
            if abs(z_orientation) > 0.7:  # Robot might be falling
                self.get_logger().info(f'Robot orientation: {z_orientation}')

        if self.check_count % 10 == 0:  # Every 10 seconds
            self.get_logger().info(f'Physics check #{self.check_count}: OK')

def main(args=None):
    rclpy.init(args=args)
    node = PhysicsTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Physics test node shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Next Steps

Now that you understand physics modeling for humanoid robots, let's explore how to integrate your simulation with ROS 2 through Gazebo plugins. In the next section, we'll cover the Gazebo-ROS integration that allows your simulated humanoid robot to communicate with the ROS 2 ecosystem.