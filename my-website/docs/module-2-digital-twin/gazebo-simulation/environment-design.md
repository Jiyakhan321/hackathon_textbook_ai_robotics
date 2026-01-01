---
sidebar_position: 1
---

# Environment Design in Gazebo

## Overview

Creating effective simulation environments in Gazebo is crucial for developing and testing humanoid robots. In this section, we'll explore how to design realistic environments that accurately represent real-world conditions, with attention to physics properties, visual elements, and interaction dynamics.

## Gazebo World Structure

A Gazebo world is defined using SDF (Simulation Description Format), an XML-based format similar to URDF but designed for environments:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_world">
    <!-- World properties -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Environment elements -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom models and objects -->
    <!-- Your environment objects go here -->

  </world>
</sdf>
```

## Basic Environment Elements

### 1. Ground Plane
The ground plane provides the basic surface for your robot to interact with:

```xml
<include>
  <uri>model://ground_plane</uri>
</include>

<!-- Or create a custom ground plane -->
<model name="custom_ground">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>100 100</size>
        </plane>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>1.0</mu>
            <mu2>1.0</mu2>
          </ode>
        </friction>
      </surface>
    </collision>
    <visual name="visual">
      <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>100 100</size>
        </plane>
      </geometry>
      <material>
        <ambient>0.7 0.7 0.7 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
        <specular>0.0 0.0 0.0 1</specular>
      </material>
    </visual>
  </link>
</model>
```

### 2. Lighting
Proper lighting is essential for camera sensors and visual realism:

```xml
<include>
  <uri>model://sun</uri>
</include>

<!-- Or create custom lighting -->
<light name="custom_light" type="directional">
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>-0.3 0.3 -1</direction>
</light>
```

## Designing Humanoid-Friendly Environments

### 1. Indoor Environment Example
Let's create an indoor environment suitable for humanoid robot testing:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_indoor_world">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.000001</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Indoor walls -->
    <model name="room_walls">
      <static>true</static>

      <!-- Wall 1: Front -->
      <link name="front_wall">
        <pose>0 -5 2.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Wall 2: Back -->
      <link name="back_wall">
        <pose>0 5 2.5 0 0 3.14159</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Wall 3: Left -->
      <link name="left_wall">
        <pose>-5 0 2.5 0 0 1.5708</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Wall 4: Right -->
      <link name="right_wall">
        <pose>5 0 2.5 0 0 -1.5708</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture for humanoid testing -->
    <model name="test_table">
      <pose>2 0 0.4 0 0 0</pose>
      <link name="table_base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacle for navigation testing -->
    <model name="navigation_obstacle">
      <pose>-2 2 0.5 0 0 0</pose>
      <link name="obstacle">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.9 0.1 0.1 1</ambient>
            <diffuse>0.9 0.1 0.1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

### 2. Outdoor Environment Example
For outdoor humanoid testing:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_outdoor_world">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Ground with terrain -->
    <model name="terrain">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <heightmap>
              <uri>model://terrain/images/heightmap.png</uri>
              <size>100 100 20</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <heightmap>
              <uri>model://terrain/images/heightmap.png</uri>
              <size>100 100 20</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Dirt</name>
            </script>
          </material>
        </visual>
      </link>
    </model>

    <!-- Sun with realistic positioning -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.3 -1</direction>
    </light>

    <!-- Trees and obstacles -->
    <include>
      <uri>model://tree</uri>
      <pose>5 5 0 0 0 0</pose>
    </include>

    <include>
      <uri>model://tree</uri>
      <pose>-5 -5 0 0 0 1.57</pose>
    </include>

    <!-- Path for humanoid navigation -->
    <model name="walking_path">
      <static>true</static>
      <link name="path_segment_1">
        <pose>0 0 0.01 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>8 1 0.02</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>8 1 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

## Physics Properties for Humanoid Robots

### 1. Contact Parameters
For realistic humanoid interaction with the environment:

```xml
<world name="humanoid_world">
  <physics type="ode">
    <ode>
      <constraints>
        <!-- For stable foot contacts -->
        <contact_surface_layer>0.002</contact_surface_layer>
        <contact_max_correcting_vel>10</contact_max_correcting_vel>
      </constraints>
      <solver>
        <!-- For stable multi-body dynamics -->
        <type>quick</type>
        <iters>50</iters>
        <sor>1.3</sor>
      </solver>
    </ode>
  </physics>
</world>
```

### 2. Material Properties
Define appropriate friction for different surfaces:

```xml
<model name="different_surfaces">
  <!-- High friction surface (good for walking) -->
  <link name="high_friction_surface">
    <collision name="collision">
      <geometry>
        <box>
          <size>2 2 0.1</size>
        </box>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>1.0</mu>
            <mu2>1.0</mu2>
          </ode>
        </friction>
      </surface>
    </collision>
  </link>

  <!-- Low friction surface (challenging for humanoid) -->
  <link name="low_friction_surface">
    <collision name="collision">
      <geometry>
        <box>
          <size>2 2 0.1</size>
        </box>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>0.1</mu>
            <mu2>0.1</mu2>
          </ode>
        </friction>
      </surface>
    </collision>
  </link>
</model>
```

## Creating Custom Models for Environments

### 1. Stairs for Humanoid Training
```xml
<model name="training_stairs">
  <static>true</static>
  <link name="step_1">
    <pose>0 0 0.1 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
      </material>
    </visual>
  </link>

  <link name="step_2">
    <pose>0 0 0.3 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
      </material>
    </visual>
  </link>

  <!-- Add more steps as needed -->
</model>
```

### 2. Balance Beam
```xml
<model name="balance_beam">
  <static>true</static>
  <link name="beam">
    <pose>0 0 0.1 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>4 0.1 0.1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>4 0.1 0.1</size>
        </box>
      </geometry>
      <material>
        <ambient>1 0.5 0 1</ambient>
        <diffuse>1 0.5 0 1</diffuse>
      </material>
    </visual>
  </link>
</model>
```

## Environment Organization Best Practices

### 1. Modular Design
Organize your environment into reusable components:

```
models/
├── humanoid_test_world/
│   ├── model.sdf
│   ├── materials/
│   │   └── scripts/
│   └── meshes/
├── indoor_room/
│   ├── model.sdf
│   └── materials/
└── outdoor_park/
    ├── model.sdf
    └── materials/
```

### 2. Performance Optimization
- Use static models for non-moving elements
- Optimize mesh complexity
- Use appropriate collision geometries (simpler than visual when possible)
- Limit the number of active physics objects

### 3. Realistic Scaling
- Ensure proper scaling for humanoid proportions
- Use realistic dimensions for furniture and obstacles
- Consider human-scale measurements for doorways, stairs, etc.

## Testing Your Environments

Create a simple launch file to test your environments:

```xml
<!-- launch/humanoid_environment.launch -->
<launch>
  <arg name="world" default="humanoid_test_world"/>

  <node name="gazebo" pkg="gazebo_ros" exec="gazebo" args="-v 4 -s libgazebo_ros_factory.so worlds/$(var world).world"/>

  <!-- Spawn your humanoid robot -->
  <node name="spawn_robot" pkg="gazebo_ros" exec="spawn_entity.py"
        args="-entity humanoid_robot -topic robot_description -x 0 -y 0 -z 1.0"/>
</launch>
```

## Next Steps

Now that you understand how to design effective environments in Gazebo, let's explore physics modeling and how to ensure your simulations accurately represent real-world dynamics. In the next section, we'll dive into configuring physics properties and understanding the parameters that affect humanoid robot simulation fidelity.