---
sidebar_position: 6
---

# Nav2: Bipedal Path Planning for Humanoid Robots

## Overview

Navigation 2 (Nav2) is the state-of-the-art navigation framework for ROS 2, but it requires significant adaptation for humanoid robots with bipedal locomotion. This section covers configuring Nav2 specifically for humanoid robots, including footstep planning, balance-aware path planning, and multi-floor navigation that accounts for the unique challenges of walking robots.

Unlike wheeled robots, humanoid robots must consider balance, foot placement, and dynamic stability during navigation, making traditional path planning approaches insufficient for safe and stable locomotion.

## Nav2 Architecture for Humanoids

### 1. System Architecture Overview

The Nav2 system for humanoid robots includes specialized components:

```
┌─────────────────────────────────────────────────────────────┐
│                 HUMANOID NAV2 ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   PLANNING  │    │  FOOTSTEP   │    │  CONTROLLER │     │
│  │             │───▶│             │───▶│             │     │
│  │ • Global    │    │ • Footstep  │    │ • Balance   │     │
│  │   Planner   │    │   Planning  │    │   Control   │     │
│  │ • Local     │    │ • Stability │    │ • Walking   │     │
│  │   Planner   │    │   Checks    │    │   Control   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 2. Humanoid-Specific Challenges

Humanoid navigation presents unique challenges:

- **Balance Constraints**: Paths must maintain robot stability
- **Footstep Planning**: Each step must be carefully planned
- **Dynamic Locomotion**: Walking motion affects localization
- **Stair Navigation**: Multi-floor capability required
- **Terrain Adaptation**: Different surfaces require different gaits

## Nav2 Installation and Basic Setup

### 1. Installing Nav2 for Humanoid Applications

Setting up Nav2 with humanoid-specific packages:

```bash
# Install Nav2 base packages
sudo apt update
sudo apt install -y ros-humble-navigation2
sudo apt install -y ros-humble-nav2-bringup
sudo apt install -y ros-humble-nav2-gui

# Install additional packages for humanoid navigation
sudo apt install -y ros-humble-nav2-rviz-plugins
sudo apt install -y ros-humble-robot-localization
sudo apt install -y ros-humble-interactive-markers

# Install humanoid-specific packages (if available)
# For custom implementations, we'll create our own packages
```

### 2. Basic Nav2 Configuration

Creating a basic Nav2 configuration for humanoid robots:

```yaml
# config/humanoid_nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.2
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    # Humanoid-specific behavior tree
    # We'll define this in the next section
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    # Humanoid-specific controllers
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid bipedal controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 50
      model_dt: 0.05
      batch_size: 1000
      vx_std: 0.2
      vy_std: 0.0
      wz_std: 0.3
      vx_max: 0.5
      vx_min: -0.1
      vy_max: 1.0
      wz_max: 1.5
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      state_reset_tolerance: 0.5
      control_horizon: 10
      trajectory_visualization_enabled: true
      # Humanoid-specific parameters
      balance_weight: 10.0
      step_size: 0.3  # Typical humanoid step size
      max_step_height: 0.15  # Maximum step height (15cm)

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      state_tolerance: 0.05

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: "odom"
      robot_base_frame: "base_link"
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.4  # Humanoid robot radius
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 8
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: "/scan"
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      always_send_full_costmap: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: "map"
      robot_base_frame: "base_link"
      use_sim_time: True
      robot_radius: 0.4
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 8
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: "/scan"
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      # For humanoid robots, we'll use a custom planner
      # that considers balance and footstep constraints
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

## Humanoid-Specific Path Planning

### 1. Footstep Planning Integration

Creating a custom footstep planner for humanoid navigation:

```python
#!/usr/bin/env python3
# footstep_planner.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from std_msgs.msg import Header
import numpy as np
import math

class FootstepPlanner(Node):
    """
    Footstep planner for humanoid robot navigation
    """
    def __init__(self):
        super().__init__('footstep_planner')

        # Create subscribers
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            10
        )

        # Create publishers
        self.footstep_path_pub = self.create_publisher(
            Path,
            '/footstep_plan',
            10
        )

        self.footstep_viz_pub = self.create_publisher(
            MarkerArray,
            '/footsteps_visualization',
            10
        )

        # Initialize variables
        self.costmap = None
        self.resolution = 0.05
        self.origin = None
        self.width = 0
        self.height = 0

        # Footstep parameters
        self.step_size = 0.3  # 30cm step
        self.step_width = 0.2  # 20cm foot width
        self.max_step_height = 0.15  # 15cm max step height
        self.balance_margin = 0.1  # 10cm balance margin

        # Visualization markers
        self.footstep_markers = MarkerArray()

        self.get_logger().info('Footstep Planner initialized')

    def costmap_callback(self, msg):
        """
        Store costmap information
        """
        self.costmap = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.resolution = msg.info.resolution
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.width = msg.info.width
        self.height = msg.info.height

    def path_callback(self, msg):
        """
        Process incoming path and generate footstep plan
        """
        if self.costmap is None:
            self.get_logger().warn('Costmap not received yet')
            return

        # Convert path to footstep plan
        footstep_path = self.generate_footsteps(msg)

        # Publish footstep path
        self.footstep_path_pub.publish(footstep_path)

        # Publish visualization
        self.publish_footstep_visualization(footstep_path)

    def generate_footsteps(self, global_path):
        """
        Generate footstep plan from global path
        """
        footstep_path = Path()
        footstep_path.header = global_path.header

        if len(global_path.poses) < 2:
            return footstep_path

        # Start with left foot
        current_foot = "left"
        current_pos = np.array([global_path.poses[0].pose.position.x,
                               global_path.poses[0].pose.position.y])

        footsteps = [current_pos.copy()]

        # Process path in segments
        for i in range(1, len(global_path.poses)):
            target_pos = np.array([global_path.poses[i].pose.position.x,
                                  global_path.poses[i].pose.position.y])

            # Calculate direction and distance
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)

            if distance > self.step_size:
                # Calculate number of steps needed
                num_steps = int(distance / self.step_size)
                step_vector = direction / distance * self.step_size

                # Generate intermediate footsteps
                for j in range(num_steps):
                    next_pos = current_pos + step_vector
                    if self.is_valid_footstep(next_pos):
                        footsteps.append(next_pos.copy())
                        current_pos = next_pos
                        # Alternate feet
                        current_foot = "right" if current_foot == "left" else "left"
                    else:
                        # Find alternative path if step is invalid
                        alternative_pos = self.find_alternative_footstep(current_pos, target_pos)
                        if alternative_pos is not None:
                            footsteps.append(alternative_pos.copy())
                            current_pos = alternative_pos
                            current_foot = "right" if current_foot == "left" else "left"
                        else:
                            self.get_logger().warn(f'Could not find valid footstep near {next_pos}')
                            break

        # Convert to Path message
        for i, pos in enumerate(footsteps):
            pose_stamped = PoseStamped()
            pose_stamped.header = global_path.header
            pose_stamped.pose.position.x = float(pos[0])
            pose_stamped.pose.position.y = float(pos[1])
            pose_stamped.pose.position.z = 0.0

            # Add orientation (facing direction of next step)
            if i < len(footsteps) - 1:
                next_pos = footsteps[i + 1]
                yaw = math.atan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
                # Convert to quaternion
                pose_stamped.pose.orientation.z = math.sin(yaw / 2)
                pose_stamped.pose.orientation.w = math.cos(yaw / 2)

            footstep_path.poses.append(pose_stamped)

        return footstep_path

    def is_valid_footstep(self, position):
        """
        Check if a footstep is valid (not on obstacles, within bounds)
        """
        if self.costmap is None:
            return False

        # Convert world coordinates to costmap indices
        map_x = int((position[0] - self.origin[0]) / self.resolution)
        map_y = int((position[1] - self.origin[1]) / self.resolution)

        # Check bounds
        if map_x < 0 or map_x >= self.width or map_y < 0 or map_y >= self.height:
            return False

        # Check cost (should be free space)
        cost = self.costmap[map_y, map_x]
        if cost >= 50:  # Threshold for obstacle
            return False

        # Check surrounding area for balance (3x3 area around foot)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                check_x, check_y = map_x + dx, map_y + dy
                if (0 <= check_x < self.width and 0 <= check_y < self.height):
                    if self.costmap[check_y, check_x] >= 75:  # Higher threshold for immediate area
                        return False

        return True

    def find_alternative_footstep(self, current_pos, target_pos):
        """
        Find alternative footstep if primary location is invalid
        """
        # Try circular search around current position
        search_radius = self.step_size * 1.5
        step_angle = 0.2  # 0.2 radian steps (~11 degrees)

        for radius in np.arange(0.1, search_radius, 0.1):
            for angle in np.arange(0, 2 * math.pi, step_angle):
                alt_x = current_pos[0] + radius * math.cos(angle)
                alt_y = current_pos[1] + radius * math.sin(angle)
                alt_pos = np.array([alt_x, alt_y])

                # Check if this alternative is closer to target and valid
                if (np.linalg.norm(alt_pos - target_pos) < np.linalg.norm(current_pos - target_pos) and
                    self.is_valid_footstep(alt_pos)):
                    return alt_pos

        return None

    def publish_footstep_visualization(self, footstep_path):
        """
        Publish visualization markers for footsteps
        """
        marker_array = MarkerArray()

        for i, pose_stamped in enumerate(footstep_path.poses):
            # Create foot marker
            foot_marker = Marker()
            foot_marker.header = footstep_path.header
            foot_marker.ns = "footsteps"
            foot_marker.id = i
            foot_marker.type = Marker.CUBE
            foot_marker.action = Marker.ADD

            foot_marker.pose = pose_stamped.pose
            foot_marker.pose.position.z = 0.02  # Slightly above ground

            # Different colors for left/right feet
            if i % 2 == 0:  # Left foot
                foot_marker.color.r = 1.0
                foot_marker.color.g = 0.0
                foot_marker.color.b = 0.0
            else:  # Right foot
                foot_marker.color.r = 0.0
                foot_marker.color.g = 0.0
                foot_marker.color.b = 1.0
            foot_marker.color.a = 0.8

            foot_marker.scale.x = self.step_size * 0.8  # Foot length
            foot_marker.scale.y = self.step_width  # Foot width
            foot_marker.scale.z = 0.01  # Thickness

            marker_array.markers.append(foot_marker)

        self.footstep_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    footstep_planner = FootstepPlanner()

    try:
        rclpy.spin(footstep_planner)
    except KeyboardInterrupt:
        footstep_planner.get_logger().info('Shutting down footstep planner')
    finally:
        footstep_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Balance-Aware Path Planning

Implementing balance-aware navigation planning:

```python
#!/usr/bin/env python3
# balance_aware_planner.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, PolygonStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
import numpy as np
import math
from scipy.spatial import distance

class BalanceAwarePlanner(Node):
    """
    Balance-aware path planner for humanoid robots
    """
    def __init__(self):
        super().__init__('balance_aware_planner')

        # Create subscribers
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Float64,
            '/walking_state',  # From walking motion compensation
            self.walking_state_callback,
            10
        )

        # Create publishers
        self.balance_path_pub = self.create_publisher(
            Path,
            '/balance_aware_plan',
            10
        )

        self.support_polygon_pub = self.create_publisher(
            PolygonStamped,
            '/support_polygon',
            10
        )

        self.balance_viz_pub = self.create_publisher(
            MarkerArray,
            '/balance_visualization',
            10
        )

        # Initialize variables
        self.costmap = None
        self.laser_data = None
        self.walking_state = 0.0  # 0.0 = standing, 1.0 = walking
        self.current_support_polygon = []
        self.balance_margin = 0.15  # 15cm balance margin

        # Robot dimensions (for support polygon)
        self.foot_length = 0.25
        self.foot_width = 0.15
        self.foot_separation = 0.2  # Distance between feet

        self.get_logger().info('Balance-aware Planner initialized')

    def costmap_callback(self, msg):
        """
        Store costmap information
        """
        self.costmap = msg

    def laser_callback(self, msg):
        """
        Store laser scan data for obstacle detection
        """
        self.laser_data = msg

    def walking_state_callback(self, msg):
        """
        Update walking state for balance planning
        """
        self.walking_state = msg.data

    def path_callback(self, msg):
        """
        Process incoming path and apply balance constraints
        """
        if self.costmap is None:
            self.get_logger().warn('Costmap not received yet')
            return

        # Apply balance constraints to path
        balance_path = self.apply_balance_constraints(msg)

        # Publish balance-aware path
        self.balance_path_pub.publish(balance_path)

        # Publish support polygon
        self.publish_support_polygon()

        # Publish visualization
        self.publish_balance_visualization(balance_path)

    def apply_balance_constraints(self, original_path):
        """
        Apply balance constraints to the original path
        """
        if len(original_path.poses) < 2:
            return original_path

        balance_path = Path()
        balance_path.header = original_path.header

        # Calculate support polygon based on current stance
        support_polygon = self.calculate_support_polygon()

        for i, pose_stamped in enumerate(original_path.poses):
            # Get the original position
            original_pos = np.array([
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y
            ])

            # Check if position is within balance margin of support polygon
            balanced_pos = self.ensure_balance(original_pos, support_polygon)

            # Create new pose with balanced position
            new_pose = PoseStamped()
            new_pose.header = pose_path.header
            new_pose.pose.position.x = balanced_pos[0]
            new_pose.pose.position.y = balanced_pos[1]
            new_pose.pose.position.z = pose_stamped.pose.position.z

            # Copy orientation
            new_pose.pose.orientation = pose_stamped.pose.orientation

            balance_path.poses.append(new_pose)

            # Update support polygon for next step
            if i > 0:  # After first step, update support polygon
                self.update_support_polygon(balanced_pos)

        return balance_path

    def calculate_support_polygon(self):
        """
        Calculate current support polygon based on foot positions
        """
        # For standing: rectangular area between feet
        # For walking: area around single supporting foot
        if self.walking_state < 0.5:  # Standing
            # Create rectangular support polygon between feet
            center_x = 0  # Robot center
            center_y = 0
            half_length = self.foot_length / 2 + 0.05  # Add small margin
            half_width = self.foot_separation / 2 + self.foot_width / 2

            polygon = [
                (center_x - half_length, center_y - half_width),
                (center_x + half_length, center_y - half_width),
                (center_x + half_length, center_y + half_width),
                (center_x - half_length, center_y + half_width)
            ]
        else:  # Walking - single foot support
            # Simplified: just around the supporting foot
            support_x = 0
            support_y = 0  # This would come from actual foot position
            half_length = self.foot_length / 2
            half_width = self.foot_width / 2

            polygon = [
                (support_x - half_length, support_y - half_width),
                (support_x + half_length, support_y - half_width),
                (support_x + half_length, support_y + half_width),
                (support_x - half_length, support_y + half_width)
            ]

        return polygon

    def ensure_balance(self, target_pos, support_polygon):
        """
        Ensure target position is within balance constraints
        """
        # Calculate centroid of support polygon
        centroid_x = sum(p[0] for p in support_polygon) / len(support_polygon)
        centroid_y = sum(p[1] for p in support_polygon) / len(support_polygon)
        centroid = np.array([centroid_x, centroid_y])

        # Check if target is within balance margin
        dist_to_centroid = np.linalg.norm(target_pos - centroid)

        # Calculate max allowable distance based on support polygon
        max_balance_dist = min(
            self.foot_length, self.foot_width
        ) / 2 + self.balance_margin

        if dist_to_centroid > max_balance_dist:
            # Project target position onto balance boundary
            direction = (target_pos - centroid) / dist_to_centroid
            balanced_pos = centroid + direction * max_balance_dist
            self.get_logger().warn(f'Adjusting path for balance: {target_pos} -> {balanced_pos}')
            return balanced_pos
        else:
            return target_pos

    def update_support_polygon(self, foot_pos):
        """
        Update support polygon based on new foot position
        """
        # This would update the support polygon as the robot moves
        # For now, just store the position
        pass

    def publish_support_polygon(self):
        """
        Publish current support polygon
        """
        polygon_msg = PolygonStamped()
        polygon_msg.header.stamp = self.get_clock().now().to_msg()
        polygon_msg.header.frame_id = "base_link"

        # Create polygon points (simplified)
        support_polygon = self.calculate_support_polygon()
        for x, y in support_polygon:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            polygon_msg.polygon.points.append(point)

        self.support_polygon_pub.publish(polygon_msg)

    def publish_balance_visualization(self, path):
        """
        Publish balance-related visualization markers
        """
        marker_array = MarkerArray()

        # Support polygon visualization
        polygon_marker = Marker()
        polygon_marker.header.stamp = self.get_clock().now().to_msg()
        polygon_marker.header.frame_id = "base_link"
        polygon_marker.ns = "support_polygon"
        polygon_marker.id = 0
        polygon_marker.type = Marker.LINE_STRIP
        polygon_marker.action = Marker.ADD

        # Add points to form the polygon
        support_polygon = self.calculate_support_polygon()
        for x, y in support_polygon:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.05  # Slightly above ground
            polygon_marker.points.append(point)

        # Close the polygon
        if support_polygon:
            point = Point()
            point.x = support_polygon[0][0]
            point.y = support_polygon[0][1]
            point.z = 0.05
            polygon_marker.points.append(point)

        polygon_marker.color.r = 0.0
        polygon_marker.color.g = 1.0
        polygon_marker.color.b = 0.0
        polygon_marker.color.a = 0.8
        polygon_marker.scale.x = 0.02

        marker_array.markers.append(polygon_marker)

        # Path visualization
        path_marker = Marker()
        path_marker.header = path.header
        path_marker.ns = "balance_path"
        path_marker.id = 1
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD

        for pose_stamped in path.poses:
            point = Point()
            point.x = pose_stamped.pose.position.x
            point.y = pose_stamped.pose.position.y
            point.z = 0.1  # Above polygon
            path_marker.points.append(point)

        path_marker.color.r = 1.0
        path_marker.color.g = 0.0
        path_marker.color.b = 1.0
        path_marker.color.a = 0.8
        path_marker.scale.x = 0.05

        marker_array.markers.append(path_marker)

        self.balance_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    balance_planner = BalanceAwarePlanner()

    try:
        rclpy.spin(balance_planner)
    except KeyboardInterrupt:
        balance_planner.get_logger().info('Shutting down balance-aware planner')
    finally:
        balance_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Stair and Multi-Floor Navigation

### 1. Stair Navigation Planning

Implementing navigation for stairs and multi-floor environments:

```python
#!/usr/bin/env python3
# stair_navigation.py

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8
import numpy as np
import math

class StairNavigation(Node):
    """
    Stair navigation system for humanoid robots
    """
    def __init__(self):
        super().__init__('stair_navigation')

        # Create subscribers
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Create publishers
        self.stair_path_pub = self.create_publisher(
            Path,
            '/stair_navigation_plan',
            10
        )

        self.stair_type_pub = self.create_publisher(
            Int8,
            '/stair_type',
            10
        )

        self.stair_viz_pub = self.create_publisher(
            MarkerArray,
            '/stair_visualization',
            10
        )

        # Initialize variables
        self.laser_data = None
        self.floor_height = 0.0  # Current floor height
        self.step_height = 0.15  # 15cm per step (typical)
        self.step_depth = 0.30   # 30cm depth (typical)

        # Stair detection parameters
        self.stair_threshold = 0.1  # Height difference threshold
        self.min_stair_steps = 3    # Minimum steps to qualify as stairs

        self.get_logger().info('Stair Navigation initialized')

    def laser_callback(self, msg):
        """
        Process laser scan data for stair detection
        """
        self.laser_data = msg
        self.detect_stairs()

    def path_callback(self, msg):
        """
        Process incoming path and adapt for stairs
        """
        if len(msg.poses) < 2:
            self.stair_path_pub.publish(msg)
            return

        # Check if path involves stairs
        stair_path = self.adapt_path_for_stairs(msg)

        # Publish stair-adapted path
        self.stair_path_pub.publish(stair_path)

        # Publish stair visualization
        self.publish_stair_visualization(stair_path)

    def detect_stairs(self):
        """
        Detect stairs in laser scan data
        """
        if self.laser_data is None:
            return

        # Analyze laser data for step patterns
        ranges = np.array(self.laser_data.ranges)
        angles = np.linspace(
            self.laser_data.angle_min,
            self.laser_data.angle_max,
            len(ranges)
        )

        # Convert to Cartesian coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)

        # Look for step-like patterns in the data
        # This is a simplified approach - real implementation would be more sophisticated
        height_changes = np.diff(y_coords)
        potential_steps = np.where(np.abs(height_changes) > self.stair_threshold)[0]

        if len(potential_steps) >= self.min_stair_steps:
            # Likely stairs detected
            stair_msg = Int8()
            stair_msg.data = 1  # Stairs detected
            self.stair_type_pub.publish(stair_msg)
            self.get_logger().info('Stairs detected in environment')
        else:
            # No stairs
            stair_msg = Int8()
            stair_msg.data = 0  # No stairs
            self.stair_type_pub.publish(stair_msg)

    def adapt_path_for_stairs(self, original_path):
        """
        Adapt path for stair navigation
        """
        if len(original_path.poses) < 2:
            return original_path

        stair_path = Path()
        stair_path.header = original_path.header

        # For stair navigation, we need to plan each step explicitly
        # This is a simplified approach - real implementation would be more detailed
        current_height = self.floor_height

        for i in range(len(original_path.poses)):
            original_pose = original_path.poses[i]

            # Create stair-adapted pose
            stair_pose = PoseStamped()
            stair_pose.header = original_path.header

            # Copy position (for now, in real implementation, adjust for steps)
            stair_pose.pose.position.x = original_pose.pose.position.x
            stair_pose.pose.position.y = original_pose.pose.position.y

            # For stairs, we need to account for step height
            if self.is_approaching_stairs(original_pose):
                # Calculate appropriate Z height for this step
                step_number = self.calculate_step_number(original_pose)
                stair_pose.pose.position.z = self.floor_height + (step_number * self.step_height)
            else:
                stair_pose.pose.position.z = self.floor_height

            # Copy orientation
            stair_pose.pose.orientation = original_pose.pose.orientation

            stair_path.poses.append(stair_pose)

        return stair_path

    def is_approaching_stairs(self, pose):
        """
        Check if the robot is approaching stairs
        """
        # This would use more sophisticated detection in real implementation
        # For now, return False to avoid complications
        return False

    def calculate_step_number(self, pose):
        """
        Calculate which step the robot should be on
        """
        # This would be calculated based on the path and stair location
        # For now, return a simple increment
        return 0

    def publish_stair_visualization(self, path):
        """
        Publish stair-related visualization
        """
        marker_array = MarkerArray()

        # Visualize the path adapted for stairs
        path_marker = Marker()
        path_marker.header = path.header
        path_marker.ns = "stair_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD

        for pose_stamped in path.poses:
            point = Point()
            point.x = pose_stamped.pose.position.x
            point.y = pose_stamped.pose.position.y
            point.z = pose_stamped.pose.position.z
            path_marker.points.append(point)

        path_marker.color.r = 0.0
        path_marker.color.g = 1.0
        path_marker.color.b = 1.0
        path_marker.color.a = 0.8
        path_marker.scale.x = 0.05

        marker_array.markers.append(path_marker)

        # If stairs are detected, visualize them
        if self.laser_data is not None:
            # Create stair visualization markers
            stair_marker = Marker()
            stair_marker.header = path.header
            stair_marker.ns = "detected_stairs"
            stair_marker.id = 1
            stair_marker.type = Marker.CUBE_LIST
            stair_marker.action = Marker.ADD

            # Add points for stair visualization
            # This is a simplified visualization
            for step in range(5):  # Visualize 5 steps
                point = Point()
                point.x = 2.0  # Position of stairs
                point.y = step * self.step_depth
                point.z = step * self.step_height + self.step_height / 2
                stair_marker.points.append(point)

            stair_marker.color.r = 1.0
            stair_marker.color.g = 0.5
            stair_marker.color.b = 0.0
            stair_marker.color.a = 0.8
            stair_marker.scale.x = self.step_depth
            stair_marker.scale.y = 0.2
            stair_marker.scale.z = self.step_height

            marker_array.markers.append(stair_marker)

        self.stair_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    stair_nav = StairNavigation()

    try:
        rclpy.spin(stair_nav)
    except KeyboardInterrupt:
        stair_nav.get_logger().info('Shutting down stair navigation')
    finally:
        stair_nav.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Humanoid Navigation Controller

### 1. Walking Controller Integration

Creating a humanoid-specific navigation controller:

```python
#!/usr/bin/env python3
# humanoid_navigation_controller.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64, Bool
import numpy as np
import math
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs

class HumanoidNavigationController(Node):
    """
    Navigation controller specifically designed for humanoid robots
    """
    def __init__(self):
        super().__init__('humanoid_navigation_controller')

        # Create subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.path_sub = self.create_subscription(
            Path,
            '/footstep_plan',  # From footstep planner
            self.path_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Create publishers
        self.velocity_cmd_pub = self.create_publisher(
            Twist,
            '/humanoid_velocity_controller/cmd_vel',
            10
        )

        self.balance_state_pub = self.create_publisher(
            Bool,
            '/balance_state',
            10
        )

        self.step_command_pub = self.create_publisher(
            Float64,
            '/step_command',
            10
        )

        # Initialize TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.current_velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, wz
        self.imu_data = None
        self.joint_states = None
        self.current_path = []
        self.path_index = 0

        # Walking parameters
        self.step_size = 0.3  # 30cm step
        self.step_height = 0.05  # 5cm step height
        self.walking_speed = 0.3  # 0.3 m/s walking speed
        self.balance_threshold = 0.1  # Balance threshold

        # Control parameters
        self.linear_kp = 1.0
        self.angular_kp = 2.0
        self.max_linear_vel = 0.4
        self.max_angular_vel = 0.8

        # Walking state machine
        self.walking_state = "standing"  # standing, stepping, walking
        self.step_phase = 0.0  # 0.0 to 1.0

        self.get_logger().info('Humanoid Navigation Controller initialized')

    def cmd_vel_callback(self, msg):
        """
        Handle velocity commands
        """
        # In humanoid navigation, we might need to convert this
        # to step-based commands rather than direct velocity
        self.execute_navigation_command(msg)

    def path_callback(self, msg):
        """
        Handle path following commands
        """
        self.current_path = msg.poses
        self.path_index = 0
        self.get_logger().info(f'New path received with {len(self.current_path)} waypoints')

    def odom_callback(self, msg):
        """
        Update robot pose from odometry
        """
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y

        # Convert quaternion to euler
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_pose[2] = math.atan2(siny_cosp, cosy_cosp)

        # Update velocity
        self.current_velocity[0] = msg.twist.twist.linear.x
        self.current_velocity[1] = msg.twist.twist.linear.y
        self.current_velocity[2] = msg.twist.twist.angular.z

    def imu_callback(self, msg):
        """
        Store IMU data for balance control
        """
        self.imu_data = msg

    def joint_state_callback(self, msg):
        """
        Store joint state data
        """
        self.joint_states = msg

    def execute_navigation_command(self, cmd_vel):
        """
        Execute navigation command with humanoid-specific control
        """
        # Check balance before executing command
        if not self.is_balanced():
            self.get_logger().warn('Robot is not balanced, stopping navigation')
            stop_cmd = Twist()
            self.velocity_cmd_pub.publish(stop_cmd)
            return

        # For humanoid robots, we might need to convert linear/angular
        # commands to step-based commands
        humanoid_cmd = self.convert_to_humanoid_command(cmd_vel)

        # Publish the command
        self.velocity_cmd_pub.publish(humanoid_cmd)

        # Update walking state
        if abs(cmd_vel.linear.x) > 0.01 or abs(cmd_vel.angular.z) > 0.01:
            self.walking_state = "walking"
        else:
            self.walking_state = "standing"

    def convert_to_humanoid_command(self, cmd_vel):
        """
        Convert standard velocity command to humanoid-appropriate command
        """
        humanoid_cmd = Twist()

        # Apply humanoid-specific constraints
        linear_x = max(-self.max_linear_vel, min(self.max_linear_vel, cmd_vel.linear.x))
        angular_z = max(-self.max_angular_vel, min(self.max_angular_vel, cmd_vel.angular.z))

        # For humanoid, we might want to convert to step commands
        # For now, just apply constraints
        humanoid_cmd.linear.x = linear_x
        humanoid_cmd.angular.z = angular_z

        # Keep other components zero
        humanoid_cmd.linear.y = cmd_vel.linear.y
        humanoid_cmd.linear.z = cmd_vel.linear.z
        humanoid_cmd.angular.x = cmd_vel.angular.x
        humanoid_cmd.angular.y = cmd_vel.angular.y

        return humanoid_cmd

    def is_balanced(self):
        """
        Check if robot is in balanced state using IMU data
        """
        if self.imu_data is None:
            return True  # Assume balanced if no data

        # Check if orientation is within balance limits
        quat = self.imu_data.orientation
        # Convert to roll/pitch to check balance
        sinr_cosp = 2 * (quat.w * quat.x + quat.y * quat.z)
        cosr_cosp = 1 - 2 * (quat.x * quat.x + quat.y * quat.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (quat.w * quat.y - quat.z * quat.x)
        pitch = math.asin(sinp)

        # Check if roll and pitch are within balance thresholds
        if abs(roll) > self.balance_threshold or abs(pitch) > self.balance_threshold:
            return False

        return True

    def follow_path(self):
        """
        Follow the current path using humanoid-specific control
        """
        if not self.current_path or self.path_index >= len(self.current_path):
            return

        # Get the next waypoint
        target_pose = self.current_path[self.path_index]
        target_x = target_pose.pose.position.x
        target_y = target_pose.pose.position.y

        # Calculate distance to target
        dx = target_x - self.current_pose[0]
        dy = target_y - self.current_pose[1]
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate target angle
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.current_pose[2]

        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Create command to follow path
        cmd_vel = Twist()
        cmd_vel.linear.x = min(self.max_linear_vel, self.linear_kp * distance)
        cmd_vel.angular.z = min(self.max_angular_vel, self.angular_kp * angle_diff)

        # Check if we've reached this waypoint
        if distance < 0.2:  # Waypoint tolerance
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                self.get_logger().info('Path completed')
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0

        # Execute the command
        self.execute_navigation_command(cmd_vel)

    def step_callback(self):
        """
        Timer callback for step-based control
        """
        if self.walking_state == "walking" and self.current_path:
            self.follow_path()

        # Publish balance state
        balance_msg = Bool()
        balance_msg.data = self.is_balanced()
        self.balance_state_pub.publish(balance_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidNavigationController()

    # Create timer for path following
    controller.create_timer(0.1, controller.step_callback)  # 10Hz

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down humanoid navigation controller')
    finally:
        # Stop the robot on shutdown
        stop_cmd = Twist()
        controller.velocity_cmd_pub.publish(stop_cmd)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Complete Navigation Launch Files

### 1. Humanoid Navigation Launch

Creating launch files for the complete humanoid navigation system:

```python
# launch/humanoid_navigation.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import ReplaceString

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')
    container_name_full = (container_name, '_container')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    declare_autostart = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack'
    )

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_nav2'),
            'config',
            'humanoid_nav2_params.yaml'
        ]),
        description='Full path to the ROS2 parameters file to use for all launched nodes'
    )

    declare_use_composition = DeclareLaunchArgument(
        'use_composition',
        default_value='False',
        description='Use composed bringup if True'
    )

    declare_container_name = DeclareLaunchArgument(
        'container_name',
        default_value='nav2_container',
        description='the name of conatiner that nodes will load in if use composition'
    )

    # Robot description (for transforms)
    robot_description = {'robot_description':
        PathJoinSubstitution([
            FindPackageShare('humanoid_description'),
            'urdf',
            'humanoid.urdf'
        ])
    }

    # Static transform publisher for robot base
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
    )

    # Navigation lifecycle manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': ['map_server',
                                   'planner_server',
                                   'controller_server',
                                   'behavior_server',
                                   'bt_navigator',
                                   'waypoint_follower',
                                   'velocity_smoother']}]
    )

    # Footstep planner node
    footstep_planner = Node(
        package='humanoid_nav2',
        executable='footstep_planner',
        name='footstep_planner',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/plan', '/global_costmap/costmap_updates'),
            ('/footstep_plan', '/footstep_plan'),
        ]
    )

    # Balance-aware planner node
    balance_planner = Node(
        package='humanoid_nav2',
        executable='balance_aware_planner',
        name='balance_aware_planner',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/plan', '/plan'),
            ('/balance_aware_plan', '/balance_aware_plan'),
        ]
    )

    # Stair navigation node
    stair_navigation = Node(
        package='humanoid_nav2',
        executable='stair_navigation',
        name='stair_navigation',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/scan', '/scan'),
            ('/stair_navigation_plan', '/stair_navigation_plan'),
        ]
    )

    # Humanoid navigation controller
    humanoid_controller = Node(
        package='humanoid_nav2',
        executable='humanoid_navigation_controller',
        name='humanoid_navigation_controller',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/cmd_vel', '/cmd_vel'),
            ('/footstep_plan', '/footstep_plan'),
        ]
    )

    # AMCL localization node
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('scan', 'scan')]
    )

    # Map server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('map', 'map'),
                   ('map_metadata', 'map_metadata')]
    )

    # Planner server
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('~/global_costmap/costmap', 'global_costmap/costmap'),
                   ('~/global_costmap/costmap_updates', 'global_costmap/costmap_updates')]
    )

    # Controller server
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel'),
                   ('~/local_costmap/costmap', 'local_costmap/costmap'),
                   ('~/local_costmap/costmap_updates', 'local_costmap/costmap_updates'),
                   ('~/global_costmap/costmap', 'global_costmap/costmap')]
    )

    # Behavior server
    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel'),
                   ('/local_costmap/costmap', '/local_costmap/costmap'),
                   ('/local_costmap/costmap_updates', '/local_costmap/costmap_updates')]
    )

    # BT Navigator
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('navigate_to_pose', 'navigate_to_pose'),
                   ('navigate_through_poses', 'navigate_through_poses')]
    )

    # Waypoint follower
    waypoint_follower = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel')]
    )

    # Velocity smoother
    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel'),
                   ('smoothed_cmd_vel', 'smoothed_cmd_vel')]
    )

    # Lifecycle manager should start after other nodes
    delayed_lifecycle_manager = RegisterEventHandler(
        OnProcessStart(
            target_action=velocity_smoother,
            on_start=[lifecycle_manager]
        )
    )

    # Create groups for organized launching
    navigation_group = GroupAction(
        actions=[
            SetParameter('use_sim_time', use_sim_time),
            amcl,
            map_server,
            planner_server,
            controller_server,
            behavior_server,
            bt_navigator,
            waypoint_follower,
            velocity_smoother
        ]
    )

    humanoid_group = GroupAction(
        actions=[
            SetParameter('use_sim_time', use_sim_time),
            footstep_planner,
            balance_planner,
            stair_navigation,
            humanoid_controller
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_autostart,
        declare_params_file,
        declare_use_composition,
        declare_container_name,

        static_transform_publisher,

        navigation_group,
        humanoid_group,

        delayed_lifecycle_manager
    ])
```

## Next Steps

With Nav2 properly configured for bipedal navigation, you're ready to move on to the Module 3 project. The next section will integrate all the components you've learned about - Isaac Sim, Isaac ROS perception, VSLAM, and Nav2 navigation - into a complete AI-powered humanoid navigation system.

This comprehensive system will demonstrate how to combine photorealistic simulation, hardware-accelerated perception, visual-inertial SLAM, and balance-aware navigation to create an intelligent humanoid robot capable of autonomous navigation in complex environments.