---
sidebar_position: 5
---

# Manipulation and Interaction Systems

## Overview

Manipulation and interaction systems form the physical execution layer of Vision-Language-Action (VLA) systems for humanoid robots. These systems translate cognitive plans and perceptual understanding into precise physical actions, enabling humanoid robots to manipulate objects, interact with humans, and perform complex tasks in real-world environments.

This module covers the implementation of robotic manipulation capabilities, human-robot interaction protocols, and safe interaction systems that work in conjunction with voice recognition, LLM planning, and multimodal perception to create natural and effective human-robot collaboration.

## Learning Objectives

By the end of this section, you will be able to:
- Implement robotic manipulation systems for humanoid robots
- Design safe and intuitive human-robot interaction protocols
- Create object manipulation pipelines with perception integration
- Implement grasp planning and execution for various object types
- Develop multimodal interaction systems combining voice, vision, and touch
- Design safe interaction validation and monitoring systems

## Prerequisites

Before implementing manipulation and interaction systems, ensure you have:
- Completed Modules 1-3 (ROS 2, Digital Twin, AI-Robot Brain)
- Voice recognition and LLM integration systems from previous modules
- Multimodal perception integration for object detection and scene understanding
- Basic understanding of robotic manipulation concepts (kinematics, grasping)
- Experience with ROS 2 action and service interfaces
- Familiarity with MoveIt! for motion planning

## Robotic Manipulation Architecture

### Manipulation Framework Design

Design a comprehensive manipulation framework for humanoid robots:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
import numpy as np
import tf2_ros
from typing import Dict, List, Tuple, Optional
from enum import Enum

class ManipulationState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SAFETY_STOP = "safety_stop"

class GraspType(Enum):
    PINCH = "pinch"
    POWER = "power"
    PRECISION = "precision"
    LATERAL = "lateral"
    SUCTION = "suction"  # For robots with suction grippers

class ManipulationController(Node):
    """Main controller for humanoid manipulation systems"""

    def __init__(self):
        super().__init__('manipulation_controller')

        # Initialize callback group for concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # MoveIt! action clients
        self.move_group_client = ActionClient(
            self, MoveGroup, 'move_group',
            callback_group=self.callback_group)
        self.pick_client = ActionClient(
            self, PickUp, 'pickup',
            callback_group=self.callback_group)
        self.place_client = ActionClient(
            self, Place, 'place',
            callback_group=self.callback_group)

        # Service clients
        self.get_ik_client = self.create_client(
            GetPositionIK, 'compute_ik',
            callback_group=self.callback_group)
        self.get_fk_client = self.create_client(
            GetPositionFK, 'compute_fk',
            callback_group=self.callback_group)

        # TF2 buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers and subscribers
        self.manipulation_status_pub = self.create_publisher(
            String, 'manipulation_status', 10)
        self.collision_object_pub = self.create_publisher(
            PlanningScene, 'planning_scene', 10)
        self.interaction_request_sub = self.create_subscription(
            String, 'interaction_request', self.interaction_callback, 10)

        # Internal state
        self.current_state = ManipulationState.IDLE
        self.robot_group = "humanoid_arm"  # Default group name
        self.end_effector_link = "humanoid_hand"  # Default end effector
        self.manipulation_history = []

        # Grasp planning parameters
        self.grasp_database = self._initialize_grasp_database()
        self.approach_distance = 0.1  # meters
        self.lift_distance = 0.05     # meters

        self.get_logger().info("Manipulation Controller initialized")

    def _initialize_grasp_database(self) -> Dict:
        """Initialize database of known grasps for common objects"""
        return {
            'cup': {
                'grasp_types': [GraspType.POWER, GraspType.PINCH],
                'approach_direction': [0, -1, 0],  # Approach from side
                'grasp_pose_offset': [0.05, 0, 0],  # Offset for cup handle
                'gripper_width': 0.08
            },
            'bottle': {
                'grasp_types': [GraspType.POWER],
                'approach_direction': [0, 0, -1],  # Approach from top
                'grasp_pose_offset': [0, 0, 0.1],  # Grasp at neck
                'gripper_width': 0.06
            },
            'book': {
                'grasp_types': [GraspType.PINCH, GraspType.PRECISION],
                'approach_direction': [0, -1, 0],  # Approach from spine
                'grasp_pose_offset': [0, 0, 0.02],  # Thickness offset
                'gripper_width': 0.04
            },
            'box': {
                'grasp_types': [GraspType.POWER],
                'approach_direction': [0, 0, -1],  # Approach from top
                'grasp_pose_offset': [0, 0, 0.05],  # Half height offset
                'gripper_width': 0.12
            }
        }

    def move_to_pose(self, target_pose: Pose, group_name: str = None) -> bool:
        """Move robot to specified pose"""
        if group_name is None:
            group_name = self.robot_group

        # Create MoveGroup goal
        goal = MoveGroup.Goal()
        goal.request.group_name = group_name
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 10.0

        # Set target pose constraint
        pose_constraint = Constraints()
        position_constraint = PositionConstraint()
        position_constraint.link_name = self.end_effector_link
        position_constraint.header.frame_id = "base_link"
        position_constraint.position = target_pose.position

        # Set position tolerance
        position_constraint.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01]))
        position_constraint.weight = 1.0

        # Set orientation constraint
        orientation_constraint = OrientationConstraint()
        orientation_constraint.link_name = self.end_effector_link
        orientation_constraint.header.frame_id = "base_link"
        orientation_constraint.orientation = target_pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0

        pose_constraint.position_constraints.append(position_constraint)
        pose_constraint.orientation_constraints.append(orientation_constraint)

        goal.request.goal_constraints.append(pose_constraint)

        # Send goal and wait for result
        self.move_group_client.wait_for_server()
        future = self.move_group_client.send_goal_async(goal)

        # Wait for result with timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        if future.result() is not None:
            goal_handle = future.result()
            if goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)

                if result_future.result() is not None:
                    result = result_future.result().result
                    return result.error_code.val == MoveItErrorCodes.SUCCESS
                else:
                    return False
            else:
                return False
        else:
            return False

    def pick_object(self, object_name: str, grasp_pose: Pose = None) -> bool:
        """Pick up an object using pre-planned grasps or provided pose"""
        if grasp_pose is None:
            # Plan grasp based on object type
            grasp_pose = self._plan_grasp_for_object(object_name)
            if grasp_pose is None:
                self.get_logger().error(f"Could not plan grasp for object: {object_name}")
                return False

        # Create pick goal
        goal = PickUp.Goal()
        goal.target_name = object_name
        goal.group_name = self.robot_group
        goal.end_effector = self.end_effector_link
        goal.allow_gripper_support_collision = True

        # Add pre-grasp and grasp poses
        pre_grasp_pose = self._calculate_pre_grasp_pose(grasp_pose)
        goal.pre_grasp_approach.direction.vector.z = -1.0  # Approach from above
        goal.pre_grasp_approach.min_distance = 0.05
        goal.pre_grasp_approach.desired_distance = 0.1

        goal.grasp_pose = grasp_pose
        goal.grasp_grasp.direction.vector.z = 1.0  # Grasp direction
        goal.grasp_grasp.min_distance = 0.02
        goal.grasp_grasp.desired_distance = 0.05

        # Send pick goal
        self.pick_client.wait_for_server()
        future = self.pick_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() is not None:
            goal_handle = future.result()
            if goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)

                if result_future.result() is not None:
                    result = result_future.result().result
                    return result.error_code.val == MoveItErrorCodes.SUCCESS
                else:
                    return False
            else:
                return False
        else:
            return False

    def place_object(self, object_name: str, place_pose: Pose) -> bool:
        """Place object at specified location"""
        goal = Place.Goal()
        goal.group_name = self.robot_group
        goal.attached_object_name = object_name
        goal.place_locations = [place_pose]

        # Configure approach and lift
        goal.place_location.pre_place_approach.direction.vector.z = -1.0
        goal.place_location.pre_place_approach.min_distance = 0.05
        goal.place_location.pre_place_approach.desired_distance = 0.1

        goal.place_location.post_place_retreat.direction.vector.x = -1.0
        goal.place_location.post_place_retreat.min_distance = 0.05
        goal.place_location.post_place_retreat.desired_distance = 0.1

        # Send place goal
        self.place_client.wait_for_server()
        future = self.place_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() is not None:
            goal_handle = future.result()
            if goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)

                if result_future.result() is not None:
                    result = result_future.result().result
                    return result.error_code.val == MoveItErrorCodes.SUCCESS
                else:
                    return False
            else:
                return False
        else:
            return False

    def _plan_grasp_for_object(self, object_name: str) -> Optional[Pose]:
        """Plan grasp pose for known object type"""
        # Get object information from perception system
        object_info = self._get_object_info(object_name)
        if not object_info:
            return None

        object_type = object_info.get('type', 'unknown')
        object_pose = object_info.get('pose')

        if object_type in self.grasp_database:
            grasp_config = self.grasp_database[object_type]

            # Calculate grasp pose based on object type and pose
            grasp_pose = Pose()
            grasp_pose.position = object_pose.position

            # Apply offset based on object type
            offset = grasp_config['grasp_pose_offset']
            grasp_pose.position.x += offset[0]
            grasp_pose.position.y += offset[1]
            grasp_pose.position.z += offset[2]

            # Set appropriate orientation based on approach direction
            approach_dir = grasp_config['approach_direction']
            grasp_pose.orientation = self._calculate_grasp_orientation(
                approach_dir, object_pose.orientation)

            return grasp_pose
        else:
            # Use generic grasp planning for unknown objects
            return self._plan_generic_grasp(object_pose)

    def _calculate_pre_grasp_pose(self, grasp_pose: Pose) -> Pose:
        """Calculate pre-grasp pose by moving away from grasp pose"""
        pre_grasp = Pose()
        pre_grasp.position = grasp_pose.position

        # Move back by approach distance along approach direction
        # Assuming approach is from above (z-axis)
        pre_grasp.position.z += self.approach_distance
        pre_grasp.orientation = grasp_pose.orientation

        return pre_grasp

    def _calculate_grasp_orientation(self, approach_dir: List[float],
                                   object_orientation: Quaternion) -> Quaternion:
        """Calculate appropriate grasp orientation based on approach direction"""
        # This is a simplified orientation calculation
        # In practice, you'd use more sophisticated orientation planning

        # For now, return a default orientation
        # In a real system, you'd calculate orientation based on:
        # - Approach direction
        # - Object orientation
        # - Grasp type requirements
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
```

### Grasp Planning and Execution

Implement advanced grasp planning for various object types:

```python
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import tf_transformations

class AdvancedGraspPlanner:
    """Advanced grasp planning using point cloud data"""

    def __init__(self):
        self.gripper_width_limits = (0.02, 0.15)  # meters
        self.min_grasp_quality = 0.7
        self.max_grasps_to_generate = 20

    def plan_grasps_from_point_cloud(self, point_cloud: np.ndarray,
                                   object_centroid: np.ndarray) -> List[Dict]:
        """Plan grasps from point cloud data"""
        # Convert numpy array to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # Downsample for faster processing
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.01)

        # Estimate normals
        pcd_downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02, max_nn=30))

        # Generate grasp candidates
        grasp_candidates = self._generate_grasp_candidates(pcd_downsampled, object_centroid)

        # Evaluate grasp quality
        valid_grasps = []
        for grasp in grasp_candidates:
            quality = self._evaluate_grasp_quality(grasp, pcd_downsampled)
            if quality >= self.min_grasp_quality:
                grasp['quality'] = quality
                valid_grasps.append(grasp)

        # Sort by quality
        valid_grasps.sort(key=lambda x: x['quality'], reverse=True)

        return valid_grasps[:self.max_grasps_to_generate]

    def _generate_grasp_candidates(self, pcd: o3d.geometry.PointCloud,
                                 object_centroid: np.ndarray) -> List[Dict]:
        """Generate potential grasp candidates"""
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        grasp_candidates = []

        for i in range(len(points)):
            point = points[i]
            normal = normals[i]

            # Skip points too far from centroid
            if np.linalg.norm(point - object_centroid) > 0.3:  # 30cm radius
                continue

            # Generate grasp poses around the surface normal
            for angle_offset in [0, 45, 90, 135]:  # Different orientations
                grasp_pose = self._create_grasp_pose(point, normal, angle_offset)

                grasp_candidates.append({
                    'pose': grasp_pose,
                    'contact_point': point,
                    'surface_normal': normal
                })

        return grasp_candidates

    def _create_grasp_pose(self, contact_point: np.ndarray,
                         surface_normal: np.ndarray,
                         angle_offset: float) -> Pose:
        """Create grasp pose from contact point and surface normal"""
        pose = Pose()

        # Set position
        pose.position.x = float(contact_point[0])
        pose.position.y = float(contact_point[1])
        pose.position.z = float(contact_point[2])

        # Calculate orientation based on surface normal and angle offset
        # Create a rotation matrix from the approach direction
        approach_dir = -surface_normal  # Grasp from opposite of surface normal

        # Create orthogonal vectors for complete orientation
        if abs(approach_dir[2]) < 0.9:
            # Use z-axis as reference if approach is not vertical
            ortho1 = np.cross(approach_dir, [0, 0, 1])
        else:
            # Use x-axis as reference if approach is nearly vertical
            ortho1 = np.cross(approach_dir, [1, 0, 0])

        ortho1 = ortho1 / np.linalg.norm(ortho1)
        ortho2 = np.cross(approach_dir, ortho1)

        # Create rotation matrix
        rotation_matrix = np.column_stack([ortho1, ortho2, approach_dir])

        # Add angle offset around approach direction
        rotation_obj = R.from_rotvec(angle_offset * np.pi / 180.0 * approach_dir)
        rotation_matrix = rotation_matrix @ rotation_obj.as_matrix()

        # Convert to quaternion
        quat = R.from_matrix(rotation_matrix).as_quat()
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])

        return pose

    def _evaluate_grasp_quality(self, grasp: Dict, pcd: o3d.geometry.PointCloud) -> float:
        """Evaluate grasp quality based on geometric criteria"""
        contact_point = grasp['contact_point']
        approach_dir = -grasp['surface_normal']  # Opposite of surface normal

        # Check if grasp is stable (not too steep)
        stability_score = self._evaluate_stability(approach_dir)

        # Check if grasp is accessible (not blocked by other geometry)
        accessibility_score = self._evaluate_accessibility(
            contact_point, approach_dir, pcd)

        # Combine scores
        quality = 0.6 * stability_score + 0.4 * accessibility_score

        return min(quality, 1.0)

    def _evaluate_stability(self, approach_dir: np.ndarray) -> float:
        """Evaluate grasp stability based on approach direction"""
        # Prefer grasps that are not too steep (z-component should be significant)
        vertical_alignment = abs(approach_dir[2])

        # Prefer upward-facing grasps for stability
        if approach_dir[2] > 0:
            return vertical_alignment
        else:
            # Downward grasps are less stable
            return vertical_alignment * 0.7

    def _evaluate_accessibility(self, contact_point: np.ndarray,
                              approach_dir: np.ndarray,
                              pcd: o3d.geometry.PointCloud) -> float:
        """Evaluate if grasp approach is accessible"""
        points = np.asarray(pcd.points)

        # Check for obstacles along approach direction
        approach_vector = approach_dir * 0.05  # 5cm approach distance

        # Create approach path
        approach_start = contact_point - approach_vector
        approach_end = contact_point + approach_vector * 2

        # Check for points in approach path
        path_points = points[
            (np.linalg.norm(points - approach_start, axis=1) < 0.02) |
            (np.linalg.norm(points - approach_end, axis=1) < 0.02)
        ]

        # If no obstacles in path, it's accessible
        accessibility = 1.0 if len(path_points) == 0 else 0.3

        return accessibility
```

## Human-Robot Interaction Systems

### Interaction Protocol Design

Implement safe and intuitive human-robot interaction protocols:

```python
class HumanRobotInteractionController(Node):
    """Controller for safe human-robot interaction"""

    def __init__(self):
        super().__init__('human_robot_interaction_controller')

        # Publishers and subscribers
        self.interaction_status_pub = self.create_publisher(
            String, 'interaction_status', 10)
        self.human_detection_sub = self.create_subscription(
            Detection2DArray, 'human_detections', self.human_detected_callback, 10)
        self.voice_command_sub = self.create_subscription(
            String, 'voice_command', self.voice_command_callback, 10)
        self.interaction_request_pub = self.create_publisher(
            String, 'interaction_request', 10)

        # Safety and proximity monitoring
        self.human_proximity_threshold = 1.0  # meters
        self.safety_zones = {}  # Track humans and their positions
        self.interaction_mode = "autonomous"  # or "collaborative", "supervised"

        # Interaction state management
        self.current_interaction = None
        self.interaction_queue = []
        self.interaction_history = []

        # Timer for safety monitoring
        self.safety_timer = self.create_timer(0.1, self._monitor_safety)

        self.get_logger().info("Human-Robot Interaction Controller initialized")

    def human_detected_callback(self, msg: Detection2DArray):
        """Process human detections for interaction management"""
        current_humans = {}

        for detection in msg.detections:
            # Check if this is a human detection
            if detection.results:
                best_result = max(detection.results,
                                key=lambda x: x.hypothesis.score)

                if best_result.hypothesis.class_id == "person":
                    human_id = f"person_{len(self.safety_zones)}"
                    position = self._convert_bbox_to_position(detection.bbox)

                    current_humans[human_id] = {
                        'position': position,
                        'confidence': best_result.hypothesis.score,
                        'timestamp': self.get_clock().now().to_msg()
                    }

        # Update safety zones
        self.safety_zones = current_humans

        # Check for potential interaction opportunities
        if self.interaction_mode == "collaborative":
            self._check_interaction_opportunities()

    def voice_command_callback(self, msg: String):
        """Process voice commands that may involve human interaction"""
        command = msg.data.lower()

        # Check for interaction-specific commands
        interaction_keywords = ['help', 'assist', 'work with', 'collaborate', 'follow']

        for keyword in interaction_keywords:
            if keyword in command:
                self._initiate_interaction(command)
                break

    def _initiate_interaction(self, command: str):
        """Initiate human-robot interaction based on command"""
        if not self.safety_zones:
            self.get_logger().warn("No humans detected for interaction")
            return

        # Find closest human for interaction
        closest_human = self._find_closest_human()
        if closest_human:
            interaction_request = {
                'type': 'initiate_interaction',
                'target_human': closest_human[0],
                'command': command,
                'timestamp': self.get_clock().now().to_msg()
            }

            request_msg = String()
            request_msg.data = json.dumps(interaction_request)
            self.interaction_request_pub.publish(request_msg)

            # Update interaction state
            self.current_interaction = interaction_request
            self.interaction_history.append(interaction_request)

    def _find_closest_human(self) -> Optional[Tuple[str, Dict]]:
        """Find the closest human to the robot"""
        if not self.safety_zones:
            return None

        robot_position = self._get_robot_position()
        if robot_position is None:
            return None

        closest_human = None
        min_distance = float('inf')

        for human_id, human_info in self.safety_zones.items():
            human_pos = human_info['position']
            distance = np.sqrt(
                (human_pos.x - robot_position.x)**2 +
                (human_pos.y - robot_position.y)**2
            )

            if distance < min_distance:
                min_distance = distance
                closest_human = (human_id, human_info)

        return closest_human

    def _check_interaction_opportunities(self):
        """Check for opportunities to initiate interaction"""
        if self.current_interaction is not None:
            # Interaction already in progress
            return

        # Check if any human is within interaction distance
        robot_position = self._get_robot_position()
        if robot_position is None:
            return

        for human_id, human_info in self.safety_zones.items():
            human_pos = human_info['position']
            distance = np.sqrt(
                (human_pos.x - robot_position.x)**2 +
                (human_pos.y - robot_position.y)**2
            )

            if distance < 2.0:  # Within 2 meters
                # Potential interaction opportunity
                self._request_interaction_permission(human_id)

    def _request_interaction_permission(self, human_id: str):
        """Request permission to interact with a human"""
        # This would typically involve visual/audible signals
        # For now, we'll simulate permission granted
        interaction_request = {
            'type': 'request_permission',
            'target_human': human_id,
            'request_type': 'approach',
            'timestamp': self.get_clock().now().to_msg()
        }

        request_msg = String()
        request_msg.data = json.dumps(interaction_request)
        self.interaction_request_pub.publish(request_msg)

    def _monitor_safety(self):
        """Monitor safety during human-robot interaction"""
        if not self.safety_zones:
            return

        robot_position = self._get_robot_position()
        if robot_position is None:
            return

        for human_id, human_info in self.safety_zones.items():
            human_pos = human_info['position']
            distance = np.sqrt(
                (human_pos.x - robot_position.x)**2 +
                (human_pos.y - robot_position.y)**2
            )

            if distance < self.human_proximity_threshold:
                # Human is too close, trigger safety response
                safety_alert = {
                    'type': 'safety_violation',
                    'human_id': human_id,
                    'distance': distance,
                    'timestamp': self.get_clock().now().to_msg()
                }

                # Publish safety alert
                alert_msg = String()
                alert_msg.data = json.dumps(safety_alert)
                self.interaction_status_pub.publish(alert_msg)

                # Trigger safety response based on interaction mode
                self._handle_safety_violation(safety_alert)

    def _handle_safety_violation(self, alert: Dict):
        """Handle safety violation during interaction"""
        if self.interaction_mode == "autonomous":
            # Stop all motion
            self._emergency_stop()
        elif self.interaction_mode == "collaborative":
            # Pause manipulation, maintain safe distance
            self._pause_manipulation()
        elif self.interaction_mode == "supervised":
            # Alert supervisor and wait for instructions
            self._alert_supervisor()

    def _convert_bbox_to_position(self, bbox: BoundingBox2D) -> Point:
        """Convert 2D bounding box to 3D position estimate"""
        # This is a simplified conversion
        # In practice, you'd use depth information
        return Point(
            x=float(bbox.center.x),
            y=float(bbox.center.y),
            z=0.0  # Assume ground level for now
        )

    def _get_robot_position(self) -> Optional[Point]:
        """Get current robot position from localization"""
        # This would interface with the localization system
        # For now, returning a placeholder
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            return Point(
                x=transform.transform.translation.x,
                y=transform.transform.translation.y,
                z=transform.transform.translation.z
            )
        except:
            return None

    def _emergency_stop(self):
        """Emergency stop for safety violations"""
        self.get_logger().warn("Emergency stop triggered due to safety violation")
        # Implementation would send stop commands to all controllers

    def _pause_manipulation(self):
        """Pause manipulation while maintaining safety"""
        self.get_logger().info("Pausing manipulation due to safety concern")
        # Implementation would pause current manipulation task

    def _alert_supervisor(self):
        """Alert supervisor for supervised mode"""
        self.get_logger().info("Alerting supervisor for safety decision")
        # Implementation would notify human supervisor
```

### Multimodal Interaction Interface

Create a multimodal interaction interface that combines voice, vision, and touch:

```python
class MultimodalInteractionInterface(Node):
    """Interface for multimodal human-robot interaction"""

    def __init__(self):
        super().__init__('multimodal_interaction_interface')

        # Publishers and subscribers
        self.interaction_response_pub = self.create_publisher(
            String, 'interaction_response', 10)
        self.audio_response_pub = self.create_publisher(
            String, 'audio_response', 10)
        self.visual_feedback_pub = self.create_publisher(
            MarkerArray, 'visual_feedback', 10)

        # Subscriptions from various modalities
        self.voice_command_sub = self.create_subscription(
            String, 'voice_command', self.voice_callback, 10)
        self.gesture_sub = self.create_subscription(
            Gesture, 'hand_gestures', self.gesture_callback, 10)
        self.touch_sub = self.create_subscription(
            TouchEvent, 'touch_events', self.touch_callback, 10)
        self.vision_command_sub = self.create_subscription(
            String, 'vision_commands', self.vision_callback, 10)

        # Interaction state
        self.interaction_context = {
            'current_user': None,
            'interaction_history': [],
            'attention_objects': [],
            'pending_requests': []
        }

        # Response generation
        self.response_generator = ResponseGenerator()

        self.get_logger().info("Multimodal Interaction Interface initialized")

    def voice_callback(self, msg: String):
        """Handle voice commands"""
        command = msg.data
        self.get_logger().info(f"Processing voice command: {command}")

        # Parse voice command and generate response
        response = self._process_voice_command(command)
        self._publish_response(response, 'voice')

    def gesture_callback(self, msg: Gesture):
        """Handle gesture commands"""
        self.get_logger().info(f"Processing gesture: {msg.type}")

        # Process gesture and generate response
        response = self._process_gesture(msg)
        self._publish_response(response, 'gesture')

    def touch_callback(self, msg: TouchEvent):
        """Handle touch events"""
        self.get_logger().info(f"Processing touch event: {msg.location}")

        # Process touch event and generate response
        response = self._process_touch_event(msg)
        self._publish_response(response, 'touch')

    def vision_callback(self, msg: String):
        """Handle vision-based commands"""
        command = msg.data
        self.get_logger().info(f"Processing vision command: {command}")

        # Process vision command and generate response
        response = self._process_vision_command(command)
        self._publish_response(response, 'vision')

    def _process_voice_command(self, command: str) -> Dict:
        """Process voice command and generate response"""
        # Simple command parsing - in practice, use NLP
        command_lower = command.lower()

        if 'hello' in command_lower or 'hi' in command_lower:
            return {
                'type': 'greeting',
                'response': 'Hello! How can I assist you today?',
                'action': 'wave'
            }
        elif 'help' in command_lower:
            return {
                'type': 'assistance_request',
                'response': 'I can help you with navigation, object manipulation, or information lookup.',
                'action': 'point_to_self'
            }
        elif 'pick' in command_lower or 'grasp' in command_lower:
            # Extract object from command
            object_name = self._extract_object_from_command(command)
            return {
                'type': 'manipulation_request',
                'response': f'I will pick up the {object_name} for you.',
                'action': 'manipulation',
                'object': object_name
            }
        else:
            return {
                'type': 'unknown_command',
                'response': 'I did not understand that command. Could you please repeat?',
                'action': 'confused'
            }

    def _process_gesture(self, gesture: Gesture) -> Dict:
        """Process gesture and generate response"""
        gesture_type = gesture.type.lower()

        if gesture_type == 'pointing':
            return {
                'type': 'object_identification',
                'response': 'I see you are pointing at something. What would you like me to do with it?',
                'action': 'look_at_pointed_location'
            }
        elif gesture_type == 'beckoning':
            return {
                'type': 'approach_request',
                'response': 'I am coming to you now.',
                'action': 'approach_user'
            }
        elif gesture_type == 'stop':
            return {
                'type': 'stop_request',
                'response': 'I will stop my current action.',
                'action': 'stop_motion'
            }
        else:
            return {
                'type': 'unknown_gesture',
                'response': 'I did not recognize that gesture.',
                'action': 'idle'
            }

    def _process_touch_event(self, touch_event: TouchEvent) -> Dict:
        """Process touch event and generate response"""
        location = touch_event.location

        if 'head' in location.lower():
            return {
                'type': 'head_touch',
                'response': 'Hello! I feel your touch on my head.',
                'action': 'happy_animation'
            }
        elif 'hand' in location.lower():
            return {
                'type': 'hand_touch',
                'response': 'Nice to meet you! I feel your handshake.',
                'action': 'greeting_response'
            }
        else:
            return {
                'type': 'touch_response',
                'response': 'I feel your touch.',
                'action': 'acknowledge_touch'
            }

    def _process_vision_command(self, command: str) -> Dict:
        """Process vision-based command"""
        # Vision commands typically come from object detection or scene analysis
        if 'red cup' in command.lower():
            return {
                'type': 'object_identification',
                'response': 'I see a red cup. Would you like me to pick it up?',
                'action': 'point_to_object',
                'object': 'red cup'
            }
        elif 'person' in command.lower():
            return {
                'type': 'person_detection',
                'response': 'I see a person. How can I assist?',
                'action': 'greeting_pose'
            }
        else:
            return {
                'type': 'scene_analysis',
                'response': f'I see {command}. How can I help?',
                'action': 'curious_pose'
            }

    def _extract_object_from_command(self, command: str) -> str:
        """Extract object name from command"""
        # Simple extraction - in practice, use NLP
        words = command.lower().split()
        common_objects = ['cup', 'bottle', 'book', 'box', 'apple', 'banana', 'phone']

        for word in words:
            if word in common_objects:
                return word

        # If no common object found, return the noun after 'the' or 'a'
        for i, word in enumerate(words):
            if word in ['the', 'a', 'an'] and i + 1 < len(words):
                return words[i + 1]

        return 'object'  # Default if no object found

    def _publish_response(self, response: Dict, modality: str):
        """Publish response through appropriate modalities"""
        # Publish to main interaction response
        response_msg = String()
        response_msg.data = json.dumps(response)
        self.interaction_response_pub.publish(response_msg)

        # Publish audio response
        if 'response' in response:
            audio_msg = String()
            audio_msg.data = response['response']
            self.audio_response_pub.publish(audio_msg)

        # Publish visual feedback if needed
        if response.get('action') == 'point_to_object':
            self._publish_pointing_marker(response.get('object'))
        elif response.get('action') == 'look_at_pointed_location':
            self._publish_attention_marker()

    def _publish_pointing_marker(self, object_name: str):
        """Publish visual marker for pointing action"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "interaction"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set arrow properties
        marker.scale.x = 0.5  # shaft diameter
        marker.scale.y = 0.1  # head diameter
        marker.scale.z = 0.1  # head length
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0  # Yellow arrow

        # Position and orientation would be set based on object location
        marker.pose.position.x = 1.0
        marker.pose.position.y = 1.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.text = f"Pointing to {object_name}"

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.visual_feedback_pub.publish(marker_array)

    def _publish_attention_marker(self):
        """Publish visual marker for attention focus"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "interaction"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 0.7
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0  # Green sphere

        marker.pose.position.x = 1.5
        marker.pose.position.y = 1.5
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0

        marker.text = "Attention Focus"

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.visual_feedback_pub.publish(marker_array)

class ResponseGenerator:
    """Generate appropriate responses for different interaction scenarios"""

    def __init__(self):
        self.response_templates = {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Greetings! I'm ready to help."
            ],
            'acknowledgment': [
                "I understand.",
                "Got it.",
                "I'll take care of that for you."
            ],
            'confusion': [
                "I didn't quite understand that. Could you please repeat?",
                "I'm not sure what you mean. Can you clarify?",
                "I didn't catch that. Could you say it again?"
            ],
            'success': [
                "Task completed successfully!",
                "I've finished what you asked.",
                "All done!"
            ],
            'failure': [
                "I couldn't complete that task. Is there something else I can help with?",
                "I encountered an issue. Would you like me to try again?",
                "I'm sorry, I couldn't do that."
            ]
        }

    def generate_response(self, response_type: str, context: str = "") -> str:
        """Generate response based on type and context"""
        import random

        if response_type in self.response_templates:
            responses = self.response_templates[response_type]
            return random.choice(responses)
        else:
            return "I'm not sure how to respond to that."
```

## Safety and Validation Systems

### Manipulation Safety Validator

Implement comprehensive safety validation for manipulation tasks:

```python
class ManipulationSafetyValidator:
    """Validate manipulation tasks for safety compliance"""

    def __init__(self):
        self.safety_rules = self._load_safety_rules()
        self.robot_limits = self._load_robot_limits()
        self.environment_constraints = {}

    def _load_safety_rules(self) -> Dict:
        """Load safety rules for manipulation"""
        return {
            'human_safety': {
                'minimum_distance': 0.5,  # meters
                'collision_threshold': 0.1  # meters
            },
            'robot_safety': {
                'joint_limits': True,
                'force_limits': True,
                'velocity_limits': True
            },
            'object_safety': {
                'fragile_objects': ['glass', 'ceramic', 'electronics'],
                'hazardous_objects': ['sharp', 'hot', 'chemical']
            },
            'environment_safety': {
                'restricted_zones': [],
                'obstacle_avoidance': True
            }
        }

    def _load_robot_limits(self) -> Dict:
        """Load robot-specific limits"""
        return {
            'max_payload': 2.0,  # kg
            'max_velocity': 1.0,  # m/s
            'max_force': 50.0,    # N
            'workspace_limits': {
                'min_x': -1.0, 'max_x': 1.0,
                'min_y': -1.0, 'max_y': 1.0,
                'min_z': 0.1, 'max_z': 2.0
            }
        }

    def validate_manipulation_task(self, task: Dict, environment_state: Dict) -> Dict:
        """Validate a manipulation task for safety compliance"""
        validation_results = {
            'task_safe': True,
            'issues': [],
            'recommendations': [],
            'risk_level': 'low'
        }

        # Check human safety
        human_safety_ok, human_issues = self._check_human_safety(task, environment_state)
        if not human_safety_ok:
            validation_results['task_safe'] = False
            validation_results['issues'].extend(human_issues)

        # Check robot limits
        robot_limits_ok, robot_issues = self._check_robot_limits(task)
        if not robot_limits_ok:
            validation_results['task_safe'] = False
            validation_results['issues'].extend(robot_issues)

        # Check object safety
        object_safety_ok, object_issues = self._check_object_safety(task)
        if not object_safety_ok:
            validation_results['task_safe'] = False
            validation_results['issues'].extend(object_issues)

        # Check environment constraints
        env_ok, env_issues = self._check_environment_constraints(task, environment_state)
        if not env_ok:
            validation_results['task_safe'] = False
            validation_results['issues'].extend(env_issues)

        # Calculate risk level
        validation_results['risk_level'] = self._calculate_risk_level(
            validation_results['issues'])

        # Generate recommendations for unsafe tasks
        if not validation_results['task_safe']:
            validation_results['recommendations'] = self._generate_recommendations(
                validation_results['issues'])

        return validation_results

    def _check_human_safety(self, task: Dict, environment_state: Dict) -> Tuple[bool, List[str]]:
        """Check if manipulation task is safe regarding humans"""
        issues = []

        humans = environment_state.get('humans', [])
        task_target = task.get('target_position', {})

        for human in humans:
            human_pos = human.get('position', {})
            distance = self._calculate_distance(task_target, human_pos)

            if distance < self.safety_rules['human_safety']['minimum_distance']:
                issues.append(f"Task target too close to human at position {human_pos}")

        return len(issues) == 0, issues

    def _check_robot_limits(self, task: Dict) -> Tuple[bool, List[str]]:
        """Check if task respects robot physical limits"""
        issues = []

        # Check payload
        object_weight = task.get('object_weight', 0)
        if object_weight > self.robot_limits['max_payload']:
            issues.append(f"Object weight {object_weight}kg exceeds maximum payload {self.robot_limits['max_payload']}kg")

        # Check workspace limits
        target_pos = task.get('target_position', {})
        if target_pos:
            if (target_pos.get('x', 0) < self.robot_limits['workspace_limits']['min_x'] or
                target_pos.get('x', 0) > self.robot_limits['workspace_limits']['max_x'] or
                target_pos.get('y', 0) < self.robot_limits['workspace_limits']['min_y'] or
                target_pos.get('y', 0) > self.robot_limits['workspace_limits']['max_y'] or
                target_pos.get('z', 0) < self.robot_limits['workspace_limits']['min_z'] or
                target_pos.get('z', 0) > self.robot_limits['workspace_limits']['max_z']):
                issues.append("Target position outside robot workspace limits")

        return len(issues) == 0, issues

    def _check_object_safety(self, task: Dict) -> Tuple[bool, List[str]]:
        """Check if object manipulation is safe"""
        issues = []

        object_type = task.get('object_type', '').lower()
        object_name = task.get('object_name', '').lower()

        # Check if object is fragile
        for fragile in self.safety_rules['object_safety']['fragile_objects']:
            if fragile in object_type or fragile in object_name:
                issues.append(f"Handling fragile object '{object_name}' requires extra care")

        # Check if object is hazardous
        for hazardous in self.safety_rules['object_safety']['hazardous_objects']:
            if hazardous in object_type or hazardous in object_name:
                issues.append(f"Object '{object_name}' may be hazardous to handle")

        return len(issues) == 0, issues

    def _check_environment_constraints(self, task: Dict, environment_state: Dict) -> Tuple[bool, List[str]]:
        """Check environment-specific constraints"""
        issues = []

        # Check restricted zones
        target_pos = task.get('target_position', {})
        for zone in self.safety_rules['environment_safety']['restricted_zones']:
            if self._point_in_zone(target_pos, zone):
                issues.append(f"Target position in restricted zone: {zone['name']}")

        # Check obstacles
        obstacles = environment_state.get('obstacles', [])
        approach_path = self._calculate_approach_path(task)

        for obstacle in obstacles:
            if self._path_intersects_obstacle(approach_path, obstacle):
                issues.append(f"Approach path intersects with obstacle at {obstacle['position']}")

        return len(issues) == 0, issues

    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate distance between two positions"""
        dx = pos1.get('x', 0) - pos2.get('x', 0)
        dy = pos1.get('y', 0) - pos2.get('y', 0)
        dz = pos1.get('z', 0) - pos2.get('z', 0)
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def _calculate_approach_path(self, task: Dict) -> List[Dict]:
        """Calculate approach path for manipulation task"""
        # Simplified path calculation
        # In practice, this would use full motion planning
        target = task.get('target_position', {})
        approach_distance = 0.1  # 10cm approach

        # Calculate approach point (10cm before target)
        approach_point = {
            'x': target.get('x', 0) - approach_distance,
            'y': target.get('y', 0),
            'z': target.get('z', 0)
        }

        return [approach_point, target]

    def _point_in_zone(self, point: Dict, zone: Dict) -> bool:
        """Check if point is within a zone"""
        zone_center = zone.get('center', {})
        zone_radius = zone.get('radius', 0)

        distance = self._calculate_distance(point, zone_center)
        return distance <= zone_radius

    def _path_intersects_obstacle(self, path: List[Dict], obstacle: Dict) -> bool:
        """Check if path intersects with obstacle"""
        obstacle_pos = obstacle.get('position', {})
        obstacle_radius = obstacle.get('radius', 0.1)

        for point in path:
            distance = self._calculate_distance(point, obstacle_pos)
            if distance <= obstacle_radius:
                return True
        return False

    def _calculate_risk_level(self, issues: List[str]) -> str:
        """Calculate risk level based on issues"""
        if not issues:
            return 'low'
        elif len(issues) <= 2:
            return 'medium'
        else:
            return 'high'

    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate safety recommendations based on issues"""
        recommendations = []

        for issue in issues:
            if 'human' in issue.lower():
                recommendations.append("Maintain safe distance from humans during manipulation")
            elif 'payload' in issue.lower():
                recommendations.append("Use appropriate gripping force for object weight")
            elif 'workspace' in issue.lower():
                recommendations.append("Verify target position is within robot reach")
            elif 'fragile' in issue.lower():
                recommendations.append("Use gentle manipulation for fragile objects")
            elif 'hazardous' in issue.lower():
                recommendations.append("Consider safety equipment when handling hazardous objects")

        return recommendations
```

## Integration with VLA System

### Main Integration Node

Create the main integration node that connects all manipulation and interaction components:

```python
class ManipulationInteractionNode(Node):
    """Main integration node for manipulation and interaction systems"""

    def __init__(self):
        super().__init__('manipulation_interaction_node')

        # Initialize components
        self.manipulation_controller = ManipulationController()
        self.interaction_controller = HumanRobotInteractionController()
        self.multimodal_interface = MultimodalInteractionInterface()
        self.safety_validator = ManipulationSafetyValidator()

        # Publishers
        self.status_pub = self.create_publisher(String, 'manipulation_status', 10)
        self.action_request_pub = self.create_publisher(
            String, 'action_requests', 10)

        # Subscriptions
        self.task_sub = self.create_subscription(
            String, 'planned_tasks', self.task_callback, 10)
        self.environment_sub = self.create_subscription(
            String, 'integrated_perception', self.environment_callback, 10)

        # Internal state
        self.current_task = None
        self.environment_state = {}
        self.manipulation_queue = []

        # Timer for task execution
        self.execution_timer = self.create_timer(0.1, self._execute_tasks)

        self.get_logger().info("Manipulation and Interaction Node initialized")

    def task_callback(self, msg: String):
        """Handle incoming manipulation tasks"""
        try:
            task_data = json.loads(msg.data)

            # Validate task for safety
            validation = self.safety_validator.validate_manipulation_task(
                task_data, self.environment_state)

            if validation['task_safe']:
                # Add safe task to execution queue
                self.manipulation_queue.append(task_data)
                self.get_logger().info(f"Added safe task to queue: {task_data.get('type', 'unknown')}")
            else:
                self.get_logger().warn(f"Unsafe task rejected: {validation['issues']}")
                # Notify system of unsafe task
                self._notify_unsafe_task(task_data, validation)

        except json.JSONDecodeError:
            self.get_logger().error("Invalid task data received")

    def environment_callback(self, msg: String):
        """Update environment state"""
        try:
            self.environment_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid environment data")

    def _execute_tasks(self):
        """Execute tasks in the queue"""
        if self.manipulation_queue and self.current_task is None:
            # Start next task
            self.current_task = self.manipulation_queue.pop(0)
            self._execute_current_task()

    def _execute_current_task(self):
        """Execute the current manipulation task"""
        if not self.current_task:
            return

        task_type = self.current_task.get('type', 'unknown')

        try:
            if task_type == 'pick_object':
                success = self._execute_pick_task(self.current_task)
            elif task_type == 'place_object':
                success = self._execute_place_task(self.current_task)
            elif task_type == 'move_to_pose':
                success = self._execute_move_task(self.current_task)
            elif task_type == 'interact_with_human':
                success = self._execute_interaction_task(self.current_task)
            else:
                self.get_logger().warn(f"Unknown task type: {task_type}")
                success = False

            if success:
                self.get_logger().info(f"Task completed successfully: {task_type}")
            else:
                self.get_logger().error(f"Task failed: {task_type}")

        except Exception as e:
            self.get_logger().error(f"Error executing task {task_type}: {e}")
            success = False

        finally:
            self.current_task = None

    def _execute_pick_task(self, task: Dict) -> bool:
        """Execute object pick task"""
        object_name = task.get('object_name')
        object_pose = task.get('object_pose')

        if object_pose:
            # Use provided pose
            pose = Pose()
            pose.position.x = object_pose.get('x', 0)
            pose.position.y = object_pose.get('y', 0)
            pose.position.z = object_pose.get('z', 0)
            # Set orientation appropriately
            pose.orientation.w = 1.0
        else:
            # Plan grasp based on object name
            pose = None

        return self.manipulation_controller.pick_object(object_name, pose)

    def _execute_place_task(self, task: Dict) -> bool:
        """Execute object place task"""
        object_name = task.get('object_name')
        place_pose = task.get('place_pose')

        if place_pose:
            pose = Pose()
            pose.position.x = place_pose.get('x', 0)
            pose.position.y = place_pose.get('y', 0)
            pose.position.z = place_pose.get('z', 0)
            pose.orientation.w = 1.0

            return self.manipulation_controller.place_object(object_name, pose)

        return False

    def _execute_move_task(self, task: Dict) -> bool:
        """Execute move to pose task"""
        target_pose = task.get('target_pose')

        if target_pose:
            pose = Pose()
            pose.position.x = target_pose.get('x', 0)
            pose.position.y = target_pose.get('y', 0)
            pose.position.z = target_pose.get('z', 0)
            pose.orientation.w = 1.0

            return self.manipulation_controller.move_to_pose(pose)

        return False

    def _execute_interaction_task(self, task: Dict) -> bool:
        """Execute human interaction task"""
        interaction_type = task.get('interaction_type')
        target_human = task.get('target_human')

        # Interaction tasks are handled by the interaction controller
        # This is a simplified implementation
        if interaction_type == 'greet':
            response = self.multimodal_interface.response_generator.generate_response('greeting')
            self.get_logger().info(f"Greeting response: {response}")
            return True
        elif interaction_type == 'assist':
            response = self.multimodal_interface.response_generator.generate_response('acknowledgment')
            self.get_logger().info(f"Assistance response: {response}")
            return True

        return False

    def _notify_unsafe_task(self, task: Dict, validation: Dict):
        """Notify system about unsafe task"""
        notification = {
            'type': 'unsafe_task_rejected',
            'task': task,
            'validation_issues': validation['issues'],
            'recommendations': validation['recommendations'],
            'timestamp': self.get_clock().now().to_msg()
        }

        notification_msg = String()
        notification_msg.data = json.dumps(notification)
        self.status_pub.publish(notification_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationInteractionNode()

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

## Implementation Steps

### 1. Set Up the Manipulation Package

Create the ROS 2 package for manipulation and interaction:

```bash
# Create package
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python manipulation_interaction
cd manipulation_interaction
mkdir -p manipulation_interaction/config manipulation_interaction/launch
```

### 2. Install Dependencies

Create requirements file:

```bash
# Create requirements.txt
cat > manipulation_interaction/requirements.txt << EOF
torch>=2.0.0
open3d>=0.17.0
scipy>=1.10.0
numpy>=1.21.0
tf-transformations>=1.0.0
moveit-ros>=2.0.0
opencv-python>=4.8.0
EOF
```

### 3. Configure the System

Create a launch file for the manipulation and interaction system:

```xml
<!-- manipulation_interaction/launch/manipulation_interaction.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory('manipulation_interaction'), 'config')

    return LaunchDescription([
        Node(
            package='manipulation_interaction',
            executable='manipulation_interaction_node',
            name='manipulation_interaction_node',
            parameters=[],
            output='screen'
        ),
        Node(
            package='manipulation_interaction',
            executable='manipulation_controller',
            name='manipulation_controller',
            output='screen'
        ),
        Node(
            package='manipulation_interaction',
            executable='human_robot_interaction_controller',
            name='human_robot_interaction_controller',
            output='screen'
        )
    ])
```

### 4. Testing the System

Create a test script to verify manipulation and interaction:

```python
#!/usr/bin/env python3
# test_manipulation_interaction.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time

class ManipulationTestClient(Node):
    def __init__(self):
        super().__init__('manipulation_test_client')

        # Publishers for testing
        self.task_pub = self.create_publisher(
            String, 'planned_tasks', 10)
        self.voice_pub = self.create_publisher(
            String, 'voice_command', 10)
        self.perception_pub = self.create_publisher(
            String, 'integrated_perception', 10)

        # Subscription to status
        self.status_sub = self.create_subscription(
            String, 'manipulation_status',
            self.status_callback, 10)

        self.timer = self.create_timer(3.0, self.send_test_commands)
        self.command_count = 0

    def send_test_commands(self):
        """Send test commands to manipulation system"""
        test_tasks = [
            {
                'type': 'pick_object',
                'object_name': 'red_cup',
                'object_pose': {'x': 1.0, 'y': 0.5, 'z': 0.2}
            },
            {
                'type': 'place_object',
                'object_name': 'red_cup',
                'place_pose': {'x': 0.5, 'y': 1.0, 'z': 0.2}
            },
            {
                'type': 'interact_with_human',
                'interaction_type': 'greet',
                'target_human': 'person_1'
            }
        ]

        if self.command_count < len(test_tasks):
            task_msg = String()
            task_msg.data = json.dumps(test_tasks[self.command_count])
            self.task_pub.publish(task_msg)

            self.get_logger().info(f"Sent test task: {test_tasks[self.command_count]['type']}")
            self.command_count += 1

    def status_callback(self, msg: String):
        """Handle status updates"""
        self.get_logger().info(f"Received status: {msg.data[:100]}...")

def main(args=None):
    rclpy.init(args=args)
    test_client = ManipulationTestClient()

    try:
        rclpy.spin(test_client)
    except KeyboardInterrupt:
        pass
    finally:
        test_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices and Considerations

### 1. Safety First Approach

- Always validate manipulation tasks before execution
- Implement multiple safety layers (hardware and software)
- Maintain emergency stop capabilities
- Monitor robot and environment continuously

### 2. Grasp Planning Optimization

- Use object-specific grasp strategies
- Consider object properties (weight, fragility, shape)
- Plan approach and retreat trajectories carefully
- Validate grasp stability before execution

### 3. Human-Robot Interaction

- Design intuitive interaction protocols
- Provide clear feedback during interactions
- Respect personal space and comfort zones
- Implement graceful failure handling

### 4. Performance Optimization

- Use efficient collision checking algorithms
- Implement grasp learning for improved performance
- Optimize motion planning for real-time execution
- Cache frequently used grasp configurations

## Troubleshooting

### Common Issues

1. **Grasp Planning Failures**: Verify object detection accuracy and point cloud quality
2. **Safety Violations**: Check safety rule configurations and sensor calibrations
3. **Motion Planning Failures**: Validate robot kinematic models and joint limits
4. **Interaction Timing**: Adjust interaction protocols for natural communication

### Debugging Tips

- Enable detailed logging for each manipulation component
- Monitor robot state and joint positions during execution
- Use RViz for visualizing planned trajectories and grasps
- Test individual components before system integration

This manipulation and interaction systems module provides the physical execution capabilities for Vision-Language-Action systems, enabling humanoid robots to perform complex tasks while maintaining safe and natural interactions with humans and the environment.