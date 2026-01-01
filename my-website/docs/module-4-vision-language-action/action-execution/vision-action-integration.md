---
sidebar_position: 4
---

# Vision-Action Integration

## Introduction

Vision-action integration represents the crucial bridge between perceiving the environment and executing physical actions in humanoid robots. This module explores the complex pipeline that transforms visual input into coordinated robotic movements, enabling robots to interact meaningfully with their surroundings.

## Object Detection Pipelines

### RGB-Based Object Detection

RGB cameras provide rich color information that enables sophisticated object recognition. Modern approaches leverage deep learning models like YOLO (You Only Look Once), SSD (Single Shot Detector), and Mask R-CNN for real-time object detection.

```python
import cv2
import numpy as np
import torch
from torchvision import transforms

class RGBObjectDetector:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((416, 416))
        ])

    def detect_objects(self, image):
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Post-process detections
        boxes = []
        scores = []
        classes = []

        for pred in predictions[0]:
            if pred['score'] > 0.5:
                boxes.append(pred['bbox'])
                scores.append(pred['score'])
                classes.append(pred['class'])

        return boxes, scores, classes
```

### Depth-Based Object Detection

Depth sensors like Intel RealSense, Kinect, or stereo cameras provide crucial 3D spatial information. Depth data enables precise distance measurements, volume estimation, and 3D scene reconstruction.

```python
import open3d as o3d
import numpy as np

class DepthObjectDetector:
    def __init__(self, camera_intrinsics):
        self.intrinsics = camera_intrinsics

    def detect_3d_objects(self, rgb_image, depth_image):
        # Convert to point cloud
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image),
            o3d.geometry.Image(depth_image),
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.intrinsics
        )

        # Extract geometric features
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )

        # Separate objects from ground plane
        object_points = pcd.select_by_index(inliers, invert=True)

        # Cluster objects using DBSCAN
        labels = np.array(object_points.cluster_dbscan(
            eps=0.02,
            min_points=10
        ))

        return self.extract_object_properties(object_points, labels)

    def extract_object_properties(self, point_cloud, labels):
        objects = []
        for label in set(labels):
            if label == -1:
                continue  # Skip noise points

            cluster = point_cloud.select_by_index(np.where(labels == label)[0])
            center = cluster.get_center()
            dimensions = cluster.get_max_bound() - cluster.get_min_bound()

            objects.append({
                'center': center,
                'dimensions': dimensions,
                'points': len(cluster.points)
            })

        return objects
```

### Multi-Modal Fusion

Combining RGB and depth information creates a more robust perception system. Multi-modal fusion techniques include early fusion (combining raw data), late fusion (combining decisions), and intermediate fusion (at feature level).

```python
class MultiModalFusion:
    def __init__(self):
        self.rgb_detector = RGBObjectDetector('yolo_weights.pt')
        self.depth_detector = DepthObjectDetector(self.get_camera_intrinsics())

    def fused_detection(self, rgb_image, depth_image):
        # Get detections from both modalities
        rgb_boxes, rgb_scores, rgb_classes = self.rgb_detector.detect_objects(rgb_image)
        depth_objects = self.depth_detector.detect_3d_objects(rgb_image, depth_image)

        # Associate RGB and depth detections
        fused_objects = []
        for i, (box, score, cls) in enumerate(zip(rgb_boxes, rgb_scores, rgb_classes)):
            box_center_2d = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

            closest_depth_obj = None
            min_distance = float('inf')

            for depth_obj in depth_objects:
                # Project 3D point to 2D
                projected_2d = self.project_3d_to_2d(depth_obj['center'])
                distance = np.linalg.norm(box_center_2d - projected_2d)

                if distance < min_distance and distance < 20:  # Threshold in pixels
                    min_distance = distance
                    closest_depth_obj = depth_obj

            if closest_depth_obj:
                fused_objects.append({
                    'class': cls,
                    'confidence': score,
                    'bbox_2d': box,
                    'position_3d': closest_depth_obj['center'],
                    'dimensions_3d': closest_depth_obj['dimensions']
                })

        return fused_objects

    def project_3d_to_2d(self, point_3d):
        # Simplified projection using camera intrinsics
        # Actual implementation would use proper camera matrix
        return np.array([point_3d[0] / point_3d[2], point_3d[1] / point_3d[2]])
```

## Perception → Decision → Action Loop

### Real-Time Processing Pipeline

The perception-decision-action loop operates in real-time, continuously updating the robot's understanding of its environment and adjusting its behavior accordingly.

```python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import threading
import time

class PerceptionDecisionActionLoop:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('vision_action_loop')

        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.command_pub = rospy.Publisher('/robot/manipulation_command', PoseStamped, queue_size=10)

        # Object detection components
        self.fusion_detector = MultiModalFusion()
        self.llm_planner = LLMPlanner()

        # State variables
        self.latest_rgb = None
        self.latest_depth = None
        self.current_target = None

        # Start processing loop
        self.processing_thread = threading.Thread(target=self.main_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def image_callback(self, msg):
        self.latest_rgb = self.ros_image_to_cv2(msg)

    def depth_callback(self, msg):
        self.latest_depth = self.ros_image_to_cv2(msg)

    def main_loop(self):
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            if self.latest_rgb is not None and self.latest_depth is not None:
                # Perception: Detect objects
                detections = self.fusion_detector.fused_detection(
                    self.latest_rgb.copy(),
                    self.latest_depth.copy()
                )

                # Decision: Determine appropriate action
                action_plan = self.make_decision(detections)

                # Action: Execute manipulation
                if action_plan:
                    self.execute_action(action_plan)

            rate.sleep()

    def make_decision(self, detections):
        if not detections:
            return None

        # Determine priority target based on task
        target_object = self.select_target(detections)

        if target_object and self.is_reachable(target_object):
            # Generate manipulation plan
            manipulation_pose = self.calculate_manipulation_pose(target_object)

            return {
                'target': target_object,
                'manipulation_pose': manipulation_pose,
                'action_type': 'grasp'
            }

        return None

    def execute_action(self, action_plan):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "base_link"

        pose_msg.pose.position.x = action_plan['manipulation_pose']['x']
        pose_msg.pose.position.y = action_plan['manipulation_pose']['y']
        pose_msg.pose.position.z = action_plan['manipulation_pose']['z']

        # Set orientation for grasping
        pose_msg.pose.orientation.w = 1.0

        self.command_pub.publish(pose_msg)
```

### Decision Making Under Uncertainty

Robotic systems must handle uncertainty in perception, environment dynamics, and sensor noise. Bayesian approaches and probabilistic reasoning help manage this uncertainty.

```python
import numpy as np
from scipy.stats import multivariate_normal

class ProbabilisticDecisionMaker:
    def __init__(self):
        self.object_poses = {}  # Track estimated poses with uncertainty
        self.confidence_threshold = 0.7

    def update_belief(self, detection_list):
        for detection in detection_list:
            obj_id = detection['class']
            measured_pose = detection['position_3d']
            measurement_covariance = self.estimate_measurement_uncertainty(detection)

            if obj_id in self.object_poses:
                # Kalman filter update
                predicted_pose, predicted_cov = self.object_poses[obj_id]

                # Innovation
                innovation = measured_pose - predicted_pose
                innovation_cov = predicted_cov + measurement_covariance

                # Kalman gain
                kalman_gain = predicted_cov @ np.linalg.inv(innovation_cov)

                # Updated estimate
                updated_pose = predicted_pose + kalman_gain @ innovation
                updated_cov = predicted_cov - kalman_gain @ innovation_cov

                self.object_poses[obj_id] = (updated_pose, updated_cov)
            else:
                # Initialize belief
                self.object_poses[obj_id] = (measured_pose, measurement_covariance)

    def select_best_action(self, possible_actions):
        best_action = None
        best_expected_utility = float('-inf')

        for action in possible_actions:
            utility = self.estimate_action_utility(action)
            probability = self.estimate_success_probability(action)

            expected_utility = utility * probability

            if expected_utility > best_expected_utility:
                best_expected_utility = expected_utility
                best_action = action

        return best_action

    def estimate_action_utility(self, action):
        # Calculate utility based on task goals, safety, efficiency
        # This is a simplified example
        if action['type'] == 'grasp':
            # Higher utility for objects closer to gripper workspace
            distance_to_workspace = self.calculate_distance_to_workspace(
                action['target_pose']
            )
            utility = max(0, 1 - distance_to_workspace / 1.0)  # Normalize
            return utility

        return 0.5  # Default utility

    def estimate_success_probability(self, action):
        # Estimate probability of successful execution
        obj_id = action['target_object']

        if obj_id in self.object_poses:
            pose_mean, pose_cov = self.object_poses[obj_id]

            # Calculate confidence based on covariance determinant
            confidence = 1.0 / (np.sqrt(np.linalg.det(pose_cov)) + 1e-6)

            # Normalize confidence
            normalized_confidence = min(1.0, confidence / 10.0)

            return normalized_confidence

        return 0.3  # Low confidence if object not tracked
```

## Humanoid Manipulation Logic

### Grasping Strategy Selection

Different objects require different grasping strategies based on their shape, size, weight, and material properties.

```python
class GraspStrategySelector:
    def __init__(self):
        self.grasp_strategies = {
            'pinch': {
                'aperture_range': (0.01, 0.05),
                'weight_limit': 0.5,
                'shape_preference': ['thin', 'small']
            },
            'power': {
                'aperture_range': (0.05, 0.15),
                'weight_limit': 5.0,
                'shape_preference': ['cylindrical', 'rectangular']
            },
            'tripod': {
                'aperture_range': (0.03, 0.1),
                'weight_limit': 2.0,
                'shape_preference': ['curved', 'round']
            }
        }

    def select_grasp_strategy(self, object_properties):
        object_shape = self.classify_object_shape(object_properties)
        object_size = object_properties.get('size', 'medium')
        object_weight = object_properties.get('weight', 1.0)
        object_surface = object_properties.get('surface', 'smooth')

        candidate_strategies = []

        for strategy_name, strategy_props in self.grasp_strategies.items():
            score = 0

            # Size compatibility
            min_aperture, max_aperture = strategy_props['aperture_range']
            object_width = object_properties.get('width', 0.1)

            if min_aperture <= object_width <= max_aperture:
                score += 2
            elif abs(object_width - (min_aperture + max_aperture) / 2) < 0.05:
                score += 1

            # Weight compatibility
            if object_weight <= strategy_props['weight_limit']:
                score += 1

            # Shape preference
            if object_shape in strategy_props['shape_preference']:
                score += 1

            # Surface considerations
            if object_surface == 'rough' and strategy_name != 'pinch':
                score += 1
            elif object_surface == 'smooth' and strategy_name in ['pinch', 'tripod']:
                score += 1

            candidate_strategies.append((strategy_name, score))

        # Select highest scoring strategy
        best_strategy = max(candidate_strategies, key=lambda x: x[1])

        if best_strategy[1] > 0:
            return best_strategy[0]
        else:
            # Fallback to power grasp for unknown objects
            return 'power'

    def classify_object_shape(self, properties):
        dimensions = properties.get('dimensions', [1, 1, 1])
        length, width, height = dimensions

        # Simple shape classification
        if height < min(length, width) * 0.5:
            return 'flat'
        elif abs(length - width) < 0.02 and abs(width - height) < 0.02:
            return 'cubic'
        elif abs(length - width) < 0.02 and height > max(length, width):
            return 'cylindrical'
        elif height > length and height > width:
            return 'tall'
        else:
            return 'irregular'

class GraspController:
    def __init__(self, robot_interface):
        self.robot = robot_interface
        self.selector = GraspStrategySelector()
        self.safe_grip_force = 50.0  # Newtons

    def execute_grasp(self, target_object):
        # Plan grasp pose
        grasp_pose = self.calculate_grasp_pose(target_object)

        # Move to pre-grasp position
        self.move_to_pregrasp(grasp_pose)

        # Approach object
        self.approach_object(grasp_pose)

        # Execute grasp based on strategy
        strategy = self.selector.select_grasp_strategy(target_object)
        grip_force = self.calculate_grip_force(target_object, strategy)

        self.close_gripper(grip_force)

        # Lift object
        self.lift_object()

        return True

    def calculate_grasp_pose(self, object_data):
        # Calculate optimal grasp pose based on object properties
        object_center = object_data['position_3d']
        object_dimensions = object_data['dimensions_3d']

        # Simple approach: grasp at top center of object
        grasp_x = object_center[0]
        grasp_y = object_center[1]
        grasp_z = object_center[2] + object_dimensions[2] / 2 + 0.02  # 2cm above object

        # Orientation for stable grasp
        grasp_orientation = self.calculate_stable_orientation(object_data)

        return {
            'position': [grasp_x, grasp_y, grasp_z],
            'orientation': grasp_orientation
        }

    def calculate_grip_force(self, object_data, strategy):
        # Calculate appropriate grip force based on object weight and friction
        object_weight = object_data.get('weight', 1.0)
        surface_friction = object_data.get('friction_coefficient', 0.5)

        # Calculate minimum required force to prevent slip
        required_force = object_weight / surface_friction

        # Apply strategy-specific multiplier
        strategy_multiplier = {'pinch': 1.5, 'power': 2.0, 'tripod': 1.8}[strategy]

        # Final grip force
        grip_force = min(required_force * strategy_multiplier, self.safe_grip_force)

        return grip_force
```

### Reachability Analysis

Ensuring that the robot can physically reach target objects is crucial for successful manipulation.

```python
import numpy as np
from scipy.spatial.distance import euclidean

class ReachabilityAnalyzer:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.workspace_bounds = self.calculate_workspace_bounds()

    def calculate_workspace_bounds(self):
        # Calculate reachable workspace based on robot kinematics
        # This is a simplified representation
        joint_limits = self.robot.get_joint_limits()

        # Conservative estimate of workspace
        max_reach = sum([link.length for link in self.robot.links])

        return {
            'min_x': -max_reach,
            'max_x': max_reach,
            'min_y': -max_reach,
            'max_y': max_reach,
            'min_z': 0.1,  # Minimum height to avoid ground
            'max_z': max_reach
        }

    def is_reachable(self, target_position, posture=None):
        """
        Check if target position is reachable by the robot.

        Args:
            target_position: [x, y, z] coordinates
            posture: Optional current joint configuration

        Returns:
            bool: True if reachable, False otherwise
        """
        x, y, z = target_position

        # Check bounds
        if not (self.workspace_bounds['min_x'] <= x <= self.workspace_bounds['max_x'] and
                self.workspace_bounds['min_y'] <= y <= self.workspace_bounds['max_y'] and
                self.workspace_bounds['min_z'] <= z <= self.workspace_bounds['max_z']):
            return False

        # Check inverse kinematics solution
        ik_solution = self.robot.inverse_kinematics(target_position)

        if ik_solution is None:
            return False

        # Check joint limits
        joint_limits = self.robot.get_joint_limits()
        for joint_val, (min_lim, max_lim) in zip(ik_solution, joint_limits):
            if not (min_lim <= joint_val <= max_lim):
                return False

        return True

    def find_alternative_approach_poses(self, target_position):
        """
        Find alternative approach poses around the target if direct approach fails.

        Args:
            target_position: [x, y, z] coordinates of target

        Returns:
            list: Alternative approach poses that are reachable
        """
        alternatives = []
        target_x, target_y, target_z = target_position

        # Generate points around the target in a sphere
        for angle in np.linspace(0, 2*np.pi, 12):
            for elevation in np.linspace(-np.pi/4, np.pi/4, 5):
                # Offset from target
                offset_distance = 0.15  # 15cm from target
                offset_x = target_x + offset_distance * np.cos(angle) * np.cos(elevation)
                offset_y = target_y + offset_distance * np.sin(angle) * np.cos(elevation)
                offset_z = target_z + offset_distance * np.sin(elevation)

                approach_pose = [offset_x, offset_y, offset_z]

                if self.is_reachable(approach_pose):
                    # Calculate approach direction for proper orientation
                    approach_direction = np.array([target_x - offset_x,
                                                 target_y - offset_y,
                                                 target_z - offset_z])
                    approach_direction = approach_direction / np.linalg.norm(approach_direction)

                    alternatives.append({
                        'position': approach_pose,
                        'direction': approach_direction
                    })

        return alternatives

    def calculate_manipulation_trajectory(self, start_pose, target_pose):
        """
        Calculate smooth trajectory from start to target pose.

        Args:
            start_pose: Starting position [x, y, z]
            target_pose: Target position [x, y, z]

        Returns:
            list: Waypoints for smooth trajectory
        """
        num_waypoints = 20
        trajectory = []

        start = np.array(start_pose)
        target = np.array(target_pose)

        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            waypoint = start + t * (target - start)

            trajectory.append(waypoint.tolist())

        return trajectory

class SafetyChecker:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.collision_detector = self.initialize_collision_detector()

    def initialize_collision_detector(self):
        # Initialize collision detection system
        # This would typically use libraries like Bullet or FCL
        return CollisionDetector()

    def check_safety_constraints(self, target_pose):
        """
        Check if moving to target pose violates safety constraints.

        Args:
            target_pose: Target position and orientation

        Returns:
            dict: Safety assessment with constraint violations
        """
        safety_check = {
            'collision_free': True,
            'joint_limits_respected': True,
            'velocity_limits_respected': True,
            'torque_limits_respected': True,
            'violations': []
        }

        # Check for collisions along trajectory
        current_pose = self.robot.get_current_end_effector_pose()
        trajectory = self.interpolate_trajectory(current_pose, target_pose)

        for waypoint in trajectory:
            if self.collision_detector.check_collision_at_pose(waypoint):
                safety_check['collision_free'] = False
                safety_check['violations'].append(f"Collision at pose: {waypoint}")

        # Check joint limits
        ik_solution = self.robot.inverse_kinematics(target_pose)
        if ik_solution is not None:
            joint_limits = self.robot.get_joint_limits()
            for joint_val, (min_lim, max_lim) in zip(ik_solution, joint_limits):
                if joint_val < min_lim or joint_val > max_lim:
                    safety_check['joint_limits_respected'] = False
                    safety_check['violations'].append(f"Joint limit violation: {joint_val}")

        return safety_check

    def interpolate_trajectory(self, start_pose, end_pose, steps=10):
        """Interpolate trajectory between start and end poses."""
        trajectory = []

        for i in range(steps + 1):
            t = i / steps
            waypoint = {}

            # Interpolate position
            start_pos = np.array(start_pose['position'])
            end_pos = np.array(end_pose['position'])
            pos_interp = start_pos + t * (end_pos - start_pos)
            waypoint['position'] = pos_interp.tolist()

            # Interpolate orientation
            start_orient = start_pose['orientation']
            end_orient = end_pose['orientation']
            orient_interp = self.slerp_quaternions(start_orient, end_orient, t)
            waypoint['orientation'] = orient_interp

            trajectory.append(waypoint)

        return trajectory

    def slerp_quaternions(self, q1, q2, t):
        """Spherical linear interpolation of quaternions."""
        # Simplified implementation
        q1 = np.array(q1) if isinstance(q1, list) else q1
        q2 = np.array(q2) if isinstance(q2, list) else q2

        dot_product = np.dot(q1, q2)

        if dot_product < 0.0:
            q2 = -q2
            dot_product = -dot_product

        DOT_THRESHOLD = 0.9995
        if dot_product > DOT_THRESHOLD:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(dot_product)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return (s0 * q1 + s1 * q2) / np.linalg.norm(s0 * q1 + s1 * q2)
```

## ROS 2 Integration

### ROS 2 Nodes and Topics

ROS 2 provides the communication infrastructure for coordinating vision and action components.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from builtin_interfaces.msg import Duration
import cv2
from cv_bridge import CvBridge

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # Publishers
        self.detection_pub = self.create_publisher(String, '/object_detections', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/scene_pointcloud', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        self.detector = MultiModalFusion()  # Our earlier detector

        # Timer for periodic processing
        self.timer = self.create_timer(0.1, self.process_vision_data)

    def image_callback(self, msg):
        """Process incoming image data."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def process_vision_data(self):
        """Process vision data and publish detections."""
        if hasattr(self, 'latest_image'):
            try:
                # Process image to detect objects
                # For simplicity, assuming we have depth info
                detections = self.detector.detect_objects(self.latest_image)

                # Publish detections
                detection_msg = String()
                detection_msg.data = str(detections)  # Serialize appropriately
                self.detection_pub.publish(detection_msg)

            except Exception as e:
                self.get_logger().error(f'Error processing vision data: {e}')

class ActionNode(Node):
    def __init__(self):
        super().__init__('action_node')

        # Publishers
        self.manipulation_pub = self.create_publisher(PoseStamped, '/manipulation_target', 10)
        self.status_pub = self.create_publisher(String, '/action_status', 10)

        # Subscribers
        self.detection_sub = self.create_subscription(
            String,
            '/object_detections',
            self.detection_callback,
            10
        )

        self.planner = LLMPlanner()  # From previous modules
        self.manipulator = GraspController(robot_interface=None)

        # Action server for complex manipulation tasks
        from rclpy.action import ActionServer
        from manipulation_msgs.action import GraspObject

        self._action_server = ActionServer(
            self,
            GraspObject,
            'grasp_object',
            self.execute_grasp_object
        )

    def detection_callback(self, msg):
        """Process object detections and plan actions."""
        try:
            detections = eval(msg.data)  # Deserialize safely in production

            if detections:
                # Plan manipulation based on detections
                action_plan = self.planner.plan_manipulation(detections)

                if action_plan:
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = 'map'

                    pose_msg.pose.position.x = action_plan['target_position'][0]
                    pose_msg.pose.position.y = action_plan['target_position'][1]
                    pose_msg.pose.position.z = action_plan['target_position'][2]

                    self.manipulation_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing detection: {e}')

    def execute_grasp_object(self, goal_handle):
        """Execute grasp object action."""
        self.get_logger().info('Executing grasp object action...')

        object_id = goal_handle.request.object_id

        # Implement grasping logic
        success = self.manipulator.grasp_object_by_id(object_id)

        result = GraspObject.Result()
        result.success = success

        if success:
            goal_handle.succeed()
            return result
        else:
            goal_handle.abort()
            return result

def main(args=None):
    rclpy.init(args=args)

    vision_node = VisionNode()
    action_node = ActionNode()

    try:
        rclpy.spin(vision_node)
        rclpy.spin(action_node)
    except KeyboardInterrupt:
        pass
    finally:
        vision_node.destroy_node()
        action_node.destroy_node()
        rclpy.shutdown()
```

### Message Definitions and Service Calls

Custom message types define the interfaces between vision and action components.

```python
# Custom message definition (in msg/ObjectDetection.msg format)
"""
Header header
string object_class
float64 confidence
float64[] bbox  # [x_min, y_min, x_max, y_max]
geometry_msgs/Point32[] points  # 3D points associated with object
"""

# Service definition (in srv/FindObject.srv format)
"""
string object_name
---
bool found
geometry_msgs/PoseStamped location
float64 confidence
"""
```

## Diagrams and Architecture

### Vision-Action Integration Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RGB Camera    │    │  Depth Sensor    │    │  IMU/Gyro       │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          │         ┌────────────▼─────────┐             │
          └────────►│   Data Fusion        │◄────────────┘
                    │   Engine             │
                    └─────────┬────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │    Object Detection     │
                    │    and Classification   │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │     3D Reconstruction   │
                    │    and Pose Estimation  │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │     LLM Cognitive       │
                    │      Reasoning          │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Manipulation Planning │
                    │    and Trajectory Gen   │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Robot Motion Control  │
                    │     and Execution       │
                    └─────────────────────────┘
```

### Perception-Decision-Action Loop Timing

```
Time:     T0 ────── T1 ────── T2 ────── T3 ────── T4 ────── T5
          │        │        │        │        │        │
Vision:   [Capture][Process][Detect  ][Update ][Ready ][Capture]
Decision:         [Analyze ][Plan   ][Check  ][Execute][Analyze]
Action:           [Ready   ][Ready   ][Move   ][Wait   ][Ready   ]

Where:
- T0-T1: Image capture and initial processing
- T1-T2: Object detection and scene understanding
- T2-T3: Decision making and action planning
- T3-T4: Motion execution and monitoring
- T4-T5: Ready for next cycle
```

## Practical Exercises

### Exercise 1: Implement a Simple Grasp Planner
Create a basic grasp planner that takes object dimensions and generates appropriate grasp poses.

**Steps:**
1. Create a function that calculates grasp positions based on object dimensions
2. Implement orientation selection for stable grasping
3. Add safety margins and collision checking
4. Test with different object shapes and sizes

### Exercise 2: Multi-Modal Object Tracking
Develop a system that combines RGB and depth data for robust object tracking.

**Steps:**
1. Subscribe to both RGB and depth camera topics
2. Implement 2D-3D correspondence matching
3. Track objects across frames using Kalman filtering
4. Visualize tracked objects with bounding boxes and 3D positions

### Exercise 3: ROS 2 Vision-Action Bridge
Build a ROS 2 node that connects perception and action layers.

**Steps:**
1. Create a node with image subscribers and pose publishers
2. Implement object detection callbacks
3. Integrate with existing manipulation stack
4. Add action servers for high-level commands

## Summary

Vision-action integration is the cornerstone of intelligent robotic manipulation. By combining sophisticated perception algorithms with robust decision-making and safe action execution, humanoid robots can operate effectively in unstructured environments. The key components include multi-modal sensing, real-time processing, probabilistic reasoning, and safe execution within the ROS 2 framework.