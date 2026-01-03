---
sidebar_position: 5
---

# VSLAM Implementation for Humanoid Robots

## Overview

Visual Simultaneous Localization and Mapping (VSLAM) is crucial for humanoid robots operating in unknown environments. This section covers the implementation of hardware-accelerated VSLAM using NVIDIA Isaac Sim and Isaac ROS, focusing on the unique challenges of bipedal locomotion and dynamic environments.

VSLAM for humanoid robots must handle the challenges of walking motion, changing viewpoints, and the need for real-time performance while maintaining accuracy for safe navigation.

## VSLAM Architecture for Humanoids

### 1. System Architecture Overview

The VSLAM system for humanoid robots consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────────┐
│                    HUMANOID VSLAM SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   SENSORS   │    │  VSLAM CORE │    │   MAPPING   │     │
│  │             │───▶│             │───▶│             │     │
│  │ • Stereo    │    │ • Feature   │    │ • Global    │     │
│  │   Cameras   │    │   Extract.  │    │   Map       │     │
│  │ • IMU       │    │ • Tracking  │    │ • Local     │     │
│  │ • LIDAR     │    │ • Loop Clos.│    │   Map       │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 2. Humanoid-Specific Considerations

Humanoid robots present unique challenges for VSLAM:

- **Dynamic Motion**: Walking motion creates complex camera trajectories
- **Height Variation**: Head height changes during walking affect viewpoint
- **Balance Constraints**: VSLAM must not compromise robot stability
- **Real-time Requirements**: Processing must keep pace with locomotion
- **Multi-floor Navigation**: 3D mapping for stair navigation

## Isaac Sim VSLAM Environment Setup

### 1. Creating VSLAM Test Environments

Setting up environments specifically for VSLAM testing:

```python
# vslam_environment.py - Create environments for VSLAM testing
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, Sdf, UsdGeom, UsdLux
import numpy as np

def create_vslam_test_environment(world: World):
    """
    Create a VSLAM test environment with features suitable for localization
    """
    stage = omni.usd.get_context().get_stage()

    # Create multi-level environment with stairs
    create_multi_level_environment(stage)

    # Add distinctive visual features for tracking
    add_vslam_features(stage)

    # Configure lighting for consistent feature detection
    configure_tracking_lighting(stage)

    print("VSLAM test environment created")

def create_multi_level_environment(stage):
    """
    Create a multi-level environment with stairs for 3D mapping
    """
    # Ground floor
    ground_floor = UsdGeom.Cube.Define(stage, Sdf.Path("/World/GroundFloor"))
    ground_floor.CreateSizeAttr(20.0)
    xform = UsdGeom.Xformable(ground_floor.GetPrim())
    xform.AddTranslateOp().Set((0, 0, -0.05))
    xform.AddScaleOp().Set((10.0, 8.0, 0.1))

    # First floor platform
    first_floor = UsdGeom.Cube.Define(stage, Sdf.Path("/World/FirstFloor"))
    first_floor.CreateSizeAttr(6.0)
    xform = UsdGeom.Xformable(first_floor.GetPrim())
    xform.AddTranslateOp().Set((0, 0, 1.0))  # 1 meter high
    xform.AddScaleOp().Set((3.0, 2.5, 0.1))

    # Stairs connecting floors
    create_stairs(stage)

    # Walls for structure
    create_walls(stage)

def create_stairs(stage):
    """
    Create stairs for multi-floor navigation
    """
    num_steps = 8
    step_height = 0.15  # 15cm per step (human scale)
    step_depth = 0.3   # 30cm depth
    step_width = 1.2   # 1.2m width

    for i in range(num_steps):
        step_path = Sdf.Path(f"/World/Stairs/Step_{i}")
        step = UsdGeom.Cube.Define(stage, step_path)
        step.CreateSizeAttr(1.0)

        xform = UsdGeom.Xformable(step.GetPrim())
        xform.AddTranslateOp().Set((
            -1.0 + i * step_depth * 0.5,  # Gradually move forward
            0,
            i * step_height + step_height / 2
        ))
        xform.AddScaleOp().Set((step_depth, step_width, step_height))

def create_walls(stage):
    """
    Create walls with distinctive features for VSLAM
    """
    wall_configs = [
        # Main room walls
        {"position": (5, 0, 1.5), "scale": (0.1, 8, 3), "name": "RightWall"},
        {"position": (-5, 0, 1.5), "scale": (0.1, 8, 3), "name": "LeftWall"},
        {"position": (0, 4, 1.5), "scale": (10, 0.1, 3), "name": "FrontWall"},
        {"position": (0, -4, 1.5), "scale": (10, 0.1, 3), "name": "BackWall"},
        # Interior walls with doors
        {"position": (0, -1, 1.5), "scale": (3, 0.1, 3), "name": "InteriorWall1"},
    ]

    for config in wall_configs:
        wall_path = Sdf.Path(f"/World/Walls/{config['name']}")
        wall = UsdGeom.Cube.Define(stage, wall_path)
        wall.CreateSizeAttr(1.0)

        xform = UsdGeom.Xformable(wall.GetPrim())
        xform.AddTranslateOp().Set(config["position"])
        xform.AddScaleOp().Set(config["scale"])

def add_vslam_features(stage):
    """
    Add distinctive visual features for robust tracking
    """
    # Add textured patterns on walls
    feature_configs = [
        # Wall patterns
        {"position": (4, 2, 1.5), "type": "pattern", "name": "Pattern1"},
        {"position": (-4, -2, 1.5), "type": "pattern", "name": "Pattern2"},
        {"position": (0, 3, 1.5), "type": "pattern", "name": "Pattern3"},
        # Objects with distinctive features
        {"position": (2, -1, 0.5), "type": "object", "name": "Bookshelf"},
        {"position": (-2, 1, 0.4), "type": "object", "name": "Plant"},
    ]

    for config in feature_configs:
        if config["type"] == "pattern":
            # Create textured pattern
            pattern_path = Sdf.Path(f"/World/Features/{config['name']}")
            pattern = UsdGeom.Cube.Define(stage, pattern_path)
            pattern.CreateSizeAttr(0.5)

            xform = UsdGeom.Xformable(pattern.GetPrim())
            xform.AddTranslateOp().Set(config["position"])
            xform.AddScaleOp().Set((0.8, 0.01, 1.0))  # Flat pattern

def configure_tracking_lighting(stage):
    """
    Configure lighting for consistent feature tracking
    """
    # Main overhead lighting
    main_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/MainTrackingLight"))
    main_light.CreateIntensityAttr(2500)
    main_light.CreateColorAttr(Gf.Vec3f(0.98, 0.98, 0.95))  # Daylight white
    main_light.AddRotateXYZOp().Set((70, 0, 0))

    # Fill lights to reduce shadows
    fill_light1 = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight1"))
    fill_light1.CreateIntensityAttr(500)
    fill_light1.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))
    fill_light1.AddTranslateOp().Set((3, 2, 2))

    fill_light2 = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/FillLight2"))
    fill_light2.CreateIntensityAttr(500)
    fill_light2.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 1.0))
    fill_light2.AddTranslateOp().Set((-3, -2, 2))

# Example usage
def setup_vslam_world():
    """
    Complete setup for VSLAM testing world
    """
    world = World(stage_units_in_meters=1.0)
    configure_humanoid_physics()  # From previous module
    create_vslam_test_environment(world)
    return world
```

### 2. Stereo Camera Configuration for VSLAM

Setting up stereo cameras optimized for VSLAM:

```python
# stereo_camera_setup.py - Configure stereo cameras for VSLAM
import omni
from omni.isaac.sensor import Camera
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, Sdf, UsdGeom
import numpy as np

def setup_vslam_stereo_cameras(world: World):
    """
    Set up stereo cameras optimized for VSLAM applications
    """
    # Create stereo camera rig
    create_camera_rig()

    # Configure left camera
    left_camera = Camera(
        prim_path="/World/HumanoidHead/LeftCamera",
        frequency=30,  # 30 Hz for VSLAM
        resolution=(640, 480),  # Good balance of speed/quality
        position=(0.0, -0.05, 0.0),  # 10cm baseline
        orientation=(0, 0, 0, 1),  # Default orientation
        fov=60,  # 60 degree field of view
        clipping_range=(0.1, 10.0)  # 10m max range
    )

    # Configure right camera
    right_camera = Camera(
        prim_path="/World/HumanoidHead/RightCamera",
        frequency=30,
        resolution=(640, 480),
        position=(0.0, 0.05, 0.0),  # 10cm baseline
        orientation=(0, 0, 0, 1),
        fov=60,
        clipping_range=(0.1, 10.0)
    )

    # Add cameras to world
    world.scene.add(left_camera)
    world.scene.add(right_camera)

    print("VSLAM stereo cameras configured")

def create_camera_rig():
    """
    Create a physical camera rig to maintain calibration
    """
    stage = omni.usd.get_context().get_stage()

    # Create camera mount
    mount_path = Sdf.Path("/World/HumanoidHead/CameraMount")
    mount = UsdGeom.Cube.Define(stage, mount_path)
    mount.CreateSizeAttr(0.1)  # 10cm cube mount

    xform = UsdGeom.Xformable(mount.GetPrim())
    xform.AddTranslateOp().Set((0.1, 0, 0))  # Position on head
    xform.AddScaleOp().Set((0.05, 0.1, 0.05))  # Small mount

def get_stereo_calibration():
    """
    Return stereo calibration parameters for VSLAM
    """
    # Camera intrinsic parameters
    fx = 320.0  # Focal length in pixels
    fy = 320.0
    cx = 320.0  # Principal point
    cy = 240.0

    # Camera matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # Distortion coefficients (assuming calibrated)
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Stereo baseline (distance between cameras)
    baseline = 0.1  # 10cm

    # Projection matrix for stereo
    P = np.array([
        [fx, 0, cx, -fx * baseline],
        [0, fy, cy, 0],
        [0, 0, 1, 0]
    ])

    return {
        'K': K,
        'D': D,
        'baseline': baseline,
        'P': P,
        'resolution': (640, 480),
        'fov': 60
    }

class StereoVSLAMInterface:
    """
    Interface for stereo VSLAM with Isaac Sim cameras
    """
    def __init__(self, world: World):
        self.world = world
        self.calibration = get_stereo_calibration()
        self.left_image = None
        self.right_image = None
        self.frame_timestamp = None

    def capture_stereo_pair(self):
        """
        Capture synchronized stereo image pair
        """
        # In Isaac Sim, this would access the camera data
        # For simulation, we'll create synchronized captures
        current_time = self.world.current_time
        self.frame_timestamp = current_time

        # This is where we'd get the actual camera data
        # left_img = self.left_camera.get_rgb()
        # right_img = self.right_camera.get_rgb()

        # For this example, return dummy data
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return dummy_img, dummy_img  # Return left, right images

    def get_camera_pose(self):
        """
        Get current camera pose for VSLAM
        """
        # This would return the current pose of the stereo rig
        # In a real implementation, this would interface with the physics engine
        return np.eye(4)  # Identity for now
```

## Isaac ROS VSLAM Integration

### 1. Isaac ROS Visual Inertial Odometry Setup

Implementing hardware-accelerated VIO using Isaac ROS:

```python
#!/usr/bin/env python3
# isaac_ros_vio_integration.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

class IsaacROSVisualInertialOdometry(Node):
    """
    Isaac ROS Visual Inertial Odometry for humanoid robots
    """
    def __init__(self):
        super().__init__('isaac_ros_vio')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # TF broadcaster for poses
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Create subscribers for stereo and IMU
        self.left_image_sub = self.create_subscription(
            Image,
            '/stereo/left/image_rect',
            self.left_image_callback,
            5
        )

        self.right_image_sub = self.create_subscription(
            Image,
            '/stereo/right/image_rect',
            self.right_image_callback,
            5
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.left_camera_info_sub = self.create_subscription(
            CameraInfo,
            '/stereo/left/camera_info',
            self.left_camera_info_callback,
            10
        )

        self.right_camera_info_sub = self.create_subscription(
            CameraInfo,
            '/stereo/right/camera_info',
            self.right_camera_info_callback,
            10
        )

        # Publishers
        self.odom_pub = self.create_publisher(
            Odometry,
            '/visual_inertial_odom',
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/vio/pose',
            10
        )

        # VIO state
        self.left_image = None
        self.right_image = None
        self.imu_data = None
        self.camera_info = None
        self.initialized = False

        # Pose tracking
        self.current_pose = np.eye(4)
        self.previous_pose = np.eye(4)

        # IMU bias tracking
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now()

        self.get_logger().info('Isaac ROS Visual Inertial Odometry initialized')

    def left_image_callback(self, msg):
        """Process left camera image"""
        try:
            self.left_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.process_vio()
        except Exception as e:
            self.get_logger().error(f'Error processing left image: {e}')

    def right_image_callback(self, msg):
        """Process right camera image"""
        try:
            self.right_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.process_vio()
        except Exception as e:
            self.get_logger().error(f'Error processing right image: {e}')

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg

    def left_camera_info_callback(self, msg):
        """Store left camera info"""
        self.camera_info = msg

    def right_camera_info_callback(self, msg):
        """Store right camera info"""
        # We'll use left camera info as primary

    def process_vio(self):
        """
        Process VIO using Isaac ROS acceleration
        """
        if (self.left_image is not None and
            self.right_image is not None and
            self.imu_data is not None and
            self.camera_info is not None):

            # In Isaac ROS, this would call the actual VIO algorithm
            # which uses GPU acceleration for feature tracking and IMU integration
            pose_increment = self.accelerated_vio_processing()

            if pose_increment is not None:
                # Update pose with the increment
                self.previous_pose = self.current_pose.copy()
                self.current_pose = self.current_pose @ pose_increment

                # Publish results
                self.publish_odometry()
                self.publish_pose()
                self.broadcast_transform()

                # Update performance metrics
                self.frame_count += 1
                current_time = self.get_clock().now()
                elapsed = (current_time - self.start_time).nanoseconds / 1e9
                if elapsed > 0 and self.frame_count % 30 == 0:
                    fps = self.frame_count / elapsed
                    self.get_logger().info(f'VIO processing at {fps:.2f} FPS')

    def accelerated_vio_processing(self):
        """
        Hardware-accelerated VIO processing using Isaac ROS
        """
        # This is a placeholder - in real Isaac ROS implementation:
        # 1. GPU-accelerated feature extraction
        # 2. Real-time stereo matching
        # 3. IMU integration with visual features
        # 4. Loop closure detection

        # For simulation, create a small pose increment based on IMU
        if self.imu_data:
            # Extract angular velocity and linear acceleration
            gyro = np.array([
                self.imu_data.angular_velocity.x,
                self.imu_data.angular_velocity.y,
                self.imu_data.angular_velocity.z
            ])

            accel = np.array([
                self.imu_data.linear_acceleration.x,
                self.imu_data.linear_acceleration.y,
                self.imu_data.linear_acceleration.z
            ])

            # Remove bias (simplified)
            gyro_corrected = gyro - self.gyro_bias
            accel_corrected = accel - self.accel_bias

            # Integrate to get pose change (simplified)
            dt = 1.0 / 30.0  # 30 FPS
            angle_change = gyro_corrected * dt
            linear_change = accel_corrected * dt * dt * 0.5  # s = 0.5*a*t^2

            # Create transformation matrix from changes
            pose_inc = self.create_pose_increment(angle_change, linear_change)

            # Update bias estimates (simplified)
            self.update_bias_estimates(gyro, accel)

            return pose_inc

        # Default: small forward movement
        pose_inc = np.eye(4)
        pose_inc[0, 3] = 0.01  # Move forward 1cm
        return pose_inc

    def create_pose_increment(self, angle_change, linear_change):
        """
        Create pose increment matrix from angle and linear changes
        """
        pose_inc = np.eye(4)

        # Convert angle changes to rotation matrix (small angle approximation)
        rx, ry, rz = angle_change
        rot_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        rot_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        rot_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        rotation = rot_z @ rot_y @ rot_x
        pose_inc[:3, :3] = rotation

        # Add linear translation
        pose_inc[:3, 3] = linear_change

        return pose_inc

    def update_bias_estimates(self, gyro, accel):
        """
        Update bias estimates using simple averaging
        """
        # Simple bias estimation (in practice, more sophisticated methods are used)
        alpha = 0.01  # Learning rate
        self.gyro_bias = self.gyro_bias * (1 - alpha) + gyro * alpha
        self.accel_bias = self.accel_bias * (1 - alpha) + accel * alpha

    def publish_odometry(self):
        """
        Publish odometry message
        """
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set pose
        pos = self.current_pose[:3, 3]
        odom_msg.pose.pose.position.x = float(pos[0])
        odom_msg.pose.pose.position.y = float(pos[1])
        odom_msg.pose.pose.position.z = float(pos[2])

        # Convert rotation matrix to quaternion
        quat = self.rotation_matrix_to_quaternion(self.current_pose[:3, :3])
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Set velocity (approximate from pose difference)
        dt = 1.0 / 30.0  # 30 FPS
        pos_diff = pos - self.previous_pose[:3, 3]
        velocity = pos_diff / dt

        odom_msg.twist.twist.linear.x = float(velocity[0])
        odom_msg.twist.twist.linear.y = float(velocity[1])
        odom_msg.twist.twist.linear.z = float(velocity[2])

        self.odom_pub.publish(odom_msg)

    def publish_pose(self):
        """
        Publish pose message
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'

        pos = self.current_pose[:3, 3]
        pose_msg.pose.position.x = float(pos[0])
        pose_msg.pose.position.y = float(pos[1])
        pose_msg.pose.position.z = float(pos[2])

        quat = self.rotation_matrix_to_quaternion(self.current_pose[:3, :3])
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

    def broadcast_transform(self):
        """
        Broadcast TF transform
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        pos = self.current_pose[:3, 3]
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])

        quat = self.rotation_matrix_to_quaternion(self.current_pose[:3, :3])
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        """
        Convert rotation matrix to quaternion
        """
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                z = 0.25 * s

        return [x, y, z, w]

def main(args=None):
    rclpy.init(args=args)
    vio_node = IsaacROSVisualInertialOdometry()

    try:
        rclpy.spin(vio_node)
    except KeyboardInterrupt:
        vio_node.get_logger().info('Shutting down Isaac ROS VIO')
    finally:
        vio_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Feature Tracking and Mapping

Implementing feature-based mapping for humanoid VSLAM:

```python
#!/usr/bin/env python3
# feature_tracking_mapping.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque

class IsaacROSFeatureTracker(Node):
    """
    Feature tracker for Isaac ROS VSLAM system
    """
    def __init__(self):
        super().__init__('isaac_ros_feature_tracker')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/stereo/left/image_rect',
            self.image_callback,
            5
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/stereo/left/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers
        self.feature_pub = self.create_publisher(
            MarkerArray,
            '/vslam/features',
            10
        )

        self.map_pub = self.create_publisher(
            MarkerArray,
            '/vslam/map',
            10
        )

        # Initialize feature tracking
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.previous_features = None
        self.current_features = None
        self.feature_ids = {}
        self.feature_poses = {}  # 3D positions of features
        self.next_feature_id = 0

        # Feature tracking parameters
        self.max_features = 1000
        self.min_distance = 20
        self.quality_level = 0.01
        self.block_size = 7

        # Pose tracking
        self.current_pose = np.eye(4)
        self.frame_count = 0

        # Feature history for mapping
        self.feature_history = deque(maxlen=100)

        self.get_logger().info('Isaac ROS Feature Tracker initialized')

    def image_callback(self, msg):
        """
        Process image for feature tracking
        """
        try:
            # Convert ROS Image to OpenCV format
            gray = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

            # Track features
            self.track_features(gray, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def camera_info_callback(self, msg):
        """
        Store camera calibration information
        """
        self.camera_matrix = np.array(msg.k).reshape(3, 2)
        self.distortion_coeffs = np.array(msg.d)

    def track_features(self, gray, header):
        """
        Track features using Lucas-Kanade optical flow
        """
        if self.camera_matrix is None:
            return

        # Detect new features if we don't have enough
        if self.current_features is None or len(self.current_features) < 100:
            new_features = self.detect_features(gray)
            if new_features is not None:
                if self.current_features is None:
                    self.current_features = new_features
                else:
                    # Combine with existing features
                    self.current_features = np.vstack([self.current_features, new_features])

        # Track existing features
        if self.previous_features is not None and len(self.previous_features) > 0:
            # Use Lucas-Kanade tracker
            next_features, status, error = cv2.calcOpticalFlowPyrLK(
                self.previous_gray, gray,
                self.previous_features.reshape(-1, 1, 2),
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            # Filter good features
            good_new = next_features[status == 1]
            good_old = self.previous_features[status == 1]

            # Update current features
            self.current_features = good_new

            # Estimate motion from feature correspondences
            if len(good_new) >= 8:  # Need minimum for pose estimation
                self.estimate_camera_motion(good_old, good_new, header)

        # Store for next iteration
        self.previous_gray = gray.copy()
        self.previous_features = self.current_features.copy() if self.current_features is not None else None

        # Publish features
        self.publish_features(header)

        self.frame_count += 1

    def detect_features(self, gray):
        """
        Detect new features to track
        """
        # Use Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )

        return corners

    def estimate_camera_motion(self, old_features, new_features, header):
        """
        Estimate camera motion from feature correspondences
        """
        # In a real VSLAM system, this would involve:
        # 1. Essential matrix estimation
        # 2. Pose extraction
        # 3. Scale estimation (for monocular, needs other sensors)
        # 4. Bundle adjustment

        # For this example, we'll use a simplified approach
        if len(old_features) >= 8:
            # Calculate fundamental matrix
            fundamental_matrix, mask = cv2.findFundamentalMat(
                old_features, new_features, cv2.RANSAC, 4, 0.999
            )

            # Use camera matrix to compute essential matrix
            if self.camera_matrix is not None:
                essential_matrix = self.camera_matrix.T @ fundamental_matrix @ self.camera_matrix

                # Decompose essential matrix to get rotation and translation
                _, rotation, translation, _ = cv2.recoverPose(essential_matrix, old_features, new_features, self.camera_matrix)

                # Create pose increment
                pose_inc = np.eye(4)
                pose_inc[:3, :3] = rotation
                pose_inc[:3, 3] = translation.flatten() * 0.1  # Scale factor for simulation

                # Update current pose
                self.current_pose = self.current_pose @ pose_inc

                # Update 3D positions of tracked features
                self.update_feature_positions(old_features, new_features)

    def update_feature_positions(self, old_features, new_features):
        """
        Update 3D positions of tracked features using triangulation
        """
        if self.frame_count > 1:  # Need at least 2 poses
            for i in range(min(len(old_features), len(new_features))):
                old_pt = old_features[i].flatten()
                new_pt = new_features[i].flatten()

                # Triangulate 3D point
                point_3d = self.triangulate_point(old_pt, new_pt)

                if point_3d is not None:
                    # Store or update feature position
                    feature_key = (int(old_pt[0]), int(old_pt[1]))
                    self.feature_poses[feature_key] = point_3d

    def triangulate_point(self, old_pt, new_pt):
        """
        Triangulate 3D point from stereo correspondences
        """
        # This is a simplified version - in real implementation:
        # 1. Use known camera poses
        # 2. Apply proper triangulation algorithm
        # 3. Use stereo baseline information

        # For simulation, return a plausible 3D point
        depth = 2.0  # Assume 2m depth
        if self.camera_matrix is not None:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            x = (old_pt[0] - cx) * depth / fx
            y = (old_pt[1] - cy) * depth / fy
            z = depth

            return np.array([x, y, z])

        return None

    def publish_features(self, header):
        """
        Publish tracked features as visualization markers
        """
        if self.current_features is not None:
            marker_array = MarkerArray()

            for i, pt in enumerate(self.current_features):
                if i < 50:  # Limit visualization
                    marker = Marker()
                    marker.header = header
                    marker.ns = "features"
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    # Position (simplified - would be 3D in real system)
                    marker.pose.position.x = float(pt[0])
                    marker.pose.position.y = float(pt[1])
                    marker.pose.position.z = 1.0  # Fixed height for visualization

                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.05
                    marker.scale.y = 0.05
                    marker.scale.z = 0.05

                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0

                    marker_array.markers.append(marker)

            self.feature_pub.publish(marker_array)

    def publish_map(self):
        """
        Publish the 3D map of features
        """
        marker_array = MarkerArray()

        for i, (key, pos) in enumerate(list(self.feature_poses.items())[:100]):  # Limit visualization
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "map"
            marker.ns = "map_points"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])

            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.map_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    feature_tracker = IsaacROSFeatureTracker()

    # Create timer to periodically publish map
    feature_tracker.create_timer(1.0, feature_tracker.publish_map)

    try:
        rclpy.spin(feature_tracker)
    except KeyboardInterrupt:
        feature_tracker.get_logger().info('Shutting down feature tracker')
    finally:
        feature_tracker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Humanoid-Specific VSLAM Considerations

### 1. Walking Motion Compensation

Compensating for humanoid walking motion in VSLAM:

```python
#!/usr/bin/env python3
# walking_motion_compensation.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64
import numpy as np
from scipy import signal

class WalkingMotionCompensation(Node):
    """
    Compensate for humanoid walking motion in VSLAM
    """
    def __init__(self):
        super().__init__('walking_motion_compensation')

        # Subscribers
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

        # Publishers
        self.compensated_imu_pub = self.create_publisher(
            Imu,
            '/imu/compensated',
            10
        )

        self.walk_state_pub = self.create_publisher(
            Float64,
            '/walking_state',
            10
        )

        # Walking pattern detection
        self.imu_buffer = []
        self.max_buffer_size = 100
        self.walk_state = 0.0  # 0.0 = standing, 1.0 = walking

        # Walking parameters
        self.leg_joints = ['left_hip_pitch', 'left_knee', 'left_ankle_pitch',
                          'right_hip_pitch', 'right_knee', 'right_ankle_pitch']
        self.joint_positions = {}
        self.joint_velocities = {}

        # Butterworth filter for motion separation
        self.filter_b, self.filter_a = signal.butter(4, 0.1, btype='high', fs=100.0)

        # Walking pattern detection
        self.walking_detected = False
        self.walking_confidence = 0.0

        self.get_logger().info('Walking Motion Compensation initialized')

    def imu_callback(self, msg):
        """
        Process IMU data for walking compensation
        """
        # Add to buffer for pattern analysis
        self.imu_buffer.append({
            'linear_acceleration': np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ]),
            'angular_velocity': np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ]),
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        })

        # Keep buffer at max size
        if len(self.imu_buffer) > self.max_buffer_size:
            self.imu_buffer.pop(0)

        # Detect walking patterns
        self.detect_walking_pattern()

        # Compensate IMU data
        compensated_imu = self.compensate_imu_data(msg)

        # Publish compensated data
        self.compensated_imu_pub.publish(compensated_imu)

        # Publish walking state
        walk_state_msg = Float64()
        walk_state_msg.data = self.walk_state
        self.walk_state_pub.publish(walk_state_msg)

    def joint_state_callback(self, msg):
        """
        Process joint state data for walking analysis
        """
        for i, name in enumerate(msg.name):
            if name in self.leg_joints:
                if i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_velocities[name] = msg.velocity[i]

    def detect_walking_pattern(self):
        """
        Detect walking pattern from IMU and joint data
        """
        if len(self.imu_buffer) < 50:  # Need sufficient data
            return

        # Analyze IMU data for walking patterns
        linear_accs = np.array([d['linear_acceleration'] for d in self.imu_buffer])

        # Look for periodic patterns in vertical acceleration (due to footsteps)
        vertical_acc = linear_accs[:, 2]  # Z-axis (vertical)

        # Apply FFT to detect periodic components
        fft_result = np.fft.fft(vertical_acc)
        freqs = np.fft.fftfreq(len(vertical_acc), d=0.01)  # Assuming 100Hz IMU

        # Look for typical walking frequencies (1-3 Hz)
        walking_freq_range = (freqs > 1.0) & (freqs < 3.0)
        walking_energy = np.sum(np.abs(fft_result[walking_freq_range]))

        # Also check for joint patterns
        joint_activity = self.calculate_joint_activity()

        # Combine indicators for walking confidence
        self.walking_confidence = min(1.0, (walking_energy * 0.001 + joint_activity * 0.5))

        # Update walk state with hysteresis
        if self.walking_confidence > 0.7 and not self.walking_detected:
            self.walking_detected = True
            self.get_logger().info('Walking detected')
        elif self.walking_confidence < 0.3 and self.walking_detected:
            self.walking_detected = False
            self.get_logger().info('Walking stopped')

        self.walk_state = self.walking_confidence

    def calculate_joint_activity(self):
        """
        Calculate joint activity as indicator of walking
        """
        activity = 0.0
        for joint in self.leg_joints:
            if joint in self.joint_velocities:
                activity += abs(self.joint_velocities[joint])

        return min(1.0, activity / 10.0)  # Normalize

    def compensate_imu_data(self, original_imu):
        """
        Compensate IMU data for walking motion
        """
        compensated_imu = Imu()
        compensated_imu.header = original_imu.header

        # In a real system, this would involve:
        # 1. Estimating walking motion from joint data
        # 2. Predicting IMU readings due to walking
        # 3. Subtracting predicted walking motion

        # For this example, we'll apply a simplified compensation
        # based on detected walking state

        walking_factor = 1.0 - self.walk_state  # Reduce trust in IMU when walking

        # Copy orientation (this is what we trust most)
        compensated_imu.orientation = original_imu.orientation
        compensated_imu.orientation_covariance = original_imu.orientation_covariance

        # Scale linear acceleration based on walking confidence
        compensated_imu.linear_acceleration.x = original_imu.linear_acceleration.x * walking_factor
        compensated_imu.linear_acceleration.y = original_imu.linear_acceleration.y * walking_factor
        compensated_imu.linear_acceleration.z = original_imu.linear_acceleration.z * walking_factor

        # Increase covariance when walking (less reliable)
        walking_cov_scale = 1.0 + self.walk_state * 2.0
        compensated_imu.linear_acceleration_covariance = [
            original_imu.linear_acceleration_covariance[i] * walking_cov_scale
            for i in range(9)
        ]

        # Similar for angular velocity
        compensated_imu.angular_velocity.x = original_imu.angular_velocity.x * walking_factor
        compensated_imu.angular_velocity.y = original_imu.angular_velocity.y * walking_factor
        compensated_imu.angular_velocity.z = original_imu.angular_velocity.z * walking_factor

        return compensated_imu

def main(args=None):
    rclpy.init(args=args)
    motion_comp = WalkingMotionCompensation()

    try:
        rclpy.spin(motion_comp)
    except KeyboardInterrupt:
        motion_comp.get_logger().info('Shutting down walking motion compensation')
    finally:
        motion_comp.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization and Real-time Considerations

### 1. Multi-threaded VSLAM Architecture

Implementing a multi-threaded architecture for real-time VSLAM:

```python
#!/usr/bin/env python3
# multithreaded_vslam.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from threading import Thread, Lock
from queue import Queue
import numpy as np
import cv2
from cv_bridge import CvBridge

class MultithreadedVSLAM(Node):
    """
    Multi-threaded VSLAM implementation for real-time performance
    """
    def __init__(self):
        super().__init__('multithreaded_vslam')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Create queues for inter-thread communication
        self.image_queue = Queue(maxsize=5)
        self.imu_queue = Queue(maxsize=10)
        self.feature_queue = Queue(maxsize=10)

        # Create locks for shared data
        self.pose_lock = Lock()
        self.feature_lock = Lock()

        # Shared state
        self.current_pose = np.eye(4)
        self.features = []
        self.is_running = True

        # Create subscribers
        self.left_image_sub = self.create_subscription(
            Image,
            '/stereo/left/image_rect',
            self.image_callback,
            5
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Start processing threads
        self.feature_thread = Thread(target=self.feature_extraction_loop)
        self.tracking_thread = Thread(target=self.tracking_loop)
        self.mapping_thread = Thread(target=self.mapping_loop)

        self.feature_thread.start()
        self.tracking_thread.start()
        self.mapping_thread.start()

        self.get_logger().info('Multithreaded VSLAM initialized')

    def image_callback(self, msg):
        """
        Image callback - adds to processing queue
        """
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            # Add to queue with timestamp
            image_data = {
                'image': cv_image,
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'header': msg.header
            }
            if not self.image_queue.full():
                self.image_queue.put(image_data)
        except:
            pass  # Drop frame if queue is full

    def imu_callback(self, msg):
        """
        IMU callback - adds to processing queue
        """
        try:
            imu_data = {
                'linear_acceleration': np.array([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ]),
                'angular_velocity': np.array([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ]),
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'orientation': msg.orientation
            }
            if not self.imu_queue.full():
                self.imu_queue.put(imu_data)
        except:
            pass  # Drop if queue is full

    def feature_extraction_loop(self):
        """
        Thread for feature extraction
        """
        while self.is_running:
            try:
                # Get image from queue
                if not self.image_queue.empty():
                    image_data = self.image_queue.get(timeout=0.01)
                    cv_image = image_data['image']

                    # Extract features
                    features = self.extract_features(cv_image)

                    # Add to feature queue
                    feature_data = {
                        'features': features,
                        'timestamp': image_data['timestamp'],
                        'header': image_data['header']
                    }
                    if not self.feature_queue.full():
                        self.feature_queue.put(feature_data)
                else:
                    # Small sleep to prevent busy waiting
                    import time
                    time.sleep(0.001)
            except:
                continue

    def tracking_loop(self):
        """
        Thread for feature tracking and pose estimation
        """
        previous_features = None
        previous_image = None

        while self.is_running:
            try:
                if not self.feature_queue.empty():
                    feature_data = self.feature_queue.get(timeout=0.01)
                    current_features = feature_data['features']

                    if previous_features is not None and len(previous_features) > 0:
                        # Track features and estimate motion
                        tracked_features, status = self.track_features(
                            previous_features, current_features
                        )

                        if len(tracked_features) > 10:  # Minimum for pose estimation
                            pose_increment = self.estimate_motion(
                                previous_features[status.flatten() == 1],
                                tracked_features
                            )

                            # Update pose
                            with self.pose_lock:
                                self.current_pose = self.current_pose @ pose_increment

                    previous_features = current_features.copy()
                else:
                    # Small sleep to prevent busy waiting
                    import time
                    time.sleep(0.001)
            except:
                continue

    def mapping_loop(self):
        """
        Thread for map building and optimization
        """
        while self.is_running:
            # In a real implementation, this would:
            # 1. Build 3D map from tracked features
            # 2. Perform loop closure detection
            # 3. Optimize map using bundle adjustment
            # 4. Manage map size and memory usage

            # For this example, just sleep and occasionally log
            import time
            time.sleep(0.1)

    def extract_features(self, image):
        """
        Extract features from image (simplified)
        """
        # Use ORB features for efficiency
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(image, None)

        if keypoints is not None:
            # Convert keypoints to numpy array
            features = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            return features
        else:
            return np.empty((0, 2), dtype=np.float32)

    def track_features(self, previous_features, current_features):
        """
        Track features between frames
        """
        # This is a simplified version
        # In real implementation, use optical flow or descriptor matching
        if len(previous_features) > 0 and len(current_features) > 0:
            # For this example, return all current features with status
            status = np.ones((len(current_features), 1), dtype=np.uint8)
            return current_features, status
        else:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 1), dtype=np.uint8)

    def estimate_motion(self, old_features, new_features):
        """
        Estimate camera motion from feature correspondences
        """
        if len(old_features) >= 8:
            # Calculate transformation matrix
            transform, mask = cv2.estimateAffinePartial2D(
                old_features, new_features, method=cv2.RANSAC
            )

            if transform is not None:
                # Convert to 4x4 transformation matrix
                pose_inc = np.eye(4)
                pose_inc[:2, :2] = transform[:2, :2]
                pose_inc[:2, 3] = transform[:2, 2]
                return pose_inc

        # Default: small identity transformation
        return np.eye(4)

    def destroy_node(self):
        """
        Clean shutdown of threads
        """
        self.is_running = False
        if self.feature_thread.is_alive():
            self.feature_thread.join()
        if self.tracking_thread.is_alive():
            self.tracking_thread.join()
        if self.mapping_thread.is_alive():
            self.mapping_thread.join()

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    vslam_node = MultithreadedVSLAM()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        vslam_node.get_logger().info('Shutting down multithreaded VSLAM')
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files for Complete VSLAM System

### 1. Complete VSLAM Launch File

Creating launch files for the complete VSLAM system:

```python
# launch/humanoid_vslam.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    camera_namespace = LaunchConfiguration('camera_namespace')
    robot_namespace = LaunchConfiguration('robot_namespace')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    declare_camera_namespace = DeclareLaunchArgument(
        'camera_namespace',
        default_value='/stereo',
        description='Namespace for stereo camera topics'
    )

    declare_robot_namespace = DeclareLaunchArgument(
        'robot_namespace',
        default_value='',
        description='Robot namespace'
    )

    # Isaac ROS Visual Inertial Odometry
    vio_node = Node(
        package='isaac_ros_visual_inertial_odometry',
        executable='visual_inertial_odometry_node',
        name='visual_inertial_odometry',
        parameters=[{
            'use_sim_time': use_sim_time,
            'enable_debug_mode': False,
            'publish_tf': True,
            'world_frame': 'odom',
            'base_frame': 'base_link'
        }],
        remappings=[
            ('left/image_rect', [camera_namespace, '/left/image_rect']),
            ('left/camera_info', [camera_namespace, '/left/camera_info']),
            ('right/image_rect', [camera_namespace, '/right/image_rect']),
            ('right/camera_info', [camera_namespace, '/right/camera_info']),
            ('imu', '/imu/data'),
            ('visual_odometry', '/visual_odometry'),
        ]
    )

    # Isaac ROS Apriltag detector (for loop closure)
    apriltag_node = Node(
        package='isaac_ros_apriltag',
        executable='apriltag_node',
        name='apriltag',
        parameters=[{
            'use_sim_time': use_sim_time,
            'family': 'tag36h11',
            'max_tags': 64,
            'tag36h11_size': 0.16
        }],
        remappings=[
            ('image', [camera_namespace, '/left/image_rect']),
            ('camera_info', [camera_namespace, '/left/camera_info']),
            ('detections', '/apriltag/detections'),
        ]
    )

    # Walking motion compensation
    walk_compensation = Node(
        package='humanoid_vslam',
        executable='walking_motion_compensation',
        name='walking_motion_compensation',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/imu/data', '/imu/data'),
            ('/joint_states', '/joint_states'),
            ('/imu/compensated', '/imu/compensated'),
        ]
    )

    # Feature tracker
    feature_tracker = Node(
        package='humanoid_vslam',
        executable='feature_tracker',
        name='feature_tracker',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/stereo/left/image_rect', [camera_namespace, '/left/image_rect']),
            ('/stereo/left/camera_info', [camera_namespace, '/left/camera_info']),
            ('/vslam/features', '/vslam/features'),
            ('/vslam/map', '/vslam/map'),
        ]
    )

    # Multi-threaded VSLAM processor
    multithreaded_vslam = Node(
        package='humanoid_vslam',
        executable='multithreaded_vslam',
        name='multithreaded_vslam',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/stereo/left/image_rect', [camera_namespace, '/left/image_rect']),
            ('/stereo/right/image_rect', [camera_namespace, '/right/image_rect']),
            ('/imu/data', '/imu/compensated'),  # Use compensated IMU
        ]
    )

    # Robot state publisher for visualization
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
        }]
    )

    # Static transform publisher for camera to base link
    camera_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=['0.1', '0', '0.1', '0', '0', '0', 'base_link', 'camera_link']
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_camera_namespace,
        declare_robot_namespace,

        robot_state_publisher,
        camera_tf_publisher,

        # Start basic nodes first
        vio_node,
        apriltag_node,

        # Then start humanoid-specific nodes
        walk_compensation,
        feature_tracker,
        multithreaded_vslam,
    ])
```

## Next Steps

With the VSLAM implementation complete, you're ready to move on to configuring Nav2 for bipedal navigation. The next section will cover adapting the Nav2 stack for humanoid robots, including footstep planning, balance-aware path planning, and multi-floor navigation specific to bipedal locomotion.