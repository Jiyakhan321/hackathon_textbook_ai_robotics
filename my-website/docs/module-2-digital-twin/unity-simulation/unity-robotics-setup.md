---
sidebar_position: 1
---

# Unity Robotics Setup and Configuration

## Overview

Unity provides a powerful platform for creating sophisticated simulation environments for humanoid robots, particularly when advanced graphics, physics, or AI training are required. The Unity Robotics Hub provides the bridge between Unity and ROS 2, enabling your humanoid robot to interact with the ROS 2 ecosystem. This section covers setting up Unity for robotics applications.

## Prerequisites and Requirements

### System Requirements
- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04, or macOS 10.14+
- **Unity Version**: Unity 2022.3 LTS (recommended for stability)
- **Hardware**:
  - GPU with DirectX 11 or OpenGL 4.3 support
  - 8GB+ RAM recommended for complex humanoid simulations
  - Multi-core processor for physics simulation

### Required Software
- Unity Hub (for version management)
- Unity 2022.3 LTS with Linux Build Support (if using ROS 2 on Linux)
- ROS 2 Humble Hawksbill
- Unity Robotics Hub package
- Unity ML-Agents (for AI training)

## Installing Unity Robotics Hub

### 1. Install Unity Hub and Unity Editor

1. Download and install Unity Hub from [Unity's website](https://unity.com/)
2. Through Unity Hub, install Unity 2022.3 LTS
3. When installing, make sure to include:
   - Linux Build Support (if using ROS 2 on Linux)
   - Windows Build Support (for Windows users)
   - macOS Build Support (for macOS users)

### 2. Create a New Unity Project

1. Open Unity Hub
2. Click "New Project"
3. Select the "3D (Built-in Render Pipeline)" template (not URP/HDRP for beginners)
4. Name your project (e.g., "HumanoidRobotSimulation")
5. Choose a location and click "Create"

### 3. Install Unity Robotics Hub

1. In Unity, open the Package Manager (Window > Package Manager)
2. Click the "+" button in the top-left corner
3. Select "Add package from git URL..."
4. Enter the URL: `https://github.com/Unity-Technologies/Unity-Robotics-Hub.git`
5. Click "Add"

Alternatively, you can install specific packages:
- **ROS-TCP-Connector**: For ROS 2 communication
- **ROS-TCP-Endpoint**: For ROS 2 endpoint management

## Basic Unity Robotics Setup

### 1. Configure ROS Settings

First, let's set up the basic ROS communication in your Unity scene:

```csharp
// Create a new C# script: ROSConnectionManager.cs

using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class ROSConnectionManager : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    public string rosIPAddress = "127.0.0.1";  // ROS master IP address
    public int rosPort = 10000;                 // ROS TCP port

    private RosConnection m_RosConnection;

    void Start()
    {
        // Get or create the ROS connection instance
        m_RosConnection = RosConnection.instance;

        if (m_RosConnection != null)
        {
            // Connect to ROS
            m_RosConnection.rosIPAddress = rosIPAddress;
            m_RosConnection.rosPort = rosPort;

            Debug.Log($"Connecting to ROS at {rosIPAddress}:{rosPort}");
        }
        else
        {
            Debug.LogError("Failed to get ROS connection instance!");
        }
    }

    // Example method to send a message to ROS
    public void SendTestMessage()
    {
        if (m_RosConnection != null)
        {
            // Publish a simple string message to a test topic
            m_RosConnection.Publish("unity_test_topic", new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.StringMsg
            {
                data = "Hello from Unity!"
            });
        }
    }

    // Example method to subscribe to a ROS topic
    public void SubscribeToTopic()
    {
        if (m_RosConnection != null)
        {
            m_RosConnection.Subscribe<Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.StringMsg>(
                "robot_commands",
                OnRobotCommandReceived
            );
        }
    }

    void OnRobotCommandReceived(Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.StringMsg msg)
    {
        Debug.Log($"Received command from ROS: {msg.data}");
    }
}
```

### 2. Set Up the Unity Scene

Create a basic scene structure for your humanoid robot:

```
Scene Hierarchy:
├── ROSConnectionManager (with ROSConnectionManager.cs)
├── RobotEnvironment
│   ├── GroundPlane
│   ├── Lighting
│   └── Obstacles
├── HumanoidRobot
│   ├── RobotBase (Rigidbody)
│   ├── Torso
│   ├── Head
│   ├── LeftArm
│   │   ├── Shoulder
│   │   ├── Elbow
│   │   └── Hand
│   ├── RightArm
│   │   ├── Shoulder
│   │   ├── Elbow
│   │   └── Hand
│   ├── LeftLeg
│   │   ├── Hip
│   │   ├── Knee
│   │   └── Foot
│   └── RightLeg
│       ├── Hip
│       ├── Knee
│       └── Foot
└── Sensors
    ├── Camera
    ├── IMU
    └── LiDAR
```

### 3. Create Humanoid Robot Components

Let's create a basic humanoid robot structure:

```csharp
// HumanoidRobotController.cs - Main robot controller script

using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;

public class HumanoidRobotController : MonoBehaviour
{
    [Header("Joint Configuration")]
    public Transform[] jointTransforms;
    public ArticulationBody[] jointArticulationBodies;

    [Header("ROS Topics")]
    public string jointStateTopic = "/joint_states";
    public string jointCommandTopic = "/joint_commands";

    [Header("Robot Parameters")]
    public float maxJointVelocity = 5.0f;
    public float jointForceLimit = 100.0f;

    private RosConnection m_RosConnection;
    private float lastPublishTime;
    private const float publishInterval = 0.05f; // 20 Hz

    void Start()
    {
        m_RosConnection = RosConnection.instance;

        // Subscribe to joint commands
        if (m_RosConnection != null)
        {
            m_RosConnection.Subscribe<Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs.JointStateMsg>(
                jointCommandTopic, OnJointCommandReceived);
        }

        // Initialize joint transforms array
        InitializeJointArrays();
    }

    void InitializeJointArrays()
    {
        // Find all ArticulationBody components in children
        jointArticulationBodies = GetComponentsInChildren<ArticulationBody>();

        // Create transforms array for easier access
        jointTransforms = new Transform[jointArticulationBodies.Length];
        for (int i = 0; i < jointArticulationBodies.Length; i++)
        {
            jointTransforms[i] = jointArticulationBodies[i].transform;
        }
    }

    void Update()
    {
        // Publish joint states at regular intervals
        if (Time.time - lastPublishTime > publishInterval)
        {
            PublishJointStates();
            lastPublishTime = Time.time;
        }
    }

    void OnJointCommandReceived(JointStateMsg msg)
    {
        // Process joint commands from ROS
        for (int i = 0; i < msg.name.Count; i++)
        {
            string jointName = msg.name[i];

            // Find the corresponding joint in our robot
            ArticulationBody joint = FindJointByName(jointName);
            if (joint != null && i < msg.position.Count)
            {
                SetJointTargetPosition(joint, msg.position[i]);
            }
        }
    }

    ArticulationBody FindJointByName(string name)
    {
        foreach (ArticulationBody joint in jointArticulationBodies)
        {
            if (joint.name == name || joint.name.ToLower().Contains(name.ToLower()))
            {
                return joint;
            }
        }
        return null;
    }

    void SetJointTargetPosition(ArticulationBody joint, double targetPosition)
    {
        ArticulationDrive drive = joint.xDrive;
        drive.target = (float)targetPosition;
        drive.forceLimit = jointForceLimit;
        drive.damping = 10f; // Adjust based on your robot's requirements
        drive.stiffness = 100f;
        joint.xDrive = drive;
    }

    void PublishJointStates()
    {
        if (m_RosConnection == null) return;

        // Create joint state message
        JointStateMsg jointStateMsg = new JointStateMsg();
        jointStateMsg.name = new System.Collections.Generic.List<string>();
        jointStateMsg.position = new System.Collections.Generic.List<double>();
        jointStateMsg.velocity = new System.Collections.Generic.List<double>();
        jointStateMsg.effort = new System.Collections.Generic.List<double>();

        // Populate with current joint states
        foreach (ArticulationBody joint in jointArticulationBodies)
        {
            jointStateMsg.name.Add(joint.name);
            jointStateMsg.position.Add(joint.jointPosition[0]);
            jointStateMsg.velocity.Add(joint.jointVelocity[0]);
            jointStateMsg.effort.Add(joint.jointForce[0]);
        }

        // Set timestamp
        jointStateMsg.header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.HeaderMsg
        {
            stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Builtin_interfaces.TimeMsg
            {
                sec = (int)Time.time,
                nanosec = (uint)((Time.time % 1) * 1e9)
            },
            frame_id = "base_link"
        };

        m_RosConnection.Publish(jointStateTopic, jointStateMsg);
    }

    // Helper method to set robot to a specific pose
    public void SetRobotPose(Vector3 position, Quaternion rotation)
    {
        transform.position = position;
        transform.rotation = rotation;
    }
}
```

### 4. Configure Articulation Bodies for Joints

Unity uses ArticulationBody components to simulate robot joints. Here's how to configure them:

```csharp
// JointConfigurationHelper.cs - Helper script for joint configuration

using UnityEngine;

public class JointConfigurationHelper : MonoBehaviour
{
    [Header("Joint Configuration")]
    public ArticulationBody joint;
    public JointType jointType = JointType.RevoluteJoint;

    [Header("Joint Limits")]
    public float lowerLimit = -45f;
    public float upperLimit = 45f;
    public float stiffness = 100f;
    public float damping = 10f;
    public float forceLimit = 100f;

    [Header("Drive Settings")]
    public bool useDrive = true;
    public float driveForceLimit = 50f;
    public float driveDamping = 10f;
    public float driveStiffness = 100f;

    [ContextMenu("Configure Joint")]
    public void ConfigureJoint()
    {
        if (joint == null)
        {
            joint = GetComponent<ArticulationBody>();
            if (joint == null)
            {
                Debug.LogError("No ArticulationBody found on this object!");
                return;
            }
        }

        // Set joint type
        joint.jointType = jointType;

        // Configure joint drive
        if (useDrive)
        {
            ArticulationDrive drive = joint.xDrive;
            drive.forceLimit = driveForceLimit;
            drive.damping = driveDamping;
            drive.stiffness = driveStiffness;
            drive.upperLimit = upperLimit;
            drive.lowerLimit = lowerLimit;
            joint.xDrive = drive;
        }

        // Set joint limits based on type
        switch (jointType)
        {
            case JointType.RevoluteJoint:
            case JointType.ContinuousJoint:
                ArticulationDrive xDrive = joint.xDrive;
                if (jointType == JointType.RevoluteJoint)
                {
                    xDrive.upperLimit = upperLimit;
                    xDrive.lowerLimit = lowerLimit;
                }
                xDrive.stiffness = stiffness;
                xDrive.damping = damping;
                xDrive.forceLimit = forceLimit;
                joint.xDrive = xDrive;
                break;

            case JointType.PrismaticJoint:
                ArticulationDrive zDrive = joint.zDrive;
                zDrive.upperLimit = upperLimit;
                zDrive.lowerLimit = lowerLimit;
                zDrive.stiffness = stiffness;
                zDrive.damping = damping;
                zDrive.forceLimit = forceLimit;
                joint.zDrive = zDrive;
                break;
        }

        Debug.Log($"Joint {gameObject.name} configured with type: {jointType}");
    }
}
```

## Setting Up Sensors in Unity

### 1. Camera Sensor Setup

```csharp
// UnityCameraSensor.cs - Camera sensor for ROS

using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using System.Collections;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera cameraComponent;
    public string imageTopic = "/camera/image_raw";
    public string infoTopic = "/camera/camera_info";
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float publishRate = 30f; // Hz

    [Header("Camera Intrinsics")]
    public float fov = 60f; // Field of view in degrees
    public float nearClip = 0.1f;
    public float farClip = 100f;

    private RosConnection m_RosConnection;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private float lastPublishTime;

    void Start()
    {
        m_RosConnection = RosConnection.instance;

        if (cameraComponent == null)
            cameraComponent = GetComponent<Camera>();

        // Create render texture for camera capture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        cameraComponent.targetTexture = renderTexture;

        // Create texture for reading pixels
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        // Set camera parameters
        cameraComponent.fieldOfView = fov;
        cameraComponent.nearClipPlane = nearClip;
        cameraComponent.farClipPlane = farClip;
    }

    void Update()
    {
        if (m_RosConnection != null && Time.time - lastPublishTime > 1f / publishRate)
        {
            CaptureAndPublishImage();
            lastPublishTime = Time.time;
        }
    }

    void CaptureAndPublishImage()
    {
        // Set the active render texture to our camera's render texture
        RenderTexture.active = renderTexture;

        // Read pixels from the active render texture
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to byte array
        byte[] imageBytes = texture2D.EncodeToJPG(85); // Use JPG for smaller size

        // Create ROS image message
        ImageMsg imageMsg = new ImageMsg
        {
            header = new HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "camera_optical_frame"
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(imageWidth * 3), // 3 bytes per pixel for RGB
            data = imageBytes
        };

        m_RosConnection.Publish(imageTopic, imageMsg);

        // Also publish camera info
        PublishCameraInfo();
    }

    void PublishCameraInfo()
    {
        CameraInfoMsg infoMsg = new CameraInfoMsg
        {
            header = new HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "camera_optical_frame"
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            distortion_model = "plumb_bob",
            d = { 0, 0, 0, 0, 0 }, // No distortion for simplicity
            k = {
                CalculateFx(), 0, CalculateCx(),
                0, CalculateFy(), CalculateCy(),
                0, 0, 1
            }, // Camera matrix
            r = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }, // Rectification matrix
            p = {
                CalculateFx(), 0, CalculateCx(), 0,
                0, CalculateFy(), CalculateCy(), 0,
                0, 0, 1, 0
            } // Projection matrix
        };

        m_RosConnection.Publish(infoTopic, infoMsg);
    }

    float CalculateFx() => imageWidth / (2.0f * Mathf.Tan(Mathf.Deg2Rad * fov / 2.0f));
    float CalculateFy() => CalculateFx(); // Assume square pixels
    float CalculateCx() => imageWidth / 2.0f;
    float CalculateCy() => imageHeight / 2.0f;
}
```

### 2. IMU Sensor Setup

```csharp
// UnityIMUMsg.cs - IMU sensor simulation

using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class UnityIMUMsg : MonoBehaviour
{
    [Header("IMU Settings")]
    public string imuTopic = "/imu/data";
    public float publishRate = 100f; // Hz

    [Header("Noise Parameters")]
    public Vector3 linearAccelerationNoise = new Vector3(0.01f, 0.01f, 0.01f);
    public Vector3 angularVelocityNoise = new Vector3(0.01f, 0.01f, 0.01f);

    private RosConnection m_RosConnection;
    private float lastPublishTime;
    private Rigidbody attachedRigidbody;

    void Start()
    {
        m_RosConnection = RosConnection.instance;
        attachedRigidbody = GetComponent<Rigidbody>() ?? GetComponentInParent<Rigidbody>();
    }

    void Update()
    {
        if (m_RosConnection != null && Time.time - lastPublishTime > 1f / publishRate)
        {
            PublishIMUData();
            lastPublishTime = Time.time;
        }
    }

    void PublishIMUData()
    {
        // Get the robot's orientation, angular velocity, and linear acceleration
        Quaternion orientation = transform.rotation;
        Vector3 angularVelocity = attachedRigidbody ? attachedRigidbody.angularVelocity : Vector3.zero;
        Vector3 linearAcceleration = attachedRigidbody ? attachedRigidbody.velocity : Vector3.zero;

        // Apply noise
        linearAcceleration += new Vector3(
            Random.Range(-linearAccelerationNoise.x, linearAccelerationNoise.x),
            Random.Range(-linearAccelerationNoise.y, linearAccelerationNoise.y),
            Random.Range(-linearAccelerationNoise.z, linearAccelerationNoise.z)
        );

        angularVelocity += new Vector3(
            Random.Range(-angularVelocityNoise.x, angularVelocityNoise.x),
            Random.Range(-angularVelocityNoise.y, angularVelocityNoise.y),
            Random.Range(-angularVelocityNoise.z, angularVelocityNoise.z)
        );

        // Create IMU message
        ImuMsg imuMsg = new ImuMsg
        {
            header = new HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.Builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "imu_link"
            },
            orientation = new QuaternionMsg
            {
                x = orientation.x,
                y = orientation.y,
                z = orientation.z,
                w = orientation.w
            },
            angular_velocity = new Vector3Msg
            {
                x = angularVelocity.x,
                y = angularVelocity.y,
                z = angularVelocity.z
            },
            linear_acceleration = new Vector3Msg
            {
                x = linearAcceleration.x,
                y = linearAcceleration.y,
                z = linearAcceleration.z - Physics.gravity.y // Subtract gravity
            }
        };

        m_RosConnection.Publish(imuTopic, imuMsg);
    }
}
```

## ROS 2 Bridge Configuration

### 1. Setting up the ROS-TCP-Endpoint

Create a ROS 2 node to act as the TCP endpoint:

```python
# ros_tcp_endpoint.py - ROS 2 TCP endpoint node

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import socket
import threading
import json
import struct

class ROSEndpoint(Node):
    def __init__(self):
        super().__init__('unity_ros_endpoint')

        # Parameters
        self.declare_parameter('tcp_port', 10000)
        self.tcp_port = self.get_parameter('tcp_port').value
        self.tcp_ip = '0.0.0.0'  # Listen on all interfaces

        # ROS publishers/subscribers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.unity_test_sub = self.create_subscription(String, '/unity_test_topic', self.unity_test_callback, 10)

        # Start TCP server
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server.bind((self.tcp_ip, self.tcp_port))
        self.tcp_server.listen(5)

        self.get_logger().info(f'ROS TCP Endpoint listening on {self.tcp_ip}:{self.tcp_port}')

        # Start server thread
        self.server_thread = threading.Thread(target=self.tcp_server_loop)
        self.server_thread.daemon = True
        self.server_thread.start()

    def tcp_server_loop(self):
        while rclpy.ok():
            try:
                client_socket, address = self.tcp_server.accept()
                self.get_logger().info(f'Connection from {address}')

                # Handle client in separate thread
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.daemon = True
                client_thread.start()

            except Exception as e:
                self.get_logger().error(f'Server error: {e}')

    def handle_client(self, client_socket):
        try:
            while rclpy.ok():
                # Receive message length (4 bytes)
                length_data = client_socket.recv(4)
                if not length_data:
                    break

                msg_length = struct.unpack('<I', length_data)[0]

                # Receive message data
                data = b''
                while len(data) < msg_length:
                    packet = client_socket.recv(msg_length - len(data))
                    if not packet:
                        break
                    data += packet

                # Process message
                self.process_message(data.decode('utf-8'))

        except Exception as e:
            self.get_logger().error(f'Client error: {e}')
        finally:
            client_socket.close()

    def process_message(self, message_str):
        try:
            message = json.loads(message_str)

            # Process based on message type
            msg_type = message.get('type', '')
            topic = message.get('topic', '')
            data = message.get('data', {})

            if topic == '/joint_states':
                self.publish_joint_states(data)
            elif topic == '/unity_test_topic':
                self.get_logger().info(f'Received from Unity: {data}')

        except Exception as e:
            self.get_logger().error(f'Message processing error: {e}')

    def publish_joint_states(self, data):
        msg = JointState()
        # Parse and populate joint state message
        # Implementation depends on your specific message format
        pass

    def unity_test_callback(self, msg):
        self.get_logger().info(f'Received from Unity: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    endpoint = ROSEndpoint()

    try:
        rclpy.spin(endpoint)
    except KeyboardInterrupt:
        endpoint.get_logger().info('Shutting down ROS TCP endpoint')
    finally:
        endpoint.tcp_server.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Launch File for Unity Integration

```xml
<!-- launch/unity_robotics.launch -->
<launch>
  <!-- ROS TCP Endpoint -->
  <node pkg="my_robot_description" exec="ros_tcp_endpoint.py" name="ros_tcp_endpoint">
    <param name="tcp_port" value="10000"/>
  </node>

  <!-- Robot State Publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description" value="$(find-pkg-share my_robot_description)/urdf/humanoid.urdf"/>
  </node>

  <!-- Joint State Publisher -->
  <node pkg="joint_state_publisher" exec="joint_state_publisher" name="joint_state_publisher"/>

</launch>
```

## Testing the Unity Setup

### 1. Basic Connection Test

Create a simple test scene to verify the connection:

```csharp
// ConnectionTest.cs - Simple connection test

using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class ConnectionTest : MonoBehaviour
{
    public string testTopic = "connection_test";
    public float testInterval = 2.0f;

    private RosConnection m_RosConnection;
    private float lastTestTime;

    void Start()
    {
        m_RosConnection = RosConnection.instance;

        if (m_RosConnection != null)
        {
            // Subscribe to echo back
            m_RosConnection.Subscribe<Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.StringMsg>(
                testTopic + "_echo", OnEchoReceived);

            Debug.Log("Connection test initialized");
        }
        else
        {
            Debug.LogError("No ROS connection available!");
        }
    }

    void Update()
    {
        if (m_RosConnection != null && Time.time - lastTestTime > testInterval)
        {
            SendTestMessage();
            lastTestTime = Time.time;
        }
    }

    void SendTestMessage()
    {
        if (m_RosConnection != null)
        {
            var msg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.StringMsg
            {
                data = $"Unity test message at {Time.time}"
            };

            m_RosConnection.Publish(testTopic, msg);
            Debug.Log($"Sent test message: {msg.data}");
        }
    }

    void OnEchoReceived(Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.StringMsg msg)
    {
        Debug.Log($"Received echo: {msg.data}");
    }
}
```

## Performance Optimization

### 1. Unity Physics Optimization

For humanoid robot simulation, optimize Unity's physics settings:

```csharp
// PhysicsOptimizer.cs - Physics optimization script

using UnityEngine;

[ExecuteInEditMode]
public class PhysicsOptimizer : MonoBehaviour
{
    [Header("Physics Settings")]
    public int fixedTimestep = 50; // 50 Hz = 0.02s
    public int maxSubsteps = 8;
    public int solverIterations = 6;
    public int solverVelocityIterations = 1;

    void OnValidate()
    {
        OptimizePhysics();
    }

    void OptimizePhysics()
    {
        // Set physics timestep for humanoid simulation
        Time.fixedDeltaTime = 1.0f / fixedTimestep;
        Time.maximumDeltaTime = 1.0f / 10; // 10 Hz max
        Time.maxFixedDeltaTime = 1.0f / 10; // 10 Hz max

        // Physics manager settings
        Physics.defaultSolverIterations = solverIterations;
        Physics.defaultSolverVelocityIterations = solverVelocityIterations;
        Physics.maxAngularVelocity = 50f; // Reasonable limit for humanoid joints
    }
}
```

### 2. Graphics Optimization

For better performance with complex humanoid models:

```csharp
// GraphicsOptimizer.cs - Graphics optimization for simulation

using UnityEngine;

public class GraphicsOptimizer : MonoBehaviour
{
    [Header("Graphics Settings for Simulation")]
    public bool useLowQuality = true;
    public int targetFrameRate = 60;
    public ShadowQuality shadowQuality = ShadowQuality.Low;
    public LODBias lodBias = LODBias.Low;

    void Start()
    {
        OptimizeForSimulation();
    }

    void OptimizeForSimulation()
    {
        if (useLowQuality)
        {
            QualitySettings.SetQualityLevel(0); // Lowest quality
        }

        Application.targetFrameRate = targetFrameRate;
        QualitySettings.shadows = shadowQuality;
        QualitySettings.lodBias = (float)lodBias;

        // Disable expensive effects for simulation
        RenderSettings.fog = false;
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
    }
}
```

## Next Steps

Now that you have Unity properly set up for robotics simulation, let's explore how to create more complex environments and integrate advanced features. In the next section, we'll cover environment creation and physics optimization specifically for humanoid robots in Unity.