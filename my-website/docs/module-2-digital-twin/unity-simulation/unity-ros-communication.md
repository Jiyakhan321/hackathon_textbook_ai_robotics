---
sidebar_position: 4
---

# Unity-ROS Communication Implementation

## Overview

Establishing reliable communication between Unity and ROS 2 is crucial for creating effective digital twin environments for humanoid robots. This section covers the implementation of ROS communication in Unity using the Unity Robotics Hub, focusing on real-time data exchange, sensor simulation, and control systems.

## Setting Up ROS Communication in Unity

### 1. ROS TCP Connector Configuration

The ROS TCP Connector is the primary communication bridge between Unity and ROS 2. Here's how to properly configure it:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections;
using System.Collections.Generic;

public class ROSCommunicationManager : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    [SerializeField] private string rosIPAddress = "127.0.0.1";
    [SerializeField] private int rosPort = 10000;
    [SerializeField] private float connectionRetryDelay = 2.0f;

    [Header("Communication Settings")]
    [SerializeField] private float publishRate = 50.0f; // Hz
    [SerializeField] private bool enableLogging = true;

    private ROSConnection rosConnection;
    private bool isConnected = false;
    private float publishTimer = 0.0f;

    void Start()
    {
        InitializeROSConnection();
    }

    private void InitializeROSConnection()
    {
        try
        {
            rosConnection = ROSConnection.instance;
            rosConnection.RegisteredAsConnected();

            // Attempt to connect to ROS
            StartCoroutine(TryConnectToROS());
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Failed to initialize ROS connection: {ex.Message}");
        }
    }

    private IEnumerator TryConnectToROS()
    {
        while (!isConnected)
        {
            try
            {
                rosConnection.ConnectToEditor(rosIPAddress, rosPort);
                isConnected = rosConnection.IsConnected();

                if (isConnected)
                {
                    if (enableLogging)
                        Debug.Log($"Successfully connected to ROS at {rosIPAddress}:{rosPort}");
                    break;
                }
                else
                {
                    if (enableLogging)
                        Debug.LogWarning($"Connection failed, retrying in {connectionRetryDelay}s");
                    yield return new WaitForSeconds(connectionRetryDelay);
                }
            }
            catch (System.Exception ex)
            {
                if (enableLogging)
                    Debug.LogError($"Connection error: {ex.Message}");
                yield return new WaitForSeconds(connectionRetryDelay);
            }
        }
    }

    void Update()
    {
        if (isConnected)
        {
            publishTimer += Time.deltaTime;
            float publishInterval = 1.0f / publishRate;

            if (publishTimer >= publishInterval)
            {
                PublishDataToROS();
                publishTimer = 0.0f;
            }
        }
    }

    private void PublishDataToROS()
    {
        // Override this method in derived classes to publish specific data
    }

    public bool IsConnected()
    {
        return isConnected && rosConnection != null && rosConnection.IsConnected();
    }

    public void SendToROS(string topicName, object message)
    {
        if (IsConnected())
        {
            rosConnection.SendUnityMessage(topicName, message);
        }
        else if (enableLogging)
        {
            Debug.LogWarning($"Cannot send to ROS: Not connected to topic {topicName}");
        }
    }

    public void SubscribeToROS<T>(string topicName, System.Action<T> callback) where T : Message
    {
        if (IsConnected())
        {
            rosConnection.Subscribe<T>(topicName, callback);
        }
        else if (enableLogging)
        {
            Debug.LogWarning($"Cannot subscribe to ROS: Not connected to topic {topicName}");
        }
    }

    void OnApplicationQuit()
    {
        if (rosConnection != null)
        {
            rosConnection.Disconnect();
        }
    }
}
```

### 2. Joint State Publisher for Humanoid Robots

Publishing joint states from Unity to ROS 2 is essential for synchronization:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections.Generic;

public class JointStatePublisher : MonoBehaviour
{
    [Header("Joint Configuration")]
    [SerializeField] private ArticulationBody[] jointBodies;
    [SerializeField] private string[] jointNames;
    [SerializeField] private string robotName = "humanoid_robot";

    [Header("Publishing Settings")]
    [SerializeField] private string jointStatesTopic = "/joint_states";
    [SerializeField] private float publishRate = 50.0f;

    private ROSConnection rosConnection;
    private float publishTimer = 0.0f;
    private List<double> jointPositions;
    private List<double> jointVelocities;
    private List<double> jointEfforts;

    void Start()
    {
        InitializeJointArrays();
        rosConnection = ROSConnection.instance;
    }

    private void InitializeJointArrays()
    {
        if (jointBodies.Length != jointNames.Length)
        {
            Debug.LogError("Joint bodies and joint names arrays must have the same length!");
            return;
        }

        jointPositions = new List<double>(new double[jointBodies.Length]);
        jointVelocities = new List<double>(new double[jointBodies.Length]);
        jointEfforts = new List<double>(new double[jointBodies.Length]);
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        float publishInterval = 1.0f / publishRate;

        if (publishTimer >= publishInterval)
        {
            PublishJointStates();
            publishTimer = 0.0f;
        }
    }

    private void PublishJointStates()
    {
        if (jointBodies == null || jointBodies.Length == 0 || rosConnection == null)
            return;

        // Update joint arrays with current values
        for (int i = 0; i < jointBodies.Length; i++)
        {
            if (jointBodies[i] != null)
            {
                jointPositions[i] = jointBodies[i].jointPosition[0];
                jointVelocities[i] = jointBodies[i].jointVelocity[0];
                jointEfforts[i] = jointBodies[i].jointForce[0];
            }
        }

        // Create and publish joint state message
        JointStateMsg jointState = new JointStateMsg
        {
            name = jointNames,
            position = jointPositions.ToArray(),
            velocity = jointVelocities.ToArray(),
            effort = jointEfforts.ToArray(),
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = robotName
            }
        };

        rosConnection.SendUnityMessage(jointStatesTopic, jointState);
    }

    public void SetJointBodies(ArticulationBody[] bodies)
    {
        jointBodies = bodies;
        if (jointNames == null || jointNames.Length != bodies.Length)
        {
            jointNames = new string[bodies.Length];
            for (int i = 0; i < bodies.Length; i++)
            {
                jointNames[i] = bodies[i].name;
            }
        }
        InitializeJointArrays();
    }
}
```

### 3. Joint State Subscriber for Control Commands

Receiving joint commands from ROS 2 to control the Unity robot:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections.Generic;

public class JointStateSubscriber : MonoBehaviour
{
    [Header("Joint Configuration")]
    [SerializeField] private Dictionary<string, ArticulationBody> jointMap = new Dictionary<string, ArticulationBody>();
    [SerializeField] private string jointCommandsTopic = "/joint_commands";

    [Header("Control Settings")]
    [SerializeField] private float positionGain = 3000f;  // kp
    [SerializeField] private float velocityGain = 200f;   // kd
    [SerializeField] private float forceLimit = 1000f;

    private ROSConnection rosConnection;
    private Dictionary<string, double> targetPositions = new Dictionary<string, double>();
    private Dictionary<string, double> targetVelocities = new Dictionary<string, double>();

    void Start()
    {
        rosConnection = ROSConnection.instance;

        // Subscribe to joint commands
        rosConnection.Subscribe<JointStateMsg>(jointCommandsTopic, OnJointCommandReceived);

        // Initialize joint map
        InitializeJointMap();
    }

    private void InitializeJointMap()
    {
        jointMap.Clear();

        // Find all ArticulationBodies in children
        ArticulationBody[] bodies = GetComponentsInChildren<ArticulationBody>();
        foreach (ArticulationBody body in bodies)
        {
            if (!jointMap.ContainsKey(body.name))
            {
                jointMap[body.name] = body;
            }
        }
    }

    private void OnJointCommandReceived(JointStateMsg jointState)
    {
        for (int i = 0; i < jointState.name.Length; i++)
        {
            string jointName = jointState.name[i];

            if (jointMap.ContainsKey(jointName))
            {
                // Store target positions and velocities
                if (i < jointState.position.Length)
                    targetPositions[jointName] = jointState.position[i];

                if (i < jointState.velocity.Length)
                    targetVelocities[jointName] = jointState.velocity[i];
            }
        }
    }

    void FixedUpdate()
    {
        ApplyJointCommands();
    }

    private void ApplyJointCommands()
    {
        foreach (var jointPair in jointMap)
        {
            string jointName = jointPair.Key;
            ArticulationBody jointBody = jointPair.Value;

            if (targetPositions.ContainsKey(jointName))
            {
                double targetPos = targetPositions[jointName];

                // Set drive parameters for position control
                ArticulationDrive drive = jointBody.xDrive;
                drive.position = (float)targetPos;
                drive.positionSpring = positionGain;
                drive.positionDamper = velocityGain;
                drive.forceLimit = forceLimit;
                drive.upperLimit = jointBody.xDrive.upperLimit;
                drive.lowerLimit = jointBody.xDrive.lowerLimit;

                jointBody.xDrive = drive;
            }
        }
    }

    public void AddJoint(string jointName, ArticulationBody jointBody)
    {
        if (!jointMap.ContainsKey(jointName))
        {
            jointMap[jointName] = jointBody;
        }
    }
}
```

## Sensor Data Communication

### 1. IMU Sensor Publisher

Publishing IMU data from Unity to ROS 2:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class IMUSensorPublisher : MonoBehaviour
{
    [Header("IMU Configuration")]
    [SerializeField] private string imuTopic = "/imu/data";
    [SerializeField] private Transform imuTransform;
    [SerializeField] private float publishRate = 100.0f;

    [Header("Noise Settings")]
    [SerializeField] private float linearAccelerationNoise = 0.01f;
    [SerializeField] private float angularVelocityNoise = 0.001f;
    [SerializeField] private float orientationNoise = 0.0001f;

    private ROSConnection rosConnection;
    private float publishTimer = 0.0f;
    private Rigidbody attachedRigidbody;

    void Start()
    {
        rosConnection = ROSConnection.instance;

        if (imuTransform == null)
            imuTransform = transform;

        attachedRigidbody = GetComponent<Rigidbody>();
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        float publishInterval = 1.0f / publishRate;

        if (publishTimer >= publishInterval)
        {
            PublishIMUData();
            publishTimer = 0.0f;
        }
    }

    private void PublishIMUData()
    {
        if (rosConnection == null)
            return;

        // Get orientation (convert Unity to ROS coordinate system)
        Quaternion orientation = imuTransform.rotation;
        // Convert from Unity (left-handed) to ROS (right-handed) coordinate system
        orientation = new Quaternion(orientation.x, orientation.y, -orientation.z, -orientation.w);

        // Get angular velocity
        Vector3 angularVelocity = Vector3.zero;
        if (attachedRigidbody != null)
        {
            angularVelocity = attachedRigidbody.angularVelocity;
        }
        // Convert to ROS coordinate system
        angularVelocity = new Vector3(angularVelocity.x, angularVelocity.y, -angularVelocity.z);

        // Get linear acceleration (approximate from gravity and movement)
        Vector3 linearAcceleration = Physics.gravity;
        if (attachedRigidbody != null)
        {
            linearAcceleration += attachedRigidbody.velocity / Time.fixedDeltaTime;
        }
        // Convert to ROS coordinate system
        linearAcceleration = new Vector3(linearAcceleration.x, linearAcceleration.y, -linearAcceleration.z);

        // Add noise to simulate real sensor
        orientation = AddOrientationNoise(orientation);
        angularVelocity = AddAngularVelocityNoise(angularVelocity);
        linearAcceleration = AddLinearAccelerationNoise(linearAcceleration);

        ImuMsg imuMsg = new ImuMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = imuTransform.name
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
                z = linearAcceleration.z
            }
        };

        rosConnection.SendUnityMessage(imuTopic, imuMsg);
    }

    private Quaternion AddOrientationNoise(Quaternion orientation)
    {
        if (orientationNoise > 0)
        {
            orientation.x += Random.Range(-orientationNoise, orientationNoise);
            orientation.y += Random.Range(-orientationNoise, orientationNoise);
            orientation.z += Random.Range(-orientationNoise, orientationNoise);
            orientation.w += Random.Range(-orientationNoise, orientationNoise);

            // Normalize quaternion
            float magnitude = Mathf.Sqrt(orientation.x * orientation.x +
                                       orientation.y * orientation.y +
                                       orientation.z * orientation.z +
                                       orientation.w * orientation.w);
            if (magnitude > 0)
            {
                orientation.x /= magnitude;
                orientation.y /= magnitude;
                orientation.z /= magnitude;
                orientation.w /= magnitude;
            }
        }
        return orientation;
    }

    private Vector3 AddAngularVelocityNoise(Vector3 angularVelocity)
    {
        if (angularVelocityNoise > 0)
        {
            angularVelocity.x += Random.Range(-angularVelocityNoise, angularVelocityNoise);
            angularVelocity.y += Random.Range(-angularVelocityNoise, angularVelocityNoise);
            angularVelocity.z += Random.Range(-angularVelocityNoise, angularVelocityNoise);
        }
        return angularVelocity;
    }

    private Vector3 AddLinearAccelerationNoise(Vector3 linearAcceleration)
    {
        if (linearAccelerationNoise > 0)
        {
            linearAcceleration.x += Random.Range(-linearAccelerationNoise, linearAccelerationNoise);
            linearAcceleration.y += Random.Range(-linearAccelerationNoise, linearAccelerationNoise);
            linearAcceleration.z += Random.Range(-linearAccelerationNoise, linearAccelerationNoise);
        }
        return linearAcceleration;
    }
}
```

### 2. Camera Sensor Publisher

Publishing camera data from Unity cameras to ROS 2:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections;
using System.Threading.Tasks;

public class CameraSensorPublisher : MonoBehaviour
{
    [Header("Camera Configuration")]
    [SerializeField] private Camera cameraComponent;
    [SerializeField] private string imageTopic = "/camera/image_raw";
    [SerializeField] private string infoTopic = "/camera/camera_info";

    [Header("Image Settings")]
    [SerializeField] private int imageWidth = 640;
    [SerializeField] private int imageHeight = 480;
    [SerializeField] private float publishRate = 30.0f;

    private ROSConnection rosConnection;
    private float publishTimer = 0.0f;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private byte[] imageBytes;

    void Start()
    {
        rosConnection = ROSConnection.instance;

        if (cameraComponent == null)
            cameraComponent = GetComponent<Camera>();

        InitializeCameraTexture();
    }

    private void InitializeCameraTexture()
    {
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        cameraComponent.targetTexture = renderTexture;
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        float publishInterval = 1.0f / publishRate;

        if (publishTimer >= publishInterval)
        {
            PublishCameraData();
            publishTimer = 0.0f;
        }
    }

    private void PublishCameraData()
    {
        if (rosConnection == null || cameraComponent == null)
            return;

        // Render the camera to texture
        RenderTexture.active = renderTexture;
        cameraComponent.Render();

        // Read pixels from render texture
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to byte array
        imageBytes = texture2D.EncodeToJPG();

        // Create and publish image message
        ImageMsg imageMsg = new ImageMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = cameraComponent.name
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(imageWidth * 3), // 3 bytes per pixel for RGB
            data = imageBytes
        };

        rosConnection.SendUnityMessage(imageTopic, imageMsg);

        // Publish camera info
        PublishCameraInfo();
    }

    private void PublishCameraInfo()
    {
        CameraInfoMsg cameraInfo = new CameraInfoMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = cameraComponent.name
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            distortion_model = "plumb_bob",
            D = new double[] { 0, 0, 0, 0, 0 }, // No distortion
            K = new double[] {
                cameraComponent.focalLength * imageWidth / cameraComponent.sensorSize.x, 0, imageWidth / 2.0,
                0, cameraComponent.focalLength * imageHeight / cameraComponent.sensorSize.y, imageHeight / 2.0,
                0, 0, 1
            },
            R = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
            P = new double[] {
                cameraComponent.focalLength * imageWidth / cameraComponent.sensorSize.x, 0, imageWidth / 2.0, 0,
                0, cameraComponent.focalLength * imageHeight / cameraComponent.sensorSize.y, imageHeight / 2.0, 0,
                0, 0, 1, 0
            }
        };

        rosConnection.SendUnityMessage(infoTopic, cameraInfo);
    }

    void OnDestroy()
    {
        if (renderTexture != null)
            RenderTexture.ReleaseTemporary(renderTexture);
    }
}
```

## Communication Architecture Best Practices

### 1. Network Configuration

Proper network setup is essential for stable Unity-ROS communication:

```csharp
using UnityEngine;

[CreateAssetMenu(fileName = "ROSNetworkConfig", menuName = "ROS/Network Configuration")]
public class ROSNetworkConfig : ScriptableObject
{
    [Header("Connection Settings")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Performance Settings")]
    public float publishRate = 50.0f;
    public float connectionRetryDelay = 2.0f;
    public int maxConnectionAttempts = 5;

    [Header("Topic Configuration")]
    public string jointStatesTopic = "/joint_states";
    public string jointCommandsTopic = "/joint_commands";
    public string imuTopic = "/imu/data";
    public string imageTopic = "/camera/image_raw";
    public string infoTopic = "/camera/camera_info";

    [Header("Quality of Service")]
    public bool enableCompression = true;
    public float compressionQuality = 0.8f;
    public bool enableThrottling = true;
    public float maxMessageRate = 100.0f;

    [Header("Security Settings")]
    public bool enableEncryption = false;
    public string encryptionKey = "default_key";

    public void ValidateSettings()
    {
        rosPort = Mathf.Clamp(rosPort, 1024, 65535);
        publishRate = Mathf.Clamp(publishRate, 1.0f, 1000.0f);
        connectionRetryDelay = Mathf.Clamp(connectionRetryDelay, 0.1f, 10.0f);
        maxConnectionAttempts = Mathf.Clamp(maxConnectionAttempts, 1, 10);
    }
}
```

### 2. Message Throttling and Buffering

Implement message throttling to prevent network overload:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class MessageThrottler : MonoBehaviour
{
    [System.Serializable]
    public class TopicThrottleConfig
    {
        public string topicName;
        public float maxRateHz = 50.0f;
        public int maxBufferSize = 10;
    }

    [SerializeField] private List<TopicThrottleConfig> throttleConfigs = new List<TopicThrottleConfig>();
    private Dictionary<string, MessageBuffer> messageBuffers = new Dictionary<string, MessageBuffer>();

    [System.Serializable]
    private class MessageBuffer
    {
        public Queue<object> buffer = new Queue<object>();
        public float lastSendTime = 0f;
        public float minInterval = 0.02f; // 50 Hz default
        public int maxBufferSize = 10;
    }

    void Start()
    {
        InitializeBuffers();
    }

    private void InitializeBuffers()
    {
        foreach (var config in throttleConfigs)
        {
            MessageBuffer buffer = new MessageBuffer
            {
                minInterval = 1.0f / config.maxRateHz,
                maxBufferSize = config.maxBufferSize
            };
            messageBuffers[config.topicName] = buffer;
        }
    }

    public bool ShouldSendMessage(string topicName)
    {
        if (!messageBuffers.ContainsKey(topicName))
            return true; // No throttling for this topic

        MessageBuffer buffer = messageBuffers[topicName];
        float currentTime = Time.time;

        return (currentTime - buffer.lastSendTime) >= buffer.minInterval;
    }

    public void QueueMessage(string topicName, object message)
    {
        if (!messageBuffers.ContainsKey(topicName))
        {
            // If no buffer exists, send immediately
            return;
        }

        MessageBuffer buffer = messageBuffers[topicName];

        // Add to buffer
        buffer.buffer.Enqueue(message);

        // Trim buffer if too large
        while (buffer.buffer.Count > buffer.maxBufferSize)
        {
            buffer.buffer.Dequeue();
        }
    }

    public bool TryGetNextMessage(string topicName, out object message)
    {
        message = null;

        if (!messageBuffers.ContainsKey(topicName))
            return false;

        MessageBuffer buffer = messageBuffers[topicName];

        if (buffer.buffer.Count > 0 && ShouldSendMessage(topicName))
        {
            message = buffer.buffer.Dequeue();
            buffer.lastSendTime = Time.time;
            return true;
        }

        return false;
    }
}
```

## Error Handling and Diagnostics

### 1. Connection Monitoring

Monitor and handle connection issues gracefully:

```csharp
using UnityEngine;
using System;

public class ROSConnectionMonitor : MonoBehaviour
{
    [Header("Connection Settings")]
    [SerializeField] private float heartbeatInterval = 1.0f;
    [SerializeField] private float timeoutThreshold = 5.0f;
    [SerializeField] private bool enableReconnection = true;

    [Header("Diagnostics")]
    [SerializeField] private bool enableLogging = true;
    [SerializeField] private float logInterval = 2.0f;

    private float heartbeatTimer = 0.0f;
    private float lastMessageTime = 0.0f;
    private bool isConnectionHealthy = true;
    private ROSCommunicationManager communicationManager;
    private float logTimer = 0.0f;

    void Start()
    {
        communicationManager = GetComponent<ROSCommunicationManager>();
        lastMessageTime = Time.time;
    }

    void Update()
    {
        // Check heartbeat
        heartbeatTimer += Time.deltaTime;
        if (heartbeatTimer >= heartbeatInterval)
        {
            SendHeartbeat();
            heartbeatTimer = 0.0f;
        }

        // Check for timeout
        CheckConnectionTimeout();

        // Log diagnostics periodically
        if (enableLogging)
        {
            logTimer += Time.deltaTime;
            if (logTimer >= logInterval)
            {
                LogDiagnostics();
                logTimer = 0.0f;
            }
        }
    }

    private void SendHeartbeat()
    {
        if (communicationManager != null && communicationManager.IsConnected())
        {
            // Send a simple heartbeat message
            var heartbeatMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.StringMsg
            {
                data = $"heartbeat_{Time.time}"
            };

            communicationManager.SendToROS("/heartbeat", heartbeatMsg);
            lastMessageTime = Time.time;
        }
    }

    private void CheckConnectionTimeout()
    {
        float timeSinceLastMessage = Time.time - lastMessageTime;

        if (timeSinceLastMessage > timeoutThreshold)
        {
            if (isConnectionHealthy)
            {
                if (enableLogging)
                    Debug.LogWarning("ROS connection timeout detected!");

                isConnectionHealthy = false;

                if (enableReconnection)
                {
                    StartCoroutine(AttemptReconnection());
                }
            }
        }
        else
        {
            isConnectionHealthy = true;
        }
    }

    private System.Collections.IEnumerator AttemptReconnection()
    {
        if (communicationManager != null)
        {
            if (enableLogging)
                Debug.Log("Attempting to reconnect to ROS...");

            // Wait a bit before attempting reconnection
            yield return new WaitForSeconds(1.0f);

            // This would typically involve restarting the ROS connection
            // Implementation depends on your specific ROS setup
        }
    }

    private void LogDiagnostics()
    {
        if (communicationManager != null)
        {
            string status = communicationManager.IsConnected() ? "CONNECTED" : "DISCONNECTED";
            float timeSinceLastMsg = Time.time - lastMessageTime;

            Debug.Log($"ROS Connection Status: {status} | Last message: {timeSinceLastMsg:F1}s ago | Healthy: {isConnectionHealthy}");
        }
    }

    public bool IsConnectionHealthy()
    {
        return isConnectionHealthy;
    }

    public float GetTimeSinceLastMessage()
    {
        return Time.time - lastMessageTime;
    }
}
```

## Performance Optimization

### 1. Efficient Message Serialization

Optimize message serialization for better performance:

```csharp
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;

public class OptimizedMessagePublisher : MonoBehaviour
{
    [Header("Optimization Settings")]
    [SerializeField] private bool enableMessagePooling = true;
    [SerializeField] private int messagePoolSize = 100;

    private Dictionary<string, Queue<object>> messagePool = new Dictionary<string, Queue<object>>();
    private Dictionary<string, int> messageCount = new Dictionary<string, int>();

    public T GetMessageFromPool<T>(string topicName) where T : new()
    {
        if (!enableMessagePooling)
            return new T();

        if (!messagePool.ContainsKey(topicName))
        {
            messagePool[topicName] = new Queue<object>();
            messageCount[topicName] = 0;
        }

        Queue<object> pool = messagePool[topicName];

        if (pool.Count > 0)
        {
            return (T)pool.Dequeue();
        }
        else
        {
            messageCount[topicName]++;
            return new T();
        }
    }

    public void ReturnMessageToPool(string topicName, object message)
    {
        if (!enableMessagePooling || message == null)
            return;

        if (!messagePool.ContainsKey(topicName))
        {
            messagePool[topicName] = new Queue<object>();
        }

        Queue<object> pool = messagePool[topicName];

        // Limit pool size to prevent memory issues
        if (pool.Count < messagePoolSize)
        {
            pool.Enqueue(message);
        }
    }

    // Example of using pooled messages
    public void PublishOptimizedJointState(string topicName, ArticulationBody[] joints, string[] jointNames)
    {
        var jointState = GetMessageFromPool<Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor.JointStateMsg>(topicName);

        // Use the message...
        jointState.name = jointNames;
        // ... set other properties

        // Publish the message
        ROSConnection.instance.SendUnityMessage(topicName, jointState);

        // Return to pool after a delay or when appropriate
        ReturnMessageToPool(topicName, jointState);
    }
}
```

## Testing and Validation

### 1. Communication Validation Script

Test and validate the communication system:

```csharp
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class ROSCommunicationValidator : MonoBehaviour
{
    [Header("Validation Settings")]
    [SerializeField] private float testDuration = 10.0f;
    [SerializeField] private float testInterval = 0.1f;
    [SerializeField] private string testTopic = "/test_communication";

    [Header("Validation Results")]
    [SerializeField] private bool isConnected = false;
    [SerializeField] private int messagesSent = 0;
    [SerializeField] private int messagesReceived = 0;
    [SerializeField] private float lastRoundTripTime = 0.0f;

    private ROSCommunicationManager communicationManager;
    private float testStartTime = 0.0f;
    private bool isTesting = false;

    void Start()
    {
        communicationManager = GetComponent<ROSCommunicationManager>();
    }

    public void StartValidationTest()
    {
        if (isTesting) return;

        if (communicationManager == null || !communicationManager.IsConnected())
        {
            Debug.LogError("Cannot start validation test: Not connected to ROS!");
            return;
        }

        isTesting = true;
        testStartTime = Time.time;
        messagesSent = 0;
        messagesReceived = 0;

        // Subscribe to echo messages
        communicationManager.SubscribeToROS<Unity.Robotics.ROSTCPConnector.MessageTypes.Std.StringMsg>(
            "/echo_test", OnEchoReceived);

        StartCoroutine(RunValidationTest());
    }

    private IEnumerator RunValidationTest()
    {
        float testStart = Time.time;

        while ((Time.time - testStart) < testDuration && isTesting)
        {
            // Send test message
            var testMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.StringMsg
            {
                data = $"test_{Time.time}_{messagesSent}"
            };

            communicationManager.SendToROS(testTopic, testMsg);
            messagesSent++;

            yield return new WaitForSeconds(testInterval);
        }

        EndValidationTest();
    }

    private void OnEchoReceived(Unity.Robotics.ROSTCPConnector.MessageTypes.Std.StringMsg msg)
    {
        messagesReceived++;
        // Calculate round trip time if possible
    }

    private void EndValidationTest()
    {
        isTesting = false;

        Debug.Log($"Communication Validation Results:\n" +
                 $"Duration: {testDuration}s\n" +
                 $"Messages Sent: {messagesSent}\n" +
                 $"Messages Received: {messagesReceived}\n" +
                 $"Success Rate: {(messagesReceived > 0 ? (float)messagesReceived / messagesSent * 100 : 0):F1}%");
    }

    public void StopValidationTest()
    {
        isTesting = false;
    }

    void OnValidate()
    {
        testDuration = Mathf.Clamp(testDuration, 1.0f, 60.0f);
        testInterval = Mathf.Clamp(testInterval, 0.01f, 1.0f);
    }
}
```

## Next Steps

Now that you have implemented comprehensive ROS communication in Unity for humanoid robots, you can integrate these components into your digital twin system. The communication layer provides the foundation for real-time synchronization between Unity and ROS 2, enabling realistic simulation and control of humanoid robots.

In the next section, we'll explore sensor simulation in Unity, covering how to create realistic sensor data for cameras, LiDAR, IMU, and other sensors commonly used in humanoid robotics applications.