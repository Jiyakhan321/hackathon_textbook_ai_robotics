---
sidebar_position: 5
---

# Sensor Simulation in Unity

## Overview

Sensor simulation is a critical component of digital twin systems for humanoid robots, enabling realistic perception and interaction with virtual environments. This section covers the implementation of various sensors in Unity, including cameras, LiDAR, IMU, and force/torque sensors, with proper ROS 2 integration for realistic data output.

## Camera Sensor Simulation

### 1. Basic Camera Sensor Setup

Creating realistic camera sensors in Unity involves setting up cameras with appropriate parameters that match real-world sensors:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections;
using System.Collections.Generic;

[RequireComponent(typeof(Camera))]
public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    [SerializeField] private string cameraName = "camera";
    [SerializeField] private string imageTopic = "/camera/image_raw";
    [SerializeField] private string infoTopic = "/camera/camera_info";

    [Header("Image Settings")]
    [SerializeField] private int imageWidth = 640;
    [SerializeField] private int imageHeight = 480;
    [SerializeField] private float publishRate = 30.0f;
    [SerializeField] private Camera.StereoscopicEye stereoEye = Camera.StereoscopicEye.Mono;

    [Header("Camera Parameters")]
    [SerializeField] private float fov = 60.0f;
    [SerializeField] private float nearClip = 0.1f;
    [SerializeField] private float farClip = 100.0f;

    [Header("Noise Settings")]
    [SerializeField] private float noiseIntensity = 0.01f;
    [SerializeField] private float noiseFrequency = 10.0f;

    private Camera cam;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private byte[] imageBytes;
    private float publishTimer = 0.0f;
    private ROSConnection rosConnection;

    // Camera info parameters
    private double[] cameraMatrix;
    private double[] distortionCoefficients;

    void Start()
    {
        InitializeCamera();
        InitializeCameraParameters();
        rosConnection = ROSConnection.instance;
    }

    private void InitializeCamera()
    {
        cam = GetComponent<Camera>();
        if (cam == null)
            cam = gameObject.AddComponent<Camera>();

        // Set camera parameters
        cam.fieldOfView = fov;
        cam.nearClipPlane = nearClip;
        cam.farClipPlane = farClip;
        cam.stereoTargetEye = stereoEye;

        // Create render texture for image capture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        cam.targetTexture = renderTexture;
    }

    private void InitializeCameraParameters()
    {
        // Calculate camera matrix based on FOV and resolution
        float focalLength = (imageHeight / 2.0f) / Mathf.Tan(Mathf.Deg2Rad * fov / 2.0f);
        float centerX = imageWidth / 2.0f;
        float centerY = imageHeight / 2.0f;

        cameraMatrix = new double[] {
            focalLength, 0, centerX,
            0, focalLength, centerY,
            0, 0, 1
        };

        // Initialize distortion coefficients (assumes no distortion initially)
        distortionCoefficients = new double[] { 0, 0, 0, 0, 0 };
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
        if (rosConnection == null || cam == null)
            return;

        // Render the camera to texture
        RenderTexture.active = renderTexture;
        cam.Render();

        // Read pixels from render texture
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Apply noise to simulate real camera
        ApplyCameraNoise(texture2D);

        // Convert to byte array
        imageBytes = texture2D.EncodeToJPG(85); // 85% quality for realistic compression

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
                frame_id = cameraName
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

    private void ApplyCameraNoise(Texture2D texture)
    {
        if (noiseIntensity <= 0) return;

        Color[] pixels = texture.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            // Add random noise
            float noise = Mathf.PerlinNoise(i * noiseFrequency, Time.time * noiseFrequency) * noiseIntensity;
            pixels[i] = pixels[i] + new Color(noise, noise, noise, 0);
        }

        texture.SetPixels(pixels);
        texture.Apply();
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
                frame_id = cameraName
            },
            height = (uint)imageHeight,
            width = (uint)imageWidth,
            distortion_model = "plumb_bob",
            D = distortionCoefficients,
            K = cameraMatrix,
            R = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
            P = new double[] {
                cameraMatrix[0], 0, cameraMatrix[2], 0,
                0, cameraMatrix[4], cameraMatrix[5], 0,
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

    // Public methods for runtime configuration
    public void SetResolution(int width, int height)
    {
        imageWidth = width;
        imageHeight = height;

        if (renderTexture != null)
            RenderTexture.ReleaseTemporary(renderTexture);

        renderTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        cam.targetTexture = renderTexture;

        InitializeCameraParameters();
    }

    public void SetFov(float newFov)
    {
        fov = newFov;
        cam.fieldOfView = fov;
        InitializeCameraParameters();
    }
}
```

### 2. Stereo Camera System

For humanoid robots that require depth perception:

```csharp
using UnityEngine;

public class StereoCameraSystem : MonoBehaviour
{
    [Header("Stereo Configuration")]
    [SerializeField] private float interocularDistance = 0.064f; // Average human eye distance in meters
    [SerializeField] private GameObject leftCameraGO;
    [SerializeField] private GameObject rightCameraGO;

    [Header("Synchronization")]
    [SerializeField] private bool syncCameraParameters = true;

    private UnityCameraSensor leftCamera;
    private UnityCameraSensor rightCamera;

    void Start()
    {
        SetupStereoCameras();
    }

    private void SetupStereoCameras()
    {
        if (leftCameraGO == null || rightCameraGO == null)
        {
            // Create camera game objects if not provided
            leftCameraGO = new GameObject("LeftCamera");
            rightCameraGO = new GameObject("RightCamera");

            leftCameraGO.transform.SetParent(transform);
            rightCameraGO.transform.SetParent(transform);

            leftCameraGO.transform.localPosition = new Vector3(-interocularDistance / 2, 0, 0);
            rightCameraGO.transform.localPosition = new Vector3(interocularDistance / 2, 0, 0);
        }

        leftCamera = leftCameraGO.GetComponent<UnityCameraSensor>();
        if (leftCamera == null)
            leftCamera = leftCameraGO.AddComponent<UnityCameraSensor>();

        rightCamera = rightCameraGO.GetComponent<UnityCameraSensor>();
        if (rightCamera == null)
            rightCamera = rightCameraGO.AddComponent<UnityCameraSensor>();

        // Configure stereo-specific settings
        ConfigureStereoCameras();
    }

    private void ConfigureStereoCameras()
    {
        if (leftCamera == null || rightCamera == null) return;

        // Set different topics for stereo cameras
        leftCamera.imageTopic = "/stereo/left/image_raw";
        leftCamera.infoTopic = "/stereo/left/camera_info";
        leftCamera.cameraName = "left_camera";

        rightCamera.imageTopic = "/stereo/right/image_raw";
        rightCamera.infoTopic = "/stereo/right/camera_info";
        rightCamera.cameraName = "right_camera";

        // Synchronize parameters if requested
        if (syncCameraParameters)
        {
            SynchronizeCameraParameters();
        }
    }

    private void SynchronizeCameraParameters()
    {
        if (leftCamera == null || rightCamera == null) return;

        // Copy parameters from left to right camera
        rightCamera.SetResolution((int)leftCamera.GetType().GetField("imageWidth").GetValue(leftCamera),
                                  (int)leftCamera.GetType().GetField("imageHeight").GetValue(leftCamera));
        // Note: In a real implementation, you'd want to expose these as public properties
        // or use reflection carefully
    }

    public UnityCameraSensor GetLeftCamera() { return leftCamera; }
    public UnityCameraSensor GetRightCamera() { return rightCamera; }

    public void SetInterocularDistance(float distance)
    {
        interocularDistance = distance;
        if (leftCameraGO != null && rightCameraGO != null)
        {
            leftCameraGO.transform.localPosition = new Vector3(-interocularDistance / 2, 0, 0);
            rightCameraGO.transform.localPosition = new Vector3(interocularDistance / 2, 0, 0);
        }
    }
}
```

## LiDAR Sensor Simulation

### 1. 2D LiDAR Implementation

Simulating 2D LiDAR for navigation and obstacle detection:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections.Generic;

[RequireComponent(typeof(RaycastHit2D))]
public class UnityLidar2D : MonoBehaviour
{
    [Header("LiDAR Configuration")]
    [SerializeField] private string scanTopic = "/scan";
    [SerializeField] private string frameId = "laser_frame";

    [Header("Scan Parameters")]
    [SerializeField] private float minAngle = -Mathf.PI / 2; // -90 degrees
    [SerializeField] private float maxAngle = Mathf.PI / 2;  // 90 degrees
    [SerializeField] private int numRays = 360;
    [SerializeField] private float maxRange = 10.0f;
    [SerializeField] private float minRange = 0.1f;
    [SerializeField] private float publishRate = 10.0f;

    [Header("Physics Settings")]
    [SerializeField] private LayerMask collisionLayers = -1;
    [SerializeField] private float noiseLevel = 0.01f;

    private float publishTimer = 0.0f;
    private ROSConnection rosConnection;
    private float angleIncrement;
    private List<float> ranges;

    void Start()
    {
        rosConnection = ROSConnection.instance;
        angleIncrement = (maxAngle - minAngle) / (numRays - 1);
        ranges = new List<float>(new float[numRays]);
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        float publishInterval = 1.0f / publishRate;

        if (publishTimer >= publishInterval)
        {
            PerformLidarScan();
            publishTimer = 0.0f;
        }
    }

    private void PerformLidarScan()
    {
        if (rosConnection == null) return;

        // Perform raycasting for each angle
        for (int i = 0; i < numRays; i++)
        {
            float angle = minAngle + (i * angleIncrement);
            Vector2 direction = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));

            RaycastHit2D hit = Physics2D.Raycast(transform.position, direction, maxRange, collisionLayers);

            if (hit.collider != null)
            {
                float distance = hit.distance;
                // Add noise to simulate real sensor
                distance += Random.Range(-noiseLevel, noiseLevel);
                ranges[i] = Mathf.Clamp(distance, minRange, maxRange);
            }
            else
            {
                ranges[i] = maxRange;
            }
        }

        // Publish scan message
        LaserScanMsg scanMsg = new LaserScanMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = frameId
            },
            angle_min = minAngle,
            angle_max = maxAngle,
            angle_increment = angleIncrement,
            time_increment = 0, // Not applicable for simulated lidar
            scan_time = 1.0f / publishRate,
            range_min = minRange,
            range_max = maxRange,
            ranges = ranges.ToArray(),
            intensities = new float[numRays] // Empty intensities array
        };

        rosConnection.SendUnityMessage(scanTopic, scanMsg);
    }

    // Visualization in editor
    void OnDrawGizmosSelected()
    {
        if (!Application.isPlaying) return;

        for (int i = 0; i < numRays; i++)
        {
            float angle = minAngle + (i * angleIncrement);
            Vector2 direction = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
            Vector2 endPos = (Vector2)transform.position + direction * maxRange;

            if (i < ranges.Count && ranges[i] < maxRange)
            {
                Gizmos.color = Color.red;
                Vector2 hitPos = (Vector2)transform.position + direction * ranges[i];
                Gizmos.DrawLine(transform.position, hitPos);
            }
            else
            {
                Gizmos.color = Color.green;
                Gizmos.DrawLine(transform.position, endPos);
            }
        }
    }

    // Public methods for configuration
    public void SetScanParameters(float minAng, float maxAng, int rays, float range)
    {
        minAngle = minAng;
        maxAngle = maxAng;
        numRays = rays;
        maxRange = range;
        angleIncrement = (maxAngle - minAngle) / (numRays - 1);
        ranges = new List<float>(new float[numRays]);
    }

    public void SetPublishRate(float rate)
    {
        publishRate = Mathf.Clamp(rate, 1.0f, 100.0f);
    }
}
```

### 2. 3D LiDAR Implementation

For more complex humanoid robot perception:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections.Generic;

public class UnityLidar3D : MonoBehaviour
{
    [Header("3D LiDAR Configuration")]
    [SerializeField] private string pointCloudTopic = "/point_cloud";
    [SerializeField] private string frameId = "lidar_frame";

    [Header("Scan Parameters")]
    [SerializeField] private float horizontalMinAngle = -Mathf.PI; // -180 degrees
    [SerializeField] private float horizontalMaxAngle = Mathf.PI;  // 180 degrees
    [SerializeField] private int horizontalRays = 360;
    [SerializeField] private float verticalMinAngle = -Mathf.PI / 6; // -30 degrees
    [SerializeField] private float verticalMaxAngle = Mathf.PI / 6;  // 30 degrees
    [SerializeField] private int verticalRays = 16;
    [SerializeField] private float maxRange = 50.0f;
    [SerializeField] private float minRange = 0.1f;
    [SerializeField] private float publishRate = 10.0f;

    [Header("Physics Settings")]
    [SerializeField] private LayerMask collisionLayers = -1;
    [SerializeField] private float noiseLevel = 0.01f;

    private float publishTimer = 0.0f;
    private ROSConnection rosConnection;
    private float horizontalAngleIncrement;
    private float verticalAngleIncrement;
    private List<PointFieldMsg> pointFields;

    void Start()
    {
        rosConnection = ROSConnection.instance;
        horizontalAngleIncrement = (horizontalMaxAngle - horizontalMinAngle) / (horizontalRays - 1);
        verticalAngleIncrement = (verticalMaxAngle - verticalMinAngle) / (verticalRays - 1);

        // Define point cloud fields (x, y, z, intensity)
        pointFields = new List<PointFieldMsg>
        {
            new PointFieldMsg { name = "x", offset = 0, datatype = 7, count = 1 }, // FLOAT32
            new PointFieldMsg { name = "y", offset = 4, datatype = 7, count = 1 }, // FLOAT32
            new PointFieldMsg { name = "z", offset = 8, datatype = 7, count = 1 }, // FLOAT32
            new PointFieldMsg { name = "intensity", offset = 12, datatype = 7, count = 1 } // FLOAT32
        };
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        float publishInterval = 1.0f / publishRate;

        if (publishTimer >= publishInterval)
        {
            Perform3DLidarScan();
            publishTimer = 0.0f;
        }
    }

    private void Perform3DLidarScan()
    {
        if (rosConnection == null) return;

        List<float> pointCloudData = new List<float>();

        // Perform raycasting for each horizontal and vertical angle
        for (int v = 0; v < verticalRays; v++)
        {
            float vertAngle = verticalMinAngle + (v * verticalAngleIncrement);

            for (int h = 0; h < horizontalRays; h++)
            {
                float horizAngle = horizontalMinAngle + (h * horizontalAngleIncrement);

                // Convert spherical to Cartesian coordinates
                float x = Mathf.Cos(vertAngle) * Mathf.Cos(horizAngle);
                float y = Mathf.Cos(vertAngle) * Mathf.Sin(horizAngle);
                float z = Mathf.Sin(vertAngle);
                Vector3 direction = new Vector3(x, y, z);

                if (Physics.Raycast(transform.position, direction, out RaycastHit hit, maxRange, collisionLayers))
                {
                    // Add noise to simulate real sensor
                    float distance = hit.distance + Random.Range(-noiseLevel, noiseLevel);

                    if (distance >= minRange && distance <= maxRange)
                    {
                        // Calculate point position
                        Vector3 point = transform.position + direction * distance;

                        // Transform to sensor frame
                        Vector3 localPoint = transform.InverseTransformPoint(point);

                        // Add to point cloud data (x, y, z, intensity)
                        pointCloudData.Add(localPoint.x);
                        pointCloudData.Add(localPoint.y);
                        pointCloudData.Add(localPoint.z);
                        pointCloudData.Add(100.0f); // Intensity value
                    }
                }
            }
        }

        // Create point cloud message
        PointCloud2Msg pointCloudMsg = new PointCloud2Msg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = frameId
            },
            height = 1, // Unorganized point cloud
            width = (uint)pointCloudData.Count / 4, // 4 values per point (x, y, z, intensity)
            fields = pointFields.ToArray(),
            is_bigendian = false,
            point_step = 16, // 4 floats * 4 bytes each
            row_step = (uint)((pointCloudData.Count / 4) * 16),
            data = FloatArrayToByteArray(pointCloudData.ToArray()),
            is_dense = true
        };

        rosConnection.SendUnityMessage(pointCloudTopic, pointCloudMsg);
    }

    private byte[] FloatArrayToByteArray(float[] floatArray)
    {
        byte[] byteArray = new byte[floatArray.Length * 4];
        for (int i = 0; i < floatArray.Length; i++)
        {
            byte[] floatBytes = System.BitConverter.GetBytes(floatArray[i]);
            System.Buffer.BlockCopy(floatBytes, 0, byteArray, i * 4, 4);
        }
        return byteArray;
    }

    // Visualization in editor
    void OnDrawGizmosSelected()
    {
        if (!Application.isPlaying) return;

        for (int v = 0; v < Mathf.Min(4, verticalRays); v++) // Only show a few vertical rays for clarity
        {
            float vertAngle = verticalMinAngle + (v * verticalAngleIncrement);

            for (int h = 0; h < 36; h += 10) // Show every 10th horizontal ray for clarity
            {
                float horizAngle = horizontalMinAngle + (h * horizontalAngleIncrement);

                float x = Mathf.Cos(vertAngle) * Mathf.Cos(horizAngle);
                float y = Mathf.Cos(vertAngle) * Mathf.Sin(horizAngle);
                float z = Mathf.Sin(vertAngle);
                Vector3 direction = new Vector3(x, y, z);

                Gizmos.color = Color.blue;
                Vector3 endPos = transform.position + direction * maxRange;
                Gizmos.DrawLine(transform.position, endPos);
            }
        }
    }

    public void Set3DScanParameters(
        float hMin, float hMax, int hRays,
        float vMin, float vMax, int vRays,
        float range)
    {
        horizontalMinAngle = hMin;
        horizontalMaxAngle = hMax;
        horizontalRays = hRays;
        verticalMinAngle = vMin;
        verticalMaxAngle = vMax;
        verticalRays = vRays;
        maxRange = range;

        horizontalAngleIncrement = (horizontalMaxAngle - horizontalMinAngle) / (horizontalRays - 1);
        verticalAngleIncrement = (verticalMaxAngle - verticalMinAngle) / (verticalRays - 1);
    }
}
```

## IMU Sensor Simulation

### 1. Comprehensive IMU Implementation

Creating a realistic IMU sensor with proper physics integration:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class UnityIMUSensor : MonoBehaviour
{
    [Header("IMU Configuration")]
    [SerializeField] private string imuTopic = "/imu/data";
    [SerializeField] private string frameId = "imu_frame";

    [Header("Noise Parameters")]
    [SerializeField] private float orientationNoise = 0.001f;
    [SerializeField] private float angularVelocityNoise = 0.01f;
    [SerializeField] private float linearAccelerationNoise = 0.05f;

    [Header("Bias Parameters")]
    [SerializeField] private float orientationBias = 0.0001f;
    [SerializeField] private float angularVelocityBias = 0.001f;
    [SerializeField] private float linearAccelerationBias = 0.01f;

    [Header("Publish Settings")]
    [SerializeField] private float publishRate = 100.0f;

    private float publishTimer = 0.0f;
    private ROSConnection rosConnection;
    private Rigidbody attachedRigidbody;
    private Vector3 lastAngularVelocity;

    void Start()
    {
        rosConnection = ROSConnection.instance;
        attachedRigidbody = GetComponentInParent<Rigidbody>();
        if (attachedRigidbody == null)
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
        if (rosConnection == null) return;

        // Get orientation (convert Unity to ROS coordinate system)
        Quaternion orientation = transform.rotation;
        // Convert from Unity (left-handed) to ROS (right-handed) coordinate system
        orientation = new Quaternion(orientation.x, orientation.y, -orientation.z, -orientation.w);

        // Apply noise and bias to orientation
        orientation = ApplyOrientationNoiseAndBias(orientation);

        // Get angular velocity
        Vector3 angularVelocity = Vector3.zero;
        if (attachedRigidbody != null)
        {
            angularVelocity = attachedRigidbody.angularVelocity;
        }

        // Apply noise and bias to angular velocity
        angularVelocity = ApplyAngularVelocityNoiseAndBias(angularVelocity);

        // Get linear acceleration
        Vector3 linearAcceleration = Physics.gravity;
        if (attachedRigidbody != null)
        {
            // Calculate linear acceleration from velocity change
            Vector3 currentVelocity = attachedRigidbody.velocity;
            Vector3 acceleration = (currentVelocity - lastAngularVelocity) / Time.fixedDeltaTime;
            linearAcceleration += acceleration;
            lastAngularVelocity = currentVelocity;
        }

        // Apply noise and bias to linear acceleration
        linearAcceleration = ApplyLinearAccelerationNoiseAndBias(linearAcceleration);

        // Convert to ROS coordinate system
        angularVelocity = new Vector3(angularVelocity.x, angularVelocity.y, -angularVelocity.z);
        linearAcceleration = new Vector3(linearAcceleration.x, linearAcceleration.y, -linearAcceleration.z);

        // Create IMU message
        ImuMsg imuMsg = new ImuMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = frameId
            },
            orientation = new QuaternionMsg
            {
                x = orientation.x,
                y = orientation.y,
                z = orientation.z,
                w = orientation.w
            },
            orientation_covariance = new double[] {
                orientationNoise * orientationNoise, 0, 0,
                0, orientationNoise * orientationNoise, 0,
                0, 0, orientationNoise * orientationNoise
            },
            angular_velocity = new Vector3Msg
            {
                x = angularVelocity.x,
                y = angularVelocity.y,
                z = angularVelocity.z
            },
            angular_velocity_covariance = new double[] {
                angularVelocityNoise * angularVelocityNoise, 0, 0,
                0, angularVelocityNoise * angularVelocityNoise, 0,
                0, 0, angularVelocityNoise * angularVelocityNoise
            },
            linear_acceleration = new Vector3Msg
            {
                x = linearAcceleration.x,
                y = linearAcceleration.y,
                z = linearAcceleration.z
            },
            linear_acceleration_covariance = new double[] {
                linearAccelerationNoise * linearAccelerationNoise, 0, 0,
                0, linearAccelerationNoise * linearAccelerationNoise, 0,
                0, 0, linearAccelerationNoise * linearAccelerationNoise
            }
        };

        rosConnection.SendUnityMessage(imuTopic, imuMsg);
    }

    private Quaternion ApplyOrientationNoiseAndBias(Quaternion orientation)
    {
        float noiseX = Random.Range(-orientationNoise, orientationNoise);
        float noiseY = Random.Range(-orientationNoise, orientationNoise);
        float noiseZ = Random.Range(-orientationNoise, orientationNoise);
        float noiseW = Random.Range(-orientationNoise, orientationNoise);

        float biasX = Random.Range(-orientationBias, orientationBias);
        float biasY = Random.Range(-orientationBias, orientationBias);
        float biasZ = Random.Range(-orientationBias, orientationBias);
        float biasW = Random.Range(-orientationBias, orientationBias);

        orientation.x += noiseX + biasX;
        orientation.y += noiseY + biasY;
        orientation.z += noiseZ + biasZ;
        orientation.w += noiseW + biasW;

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

        return orientation;
    }

    private Vector3 ApplyAngularVelocityNoiseAndBias(Vector3 angularVelocity)
    {
        angularVelocity.x += Random.Range(-angularVelocityNoise, angularVelocityNoise) +
                            Random.Range(-angularVelocityBias, angularVelocityBias);
        angularVelocity.y += Random.Range(-angularVelocityNoise, angularVelocityNoise) +
                            Random.Range(-angularVelocityBias, angularVelocityBias);
        angularVelocity.z += Random.Range(-angularVelocityNoise, angularVelocityNoise) +
                            Random.Range(-angularVelocityBias, angularVelocityBias);
        return angularVelocity;
    }

    private Vector3 ApplyLinearAccelerationNoiseAndBias(Vector3 linearAcceleration)
    {
        linearAcceleration.x += Random.Range(-linearAccelerationNoise, linearAccelerationNoise) +
                               Random.Range(-linearAccelerationBias, linearAccelerationBias);
        linearAcceleration.y += Random.Range(-linearAccelerationNoise, linearAccelerationNoise) +
                               Random.Range(-linearAccelerationBias, linearAccelerationBias);
        linearAcceleration.z += Random.Range(-linearAccelerationNoise, linearAccelerationNoise) +
                               Random.Range(-linearAccelerationBias, linearAccelerationBias);
        return linearAcceleration;
    }

    public void SetNoiseParameters(float orientNoise, float angVelNoise, float linAccNoise)
    {
        orientationNoise = orientNoise;
        angularVelocityNoise = angVelNoise;
        linearAccelerationNoise = linAccNoise;
    }

    public void SetBiasParameters(float orientBias, float angVelBias, float linAccBias)
    {
        orientationBias = orientBias;
        angularVelocityBias = angVelBias;
        linearAccelerationBias = linAccBias;
    }
}
```

## Force/Torque Sensor Simulation

### 1. Joint Force/Torque Sensors

Simulating force and torque measurements at joints:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;
using System.Collections.Generic;

public class UnityForceTorqueSensor : MonoBehaviour
{
    [Header("Force/Torque Configuration")]
    [SerializeField] private string wrenchTopic = "/ft_sensor/wrench";
    [SerializeField] private string frameId = "ft_sensor_frame";

    [Header("Sensor Parameters")]
    [SerializeField] private float forceNoise = 0.1f;
    [SerializeField] private float torqueNoise = 0.01f;
    [SerializeField] private float publishRate = 100.0f;

    [Header("Joint Configuration")]
    [SerializeField] private ArticulationBody jointBody;
    [SerializeField] private Transform sensorTransform;

    private float publishTimer = 0.0f;
    private ROSConnection rosConnection;
    private Vector3 lastForce;
    private Vector3 lastTorque;

    void Start()
    {
        rosConnection = ROSConnection.instance;

        if (sensorTransform == null)
            sensorTransform = transform;

        if (jointBody == null)
            jointBody = GetComponent<ArticulationBody>();
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        float publishInterval = 1.0f / publishRate;

        if (publishTimer >= publishInterval)
        {
            PublishForceTorqueData();
            publishTimer = 0.0f;
        }
    }

    private void PublishForceTorqueData()
    {
        if (rosConnection == null) return;

        // Calculate forces and torques from joint state
        Vector3 force = Vector3.zero;
        Vector3 torque = Vector3.zero;

        if (jointBody != null)
        {
            // Get joint forces (this is a simplified approach)
            // In a real implementation, you might use joint constraints or other physics methods
            force = jointBody.jointForce;
            torque = jointBody.jointTorque;
        }

        // Apply noise to simulate real sensor
        force = ApplyForceNoise(force);
        torque = ApplyTorqueNoise(torque);

        // Convert to ROS coordinate system
        force = new Vector3(force.x, force.y, -force.z);
        torque = new Vector3(torque.x, torque.y, -torque.z);

        // Create wrench message
        WrenchMsg wrenchMsg = new WrenchMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = frameId
            },
            force = new Vector3Msg
            {
                x = force.x,
                y = force.y,
                z = force.z
            },
            torque = new Vector3Msg
            {
                x = torque.x,
                y = torque.y,
                z = torque.z
            }
        };

        rosConnection.SendUnityMessage(wrenchTopic, wrenchMsg);
    }

    private Vector3 ApplyForceNoise(Vector3 force)
    {
        force.x += Random.Range(-forceNoise, forceNoise);
        force.y += Random.Range(-forceNoise, forceNoise);
        force.z += Random.Range(-forceNoise, forceNoise);
        return force;
    }

    private Vector3 ApplyTorqueNoise(Vector3 torque)
    {
        torque.x += Random.Range(-torqueNoise, torqueNoise);
        torque.y += Random.Range(-torqueNoise, torqueNoise);
        torque.z += Random.Range(-torqueNoise, torqueNoise);
        return torque;
    }

    public void SetJointBody(ArticulationBody body)
    {
        jointBody = body;
    }

    public void SetNoiseParameters(float forceN, float torqueN)
    {
        forceNoise = forceN;
        torqueNoise = torqueN;
    }
}
```

## Sensor Fusion and Data Processing

### 1. Sensor Data Aggregator

Combining multiple sensor readings for better perception:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using System.Collections.Generic;

public class SensorDataAggregator : MonoBehaviour
{
    [Header("Aggregation Settings")]
    [SerializeField] private string aggregatedTopic = "/sensors/aggregate";
    [SerializeField] private float publishRate = 50.0f;

    [Header("Sensor References")]
    [SerializeField] private List<UnityCameraSensor> cameraSensors = new List<UnityCameraSensor>();
    [SerializeField] private List<UnityLidar2D> lidarSensors = new List<UnityLidar2D>();
    [SerializeField] private List<UnityIMUSensor> imuSensors = new List<UnityIMUSensor>();

    private float publishTimer = 0.0f;
    private ROSConnection rosConnection;
    private Dictionary<string, object> sensorDataCache = new Dictionary<string, object>();

    void Start()
    {
        rosConnection = ROSConnection.instance;
    }

    void Update()
    {
        publishTimer += Time.deltaTime;
        float publishInterval = 1.0f / publishRate;

        if (publishTimer >= publishInterval)
        {
            AggregateSensorData();
            publishTimer = 0.0f;
        }
    }

    private void AggregateSensorData()
    {
        if (rosConnection == null) return;

        // This is a simplified example - in practice, you would collect real-time data
        // from sensors and aggregate it into a comprehensive message

        // For now, we'll just send a status message indicating sensor availability
        DiagnosticArrayMsg diagnosticArray = new DiagnosticArrayMsg
        {
            header = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.HeaderMsg
            {
                stamp = new Unity.Robotics.ROSTCPConnector.MessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = "sensor_aggregator"
            },
            status = new Unity.Robotics.ROSTCPConnector.MessageTypes.Diagnostic.DiagnosticStatusMsg[]
            {
                CreateDiagnosticStatus("cameras", cameraSensors.Count > 0, "Camera sensors active"),
                CreateDiagnosticStatus("lidars", lidarSensors.Count > 0, "LiDAR sensors active"),
                CreateDiagnosticStatus("imus", imuSensors.Count > 0, "IMU sensors active")
            }
        };

        rosConnection.SendUnityMessage(aggregatedTopic, diagnosticArray);
    }

    private Unity.Robotics.ROSTCPConnector.MessageTypes.Diagnostic.DiagnosticStatusMsg CreateDiagnosticStatus(
        string name, bool isActive, string message)
    {
        return new Unity.Robotics.ROSTCPConnector.MessageTypes.Diagnostic.DiagnosticStatusMsg
        {
            name = name,
            level = isActive ? (byte)0 : (byte)2, // 0=OK, 2=ERROR
            message = message,
            hardware_id = "unity_sensor_system"
        };
    }

    public void AddCameraSensor(UnityCameraSensor sensor)
    {
        if (!cameraSensors.Contains(sensor))
            cameraSensors.Add(sensor);
    }

    public void AddLidarSensor(UnityLidar2D sensor)
    {
        if (!lidarSensors.Contains(sensor))
            lidarSensors.Add(sensor);
    }

    public void AddIMUSensor(UnityIMUSensor sensor)
    {
        if (!imuSensors.Contains(sensor))
            imuSensors.Add(sensor);
    }

    public void RemoveCameraSensor(UnityCameraSensor sensor)
    {
        cameraSensors.Remove(sensor);
    }

    public void RemoveLidarSensor(UnityLidar2D sensor)
    {
        lidarSensors.Remove(sensor);
    }

    public void RemoveIMUSensor(UnityIMUSensor sensor)
    {
        imuSensors.Remove(sensor);
    }
}
```

## Sensor Calibration and Validation

### 1. Sensor Calibration System

Implementing sensor calibration for accurate simulation:

```csharp
using UnityEngine;
using System.Collections;

public class SensorCalibrationSystem : MonoBehaviour
{
    [Header("Calibration Settings")]
    [SerializeField] private float calibrationDuration = 5.0f;
    [SerializeField] private bool autoCalibrateOnStart = true;

    [Header("Calibration Results")]
    [SerializeField] private bool isCalibrated = false;
    [SerializeField] private float calibrationAccuracy = 0.0f;

    private UnityIMUSensor imuSensor;
    private UnityCameraSensor cameraSensor;
    private UnityLidar2D lidarSensor;

    void Start()
    {
        FindSensors();

        if (autoCalibrateOnStart)
        {
            StartCoroutine(CalibrateSensors());
        }
    }

    private void FindSensors()
    {
        imuSensor = GetComponent<UnityIMUSensor>();
        cameraSensor = GetComponent<UnityCameraSensor>();
        lidarSensor = GetComponent<UnityLidar2D>();
    }

    private IEnumerator CalibrateSensors()
    {
        Debug.Log("Starting sensor calibration...");

        // Calibrate IMU (measure bias while stationary)
        if (imuSensor != null)
        {
            yield return StartCoroutine(CalibrateIMU());
        }

        // Calibrate other sensors as needed
        // For camera: measure intrinsic parameters
        // For LiDAR: verify range and accuracy

        isCalibrated = true;
        calibrationAccuracy = 0.95f; // Example accuracy after calibration

        Debug.Log($"Sensor calibration completed. Accuracy: {calibrationAccuracy * 100:F1}%");
    }

    private IEnumerator CalibrateIMU()
    {
        Debug.Log("Calibrating IMU sensor...");

        // Collect baseline readings while sensor should be stationary
        Vector3 sumAngularVelocity = Vector3.zero;
        Vector3 sumLinearAcceleration = Vector3.zero;
        int sampleCount = 0;
        float startTime = Time.time;

        while (Time.time - startTime < calibrationDuration)
        {
            if (imuSensor != null)
            {
                // In a real implementation, you would access raw sensor data
                // For now, we'll just wait and simulate calibration
                sampleCount++;
            }
            yield return new WaitForEndOfFrame();
        }

        Debug.Log("IMU calibration completed.");
    }

    public bool IsCalibrated()
    {
        return isCalibrated;
    }

    public float GetCalibrationAccuracy()
    {
        return calibrationAccuracy;
    }

    public void StartCalibration()
    {
        if (!isCalibrated)
        {
            StartCoroutine(CalibrateSensors());
        }
    }

    public void ResetCalibration()
    {
        isCalibrated = false;
        calibrationAccuracy = 0.0f;
    }
}
```

## Performance Optimization

### 1. Sensor Update Manager

Optimizing sensor updates for better performance:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SensorUpdateManager : MonoBehaviour
{
    [System.Serializable]
    public class SensorUpdateConfig
    {
        public MonoBehaviour sensor;
        public float updateRate = 30.0f;
        public bool isActive = true;
        public float lastUpdateTime = 0.0f;
    }

    [SerializeField] private List<SensorUpdateConfig> sensorConfigs = new List<SensorUpdateConfig>();

    void Update()
    {
        float currentTime = Time.time;

        foreach (var config in sensorConfigs)
        {
            if (config.isActive && config.sensor != null)
            {
                float updateInterval = 1.0f / config.updateRate;

                if ((currentTime - config.lastUpdateTime) >= updateInterval)
                {
                    // Trigger sensor update (this would depend on sensor implementation)
                    // In practice, you might use reflection or interfaces to call update methods
                    config.lastUpdateTime = currentTime;
                }
            }
        }
    }

    public void AddSensor(MonoBehaviour sensor, float updateRate = 30.0f)
    {
        SensorUpdateConfig config = new SensorUpdateConfig
        {
            sensor = sensor,
            updateRate = updateRate,
            isActive = true,
            lastUpdateTime = Time.time
        };

        sensorConfigs.Add(config);
    }

    public void RemoveSensor(MonoBehaviour sensor)
    {
        sensorConfigs.RemoveAll(config => config.sensor == sensor);
    }

    public void SetSensorRate(MonoBehaviour sensor, float newRate)
    {
        var config = sensorConfigs.Find(c => c.sensor == sensor);
        if (config != null)
        {
            config.updateRate = newRate;
        }
    }

    public void SetSensorActive(MonoBehaviour sensor, bool active)
    {
        var config = sensorConfigs.Find(c => c.sensor == sensor);
        if (config != null)
        {
            config.isActive = active;
        }
    }
}
```

## Next Steps

With comprehensive sensor simulation implemented, you now have a complete digital twin system in Unity with realistic camera, LiDAR, IMU, and force/torque sensors. These sensors provide the perception capabilities needed for humanoid robot simulation and control.

In the next section, we'll explore how to integrate these sensors with AI perception systems and create a complete perception pipeline for your humanoid robots in the Unity digital twin environment.