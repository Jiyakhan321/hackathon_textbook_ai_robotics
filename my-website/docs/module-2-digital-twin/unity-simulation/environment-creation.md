---
sidebar_position: 2
---

# Environment Creation for Humanoid Robots in Unity

## Overview

Creating realistic environments in Unity is essential for effective humanoid robot simulation. Unlike Gazebo, Unity offers superior graphics capabilities and more flexible environment design, making it ideal for computer vision training, complex scene simulation, and visually rich testing scenarios. This section covers creating environments specifically designed for humanoid robot interaction.

## Unity Scene Architecture for Robotics

### 1. Recommended Scene Structure

For humanoid robot simulation, organize your scene with the following hierarchy:

```
Main Scene
├── Environment
│   ├── GroundPlane
│   ├── StaticObstacles
│   ├── InteractiveObjects
│   └── Waypoints
├── HumanoidRobot
│   ├── RobotBase
│   ├── Sensors
│   └── Controllers
├── Lighting
│   ├── DirectionalLight (sun)
│   ├── PointLights
│   └── ReflectionProbes
├── Physics
│   ├── PhysicsMaterials
│   └── CollisionLayers
└── Management
    ├── ROSConnectionManager
    ├── SimulationController
    └── DebugTools
```

### 2. Physics Layer Setup

Configure Unity's physics layers for humanoid robot interaction:

```csharp
// PhysicsLayerManager.cs - Manage collision layers for humanoid simulation

using UnityEngine;

public class PhysicsLayerManager : MonoBehaviour
{
    [Header("Physics Layer Configuration")]
    public LayerMask RobotLayer = 1 << 8;    // Layer 8: Robot
    public LayerMask EnvironmentLayer = 1 << 9; // Layer 9: Environment
    public LayerMask ObstacleLayer = 1 << 10;    // Layer 10: Obstacles
    public LayerMask SensorLayer = 1 << 11;      // Layer 11: Sensors
    public LayerMask InteractionLayer = 1 << 12; // Layer 12: Interactive objects

    [Header("Collision Matrix")]
    public bool RobotCollidesWithEnvironment = true;
    public bool RobotCollidesWithObstacles = true;
    public bool RobotCollidesWithSelf = false; // Usually false to prevent self-collision issues
    public bool RobotSensorsCollideWithEnvironment = true;
    public bool RobotSensorsCollideWithObstacles = true;

    void Start()
    {
        ConfigurePhysicsLayers();
    }

    void ConfigurePhysicsLayers()
    {
        // Set up layer names (requires manual setup in Unity editor first)
        SetupCollisionMatrix();
    }

    void SetupCollisionMatrix()
    {
        // Robot vs Environment
        Physics.IgnoreLayerCollision(8, 9, !RobotCollidesWithEnvironment);

        // Robot vs Obstacles
        Physics.IgnoreLayerCollision(8, 10, !RobotCollidesWithObstacles);

        // Robot vs Self (usually ignored to prevent self-collision issues)
        Physics.IgnoreLayerCollision(8, 8, !RobotCollidesWithSelf);

        // Sensors vs Environment
        Physics.IgnoreLayerCollision(11, 9, !RobotSensorsCollideWithEnvironment);

        // Sensors vs Obstacles
        Physics.IgnoreLayerCollision(11, 10, !RobotSensorsCollideWithObstacles);

        Debug.Log("Physics collision matrix configured for humanoid robot");
    }
}
```

## Creating Humanoid-Friendly Environments

### 1. Indoor Environment

Let's create an indoor environment suitable for humanoid robot testing:

```csharp
// IndoorEnvironment.cs - Generate indoor environment components

using UnityEngine;
using System.Collections.Generic;

public class IndoorEnvironment : MonoBehaviour
{
    [Header("Environment Dimensions")]
    public Vector3 roomSize = new Vector3(20f, 8f, 20f);
    public float wallThickness = 0.5f;
    public float wallHeight = 4f;

    [Header("Floor Settings")]
    public PhysicMaterial floorMaterial;
    public Color floorColor = Color.gray;

    [Header("Furniture Configuration")]
    public float tableHeight = 0.8f;
    public float tableSize = 1.5f;
    public int numTables = 3;

    [Header("Obstacle Settings")]
    public int numObstacles = 5;
    public float obstacleMinSize = 0.5f;
    public float obstacleMaxSize = 1.5f;

    [Header("Door Settings")]
    public float doorWidth = 1.0f;
    public float doorHeight = 2.0f;

    private List<GameObject> spawnedObjects = new List<GameObject>();

    [ContextMenu("Generate Environment")]
    public void GenerateEnvironment()
    {
        ClearEnvironment();
        CreateRoom();
        CreateFloor();
        CreateFurniture();
        CreateObstacles();
        CreateDoors();
        CreateWaypoints();
    }

    void ClearEnvironment()
    {
        foreach (GameObject obj in spawnedObjects)
        {
            if (obj != null)
                DestroyImmediate(obj);
        }
        spawnedObjects.Clear();
    }

    void CreateRoom()
    {
        // Create walls using the room size
        Vector3 center = transform.position;

        // Front wall
        CreateWall(center + new Vector3(0, wallHeight / 2, -roomSize.z / 2),
                  new Vector3(roomSize.x, wallHeight, wallThickness));

        // Back wall
        CreateWall(center + new Vector3(0, wallHeight / 2, roomSize.z / 2),
                  new Vector3(roomSize.x, wallHeight, wallThickness));

        // Left wall
        CreateWall(center + new Vector3(-roomSize.x / 2, wallHeight / 2, 0),
                  new Vector3(wallThickness, wallHeight, roomSize.z));

        // Right wall
        CreateWall(center + new Vector3(roomSize.x / 2, wallHeight / 2, 0),
                  new Vector3(wallThickness, wallHeight, roomSize.z));
    }

    GameObject CreateWall(Vector3 position, Vector3 size)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.name = "Wall";
        wall.transform.position = position;
        wall.transform.localScale = size;
        wall.layer = 9; // Environment layer

        // Remove collider if it has one and add custom physics properties
        DestroyImmediate(wall.GetComponent<BoxCollider>());
        BoxCollider newCollider = wall.AddComponent<BoxCollider>();
        newCollider.material = floorMaterial;

        // Make static for optimization
        wall.isStatic = true;

        spawnedObjects.Add(wall);
        return wall;
    }

    void CreateFloor()
    {
        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.position = transform.position + new Vector3(0, -0.1f, 0); // Slightly below to avoid z-fighting
        floor.transform.localScale = new Vector3(roomSize.x / 10f, 1, roomSize.z / 10f); // Plane primitive is 10x10 units
        floor.layer = 9; // Environment layer

        // Apply material
        Renderer floorRenderer = floor.GetComponent<Renderer>();
        if (floorRenderer != null)
        {
            floorRenderer.material.color = floorColor;
        }

        // Set physics material
        floor.GetComponent<Collider>().material = floorMaterial;

        // Make static for optimization
        floor.isStatic = true;

        spawnedObjects.Add(floor);
    }

    void CreateFurniture()
    {
        for (int i = 0; i < numTables; i++)
        {
            // Random position within room bounds
            Vector3 tablePos = new Vector3(
                Random.Range(-roomSize.x / 3, roomSize.x / 3),
                tableHeight / 2,
                Random.Range(-roomSize.z / 3, roomSize.z / 3)
            ) + (Vector3)transform.position;

            GameObject table = CreateTable(tablePos, tableSize);
            table.layer = 10; // Obstacle layer
            spawnedObjects.Add(table);
        }
    }

    GameObject CreateTable(Vector3 position, float size)
    {
        // Create table top
        GameObject tableTop = GameObject.CreatePrimitive(PrimitiveType.Cube);
        tableTop.name = "TableTop";
        tableTop.transform.position = position;
        tableTop.transform.localScale = new Vector3(size, 0.1f, size);
        tableTop.layer = 10;

        // Create table legs
        float legHeight = tableHeight - 0.05f; // 5cm gap between top and floor
        float legSize = 0.1f;
        float legOffset = size / 2 - 0.1f;

        // Four legs
        CreateLeg(tableTop.transform, new Vector3(-legOffset, -legHeight / 2, -legOffset), legSize, legHeight);
        CreateLeg(tableTop.transform, new Vector3(legOffset, -legHeight / 2, -legOffset), legSize, legHeight);
        CreateLeg(tableTop.transform, new Vector3(-legOffset, -legHeight / 2, legOffset), legSize, legHeight);
        CreateLeg(tableTop.transform, new Vector3(legOffset, -legHeight / 2, legOffset), legSize, legHeight);

        // Make static for optimization
        tableTop.isStatic = true;

        return tableTop;
    }

    GameObject CreateLeg(Transform parent, Vector3 localPos, float size, float height)
    {
        GameObject leg = GameObject.CreatePrimitive(PrimitiveType.Cube);
        leg.name = "TableLeg";
        leg.transform.SetParent(parent);
        leg.transform.localPosition = localPos;
        leg.transform.localScale = new Vector3(size, height, size);
        leg.layer = 10;

        // Remove the default collider and add a proper one
        DestroyImmediate(leg.GetComponent<Collider>());
        BoxCollider collider = leg.AddComponent<BoxCollider>();
        collider.material = floorMaterial;

        return leg;
    }

    void CreateObstacles()
    {
        for (int i = 0; i < numObstacles; i++)
        {
            // Random position within room
            Vector3 obstaclePos = new Vector3(
                Random.Range(-roomSize.x / 2 + 1, roomSize.x / 2 - 1),
                Random.Range(0.2f, 2f), // Height off ground
                Random.Range(-roomSize.z / 2 + 1, roomSize.z / 2 - 1)
            ) + (Vector3)transform.position;

            float obstacleSize = Random.Range(obstacleMinSize, obstacleMaxSize);

            GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
            obstacle.name = $"Obstacle_{i}";
            obstacle.transform.position = obstaclePos;
            obstacle.transform.localScale = new Vector3(obstacleSize, obstacleSize, obstacleSize);
            obstacle.layer = 10; // Obstacle layer

            // Add physics material
            obstacle.GetComponent<Collider>().material = floorMaterial;

            // Add some visual variety
            Renderer rend = obstacle.GetComponent<Renderer>();
            if (rend != null)
            {
                rend.material.color = Random.ColorHSV(0f, 1f, 0.5f, 1f, 0.5f, 1f);
            }

            spawnedObjects.Add(obstacle);
        }
    }

    void CreateDoors()
    {
        // Create a simple door opening in the front wall
        Vector3 doorPos = new Vector3(transform.position.x, doorHeight / 2, transform.position.z - roomSize.z / 2);

        // Create a frame around the door opening (not the actual door)
        CreateDoorFrame(doorPos);
    }

    void CreateDoorFrame(Vector3 position)
    {
        float frameThickness = 0.2f;
        float frameDepth = wallThickness;

        // Top frame
        GameObject topFrame = GameObject.CreatePrimitive(PrimitiveType.Cube);
        topFrame.name = "DoorFrame_Top";
        topFrame.transform.position = position + new Vector3(0, doorHeight + frameThickness / 2, frameThickness / 2);
        topFrame.transform.localScale = new Vector3(doorWidth, frameThickness, frameDepth);
        topFrame.layer = 9;
        topFrame.isStatic = true;
        spawnedObjects.Add(topFrame);

        // Side frames
        GameObject leftFrame = GameObject.CreatePrimitive(PrimitiveType.Cube);
        leftFrame.name = "DoorFrame_Left";
        leftFrame.transform.position = position + new Vector3(-doorWidth / 2 - frameThickness / 2, doorHeight / 2, frameThickness / 2);
        leftFrame.transform.localScale = new Vector3(frameThickness, doorHeight, frameDepth);
        leftFrame.layer = 9;
        leftFrame.isStatic = true;
        spawnedObjects.Add(leftFrame);

        GameObject rightFrame = GameObject.CreatePrimitive(PrimitiveType.Cube);
        rightFrame.name = "DoorFrame_Right";
        rightFrame.transform.position = position + new Vector3(doorWidth / 2 + frameThickness / 2, doorHeight / 2, frameThickness / 2);
        rightFrame.transform.localScale = new Vector3(frameThickness, doorHeight, frameDepth);
        rightFrame.layer = 9;
        rightFrame.isStatic = true;
        spawnedObjects.Add(rightFrame);
    }

    void CreateWaypoints()
    {
        // Create waypoints for navigation testing
        for (int i = 0; i < 5; i++)
        {
            GameObject waypoint = new GameObject($"Waypoint_{i}");
            waypoint.transform.position = new Vector3(
                Random.Range(-roomSize.x / 4, roomSize.x / 4),
                0.1f, // Slightly above ground
                Random.Range(-roomSize.z / 4, roomSize.z / 4)
            ) + (Vector3)transform.position;

            // Add visual indicator
            GameObject indicator = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            indicator.transform.SetParent(waypoint.transform);
            indicator.transform.localPosition = Vector3.zero;
            indicator.transform.localScale = Vector3.one * 0.2f;

            Renderer indicatorRenderer = indicator.GetComponent<Renderer>();
            if (indicatorRenderer != null)
            {
                indicatorRenderer.material.color = Color.yellow;
            }

            // Make non-physical
            DestroyImmediate(indicator.GetComponent<Collider>());

            waypoint.layer = 12; // Interaction layer
            spawnedObjects.Add(waypoint);
        }
    }
}
```

### 2. Outdoor Environment

For outdoor humanoid testing with more complex terrain:

```csharp
// OutdoorEnvironment.cs - Generate outdoor environment

using UnityEngine;
using System.Collections.Generic;

public class OutdoorEnvironment : MonoBehaviour
{
    [Header("Terrain Settings")]
    public int terrainWidth = 100;
    public int terrainLength = 100;
    public float terrainHeight = 20f;

    [Header("Ground Material")]
    public PhysicMaterial groundMaterial;
    public Texture2D groundTexture;

    [Header("Obstacle Configuration")]
    public int numTrees = 20;
    public int numRocks = 15;
    public int numSlopes = 5;

    [Header("Path Settings")]
    public bool createPath = true;
    public float pathWidth = 2.0f;
    public float pathDepth = 0.1f;

    [ContextMenu("Generate Outdoor Environment")]
    public void GenerateEnvironment()
    {
        CreateTerrain();
        CreateVegetation();
        CreateNaturalObstacles();
        CreateSlopes();
        if (createPath)
            CreatePath();
    }

    void CreateTerrain()
    {
        // Create a simple flat terrain
        GameObject terrain = GameObject.CreatePrimitive(PrimitiveType.Plane);
        terrain.name = "Terrain";
        terrain.transform.position = transform.position;
        terrain.transform.localScale = new Vector3(terrainWidth / 10f, 1, terrainLength / 10f);
        terrain.layer = 9; // Environment layer

        // Apply ground material
        Renderer terrainRenderer = terrain.GetComponent<Renderer>();
        if (terrainRenderer != null && groundTexture != null)
        {
            terrainRenderer.material.mainTexture = groundTexture;
            terrainRenderer.material.color = Color.green;
        }

        // Apply physics material
        terrain.GetComponent<Collider>().material = groundMaterial;

        // Make static for optimization
        terrain.isStatic = true;
    }

    void CreateVegetation()
    {
        for (int i = 0; i < numTrees; i++)
        {
            Vector3 treePos = new Vector3(
                Random.Range(-terrainWidth / 2 + 5, terrainWidth / 2 - 5),
                0,
                Random.Range(-terrainLength / 2 + 5, terrainLength / 2 - 5)
            ) + (Vector3)transform.position;

            CreateTree(treePos);
        }
    }

    void CreateTree(Vector3 position)
    {
        GameObject tree = new GameObject("Tree");
        tree.transform.position = position;
        tree.layer = 10; // Obstacle layer

        // Trunk
        GameObject trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        trunk.name = "Trunk";
        trunk.transform.SetParent(tree.transform);
        trunk.transform.localPosition = Vector3.zero;
        trunk.transform.localScale = new Vector3(0.3f, 2f, 0.3f);
        trunk.transform.Translate(0, 1f, 0); // Center the cylinder properly

        // Leaves (simplified)
        GameObject leaves = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        leaves.name = "Leaves";
        leaves.transform.SetParent(tree.transform);
        leaves.transform.localPosition = new Vector3(0, 2.5f, 0);
        leaves.transform.localScale = Vector3.one * 1.5f;

        // Remove colliders from decorative parts
        DestroyImmediate(leaves.GetComponent<Collider>());
        // Keep trunk collider for navigation challenges
        trunk.GetComponent<Collider>().material = groundMaterial;

        tree.isStatic = true;
    }

    void CreateNaturalObstacles()
    {
        for (int i = 0; i < numRocks; i++)
        {
            Vector3 rockPos = new Vector3(
                Random.Range(-terrainWidth / 3, terrainWidth / 3),
                0.2f, // Slightly above ground
                Random.Range(-terrainLength / 3, terrainLength / 3)
            ) + (Vector3)transform.position;

            CreateRock(rockPos, Random.Range(0.5f, 2f));
        }
    }

    void CreateRock(Vector3 position, float size)
    {
        GameObject rock = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        rock.name = "Rock";
        rock.transform.position = position;
        rock.transform.localScale = Vector3.one * size;
        rock.layer = 10; // Obstacle layer

        // Apply rock-like appearance
        Renderer rockRenderer = rock.GetComponent<Renderer>();
        if (rockRenderer != null)
        {
            rockRenderer.material.color = new Color(0.4f, 0.4f, 0.4f);
        }

        // Apply physics properties
        rock.GetComponent<Collider>().material = groundMaterial;

        // Make static for optimization
        rock.isStatic = true;
    }

    void CreateSlopes()
    {
        for (int i = 0; i < numSlopes; i++)
        {
            Vector3 slopePos = new Vector3(
                Random.Range(-terrainWidth / 4, terrainWidth / 4),
                0,
                Random.Range(-terrainLength / 4, terrainLength / 4)
            ) + (Vector3)transform.position;

            CreateSlope(slopePos, Random.Range(5f, 15f)); // Angle in degrees
        }
    }

    void CreateSlope(Vector3 position, float angle)
    {
        GameObject slope = GameObject.CreatePrimitive(PrimitiveType.Cube);
        slope.name = "Slope";
        slope.transform.position = position;
        slope.transform.localScale = new Vector3(5f, 0.1f, 3f);
        slope.transform.rotation = Quaternion.Euler(angle, Random.Range(0, 360), 0);
        slope.layer = 9; // Environment layer

        // Apply physics material
        slope.GetComponent<Collider>().material = groundMaterial;

        // Make static for optimization
        slope.isStatic = true;
    }

    void CreatePath()
    {
        // Create a winding path through the environment
        int pathSegments = 8;
        Vector3 startPos = new Vector3(-terrainWidth / 3, 0.01f, -terrainLength / 3) + (Vector3)transform.position;

        for (int i = 0; i < pathSegments; i++)
        {
            GameObject pathSegment = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pathSegment.name = $"Path_Segment_{i}";
            pathSegment.transform.position = startPos + new Vector3(
                i * 5f,
                0.01f, // Slightly above ground
                Random.Range(-2f, 2f) // Slight variation
            );
            pathSegment.transform.localScale = new Vector3(pathWidth, pathDepth, 4f);
            pathSegment.layer = 9; // Environment layer

            // Apply path material (different from ground)
            Renderer pathRenderer = pathSegment.GetComponent<Renderer>();
            if (pathRenderer != null)
            {
                pathRenderer.material.color = Color.gray;
            }

            // Apply physics properties
            pathSegment.GetComponent<Collider>().material = groundMaterial;

            // Make static for optimization
            pathSegment.isStatic = true;
        }
    }
}
```

## Physics Material Configuration

### 1. Creating Physics Materials

Physics materials are crucial for realistic humanoid interaction:

```csharp
// PhysicsMaterialManager.cs - Create and manage physics materials

using UnityEngine;

[CreateAssetMenu(fileName = "PhysicsMaterialConfig", menuName = "Robotics/Physics Material Config")]
public class PhysicsMaterialConfig : ScriptableObject
{
    [Header("Ground Materials")]
    public PhysicMaterial concrete;
    public PhysicMaterial grass;
    public PhysicMaterial wood;
    public PhysicMaterial carpet;

    [Header("Friction Settings")]
    [Tooltip("Higher values = more grip, better for walking")]
    public float highFriction = 0.8f;
    [Tooltip("Lower values = less grip, more challenging for humanoid")]
    public float lowFriction = 0.1f;

    [Header("Bounciness Settings")]
    public float lowBounce = 0.1f;
    public float highBounce = 0.5f;

    [Header("Custom Materials")]
    public PhysicMaterial[] customMaterials;

    [ContextMenu("Generate Physics Materials")]
    public void GenerateMaterials()
    {
        // Create standard materials
        concrete = CreatePhysicsMaterial("Concrete", highFriction, lowBounce);
        grass = CreatePhysicsMaterial("Grass", highFriction * 0.7f, lowBounce * 0.5f);
        wood = CreatePhysicsMaterial("Wood", highFriction * 0.9f, lowBounce * 0.8f);
        carpet = CreatePhysicsMaterial("Carpet", highFriction * 1.2f, lowBounce * 0.3f);

        Debug.Log("Physics materials generated successfully");
    }

    PhysicMaterial CreatePhysicsMaterial(string name, float friction, float bounce)
    {
        PhysicMaterial material = new PhysicMaterial(name);
        material.staticFriction = friction;
        material.dynamicFriction = friction * 0.9f; // Dynamic friction is usually slightly lower
        material.bounciness = bounce;
        material.frictionCombine = PhysicMaterialCombine.Maximum;
        material.bounceCombine = PhysicMaterialCombine.Average;

        return material;
    }
}
```

### 2. Terrain and Surface Variations

Create surfaces with different properties for humanoid training:

```csharp
// SurfaceVarietyManager.cs - Create varied surfaces for training

using UnityEngine;
using System.Collections.Generic;

public class SurfaceVarietyManager : MonoBehaviour
{
    [Header("Surface Configuration")]
    public PhysicsMaterialConfig materialConfig;

    [Header("Surface Zones")]
    public int zonesPerEnvironment = 6;
    public float zoneSize = 3f;

    [Header("Surface Types")]
    public string[] surfaceNames = { "Concrete", "Grass", "Wood", "Carpet", "Ice", "Sand" };
    public PhysicMaterial[] surfaceMaterials;

    void Start()
    {
        if (materialConfig != null)
        {
            surfaceMaterials = new PhysicMaterial[] {
                materialConfig.concrete,
                materialConfig.grass,
                materialConfig.wood,
                materialConfig.carpet,
                CreateIceMaterial(),
                CreateSandMaterial()
            };
        }
    }

    [ContextMenu("Create Surface Zones")]
    public void CreateSurfaceZones()
    {
        for (int i = 0; i < zonesPerEnvironment; i++)
        {
            CreateSurfaceZone(i);
        }
    }

    void CreateSurfaceZone(int index)
    {
        Vector3 position = transform.position + new Vector3(
            (index % 3) * (zoneSize + 1),
            0,
            (index / 3) * (zoneSize + 1)
        );

        GameObject zone = GameObject.CreatePrimitive(PrimitiveType.Cube);
        zone.name = $"{surfaceNames[index % surfaceNames.Length]}_Zone";
        zone.transform.position = position;
        zone.transform.localScale = new Vector3(zoneSize, 0.1f, zoneSize);
        zone.layer = 9; // Environment layer

        // Apply appropriate material
        if (index < surfaceMaterials.Length)
        {
            zone.GetComponent<Collider>().material = surfaceMaterials[index];

            // Apply visual color coding
            Renderer rend = zone.GetComponent<Renderer>();
            if (rend != null)
            {
                switch (index % surfaceNames.Length)
                {
                    case 0: rend.material.color = Color.gray; break;      // Concrete
                    case 1: rend.material.color = Color.green; break;     // Grass
                    case 2: rend.material.color = Color.yellow; break;    // Wood
                    case 3: rend.material.color = Color.red; break;       // Carpet
                    case 4: rend.material.color = Color.cyan; break;      // Ice
                    case 5: rend.material.color = Color.white; break;     // Sand
                }
            }
        }

        // Make static for optimization
        zone.isStatic = true;
    }

    PhysicMaterial CreateIceMaterial()
    {
        PhysicMaterial ice = new PhysicMaterial("Ice");
        ice.staticFriction = 0.1f;
        ice.dynamicFriction = 0.05f;
        ice.bounciness = 0.0f;
        return ice;
    }

    PhysicMaterial CreateSandMaterial()
    {
        PhysicMaterial sand = new PhysicMaterial("Sand");
        sand.staticFriction = 0.8f;
        sand.dynamicFriction = 0.7f;
        sand.bounciness = 0.1f;
        return sand;
    }

    // Method to test robot on different surfaces
    public PhysicMaterial GetRandomSurfaceMaterial()
    {
        if (surfaceMaterials.Length > 0)
        {
            int randomIndex = Random.Range(0, surfaceMaterials.Length);
            return surfaceMaterials[randomIndex];
        }
        return null;
    }
}
```

## Navigation and Waypoint Systems

### 1. Waypoint Manager for Training

```csharp
// WaypointNavigationSystem.cs - Create navigation challenges

using UnityEngine;
using System.Collections.Generic;

public class WaypointNavigationSystem : MonoBehaviour
{
    [Header("Waypoint Configuration")]
    public List<Transform> waypoints = new List<Transform>();
    public Color waypointColor = Color.yellow;
    public float waypointSize = 0.3f;

    [Header("Navigation Challenges")]
    public bool createNarrowPath = true;
    public bool createObstacleCourse = true;
    public bool createElevationChanges = true;

    [Header("Challenge Parameters")]
    public int narrowPathWidth = 1; // Number of waypoints wide
    public float obstacleDensity = 0.3f; // 0-1

    [ContextMenu("Generate Navigation Course")]
    public void GenerateNavigationCourse()
    {
        ClearWaypoints();
        CreateWaypointPath();
        if (createObstacleCourse)
            CreateObstacleCourse();
        if (createElevationChanges)
            CreateElevationChanges();
    }

    void ClearWaypoints()
    {
        foreach (Transform waypoint in waypoints)
        {
            if (waypoint != null)
                DestroyImmediate(waypoint.gameObject);
        }
        waypoints.Clear();
    }

    void CreateWaypointPath()
    {
        // Create a serpentine path
        int numWaypoints = 20;
        float spacing = 2f;

        for (int i = 0; i < numWaypoints; i++)
        {
            Vector3 pos = transform.position + new Vector3(
                Mathf.Sin(i * 0.5f) * 5f, // Create curves
                0.1f, // Above ground
                i * spacing
            );

            GameObject waypointObj = new GameObject($"Waypoint_{i}");
            waypointObj.transform.position = pos;

            // Visual indicator
            GameObject indicator = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            indicator.transform.SetParent(waypointObj.transform);
            indicator.transform.localPosition = Vector3.zero;
            indicator.transform.localScale = Vector3.one * waypointSize;
            indicator.GetComponent<Renderer>().material.color = waypointColor;

            // Make non-physical
            DestroyImmediate(indicator.GetComponent<Collider>());

            waypoints.Add(waypointObj.transform);
        }
    }

    void CreateObstacleCourse()
    {
        // Add obstacles around some waypoints to create challenges
        for (int i = 3; i < waypoints.Count - 3; i += 4) // Every 4th waypoint
        {
            if (Random.value < obstacleDensity)
            {
                CreateObstacleCluster(waypoints[i].position);
            }
        }
    }

    void CreateObstacleCluster(Vector3 centerPos)
    {
        int numObstacles = Random.Range(2, 5);

        for (int i = 0; i < numObstacles; i++)
        {
            Vector3 offset = new Vector3(
                Random.Range(-2f, 2f),
                Random.Range(0.5f, 2f),
                Random.Range(-2f, 2f)
            );

            GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            obstacle.name = "Navigation_Obstacle";
            obstacle.transform.position = centerPos + offset;
            obstacle.transform.localScale = new Vector3(0.5f, 1f, 0.5f);
            obstacle.layer = 10; // Obstacle layer

            // Apply physics properties
            obstacle.GetComponent<Collider>().material = new PhysicMaterial
            {
                staticFriction = 0.7f,
                dynamicFriction = 0.6f
            };

            // Make static
            obstacle.isStatic = true;
        }
    }

    void CreateElevationChanges()
    {
        // Create some elevation changes by moving certain waypoints up/down
        for (int i = 5; i < waypoints.Count; i += 6) // Every 6th waypoint
        {
            Vector3 newPos = waypoints[i].position;
            newPos.y += Random.Range(0.5f, 2f); // Raise this waypoint
            waypoints[i].position = newPos;

            // Create a ramp or step up to this waypoint
            CreateRamp(waypoints[i - 1].position, newPos);
        }
    }

    void CreateRamp(Vector3 startPos, Vector3 endPos)
    {
        Vector3 rampSize = new Vector3(2f, 0.1f, Vector3.Distance(startPos, endPos));
        Vector3 rampPos = startPos + (endPos - startPos) / 2f;
        rampPos.y = (startPos.y + endPos.y) / 2f;

        GameObject ramp = GameObject.CreatePrimitive(PrimitiveType.Cube);
        ramp.name = "Ramp";
        ramp.transform.position = rampPos;

        // Rotate to match the slope
        Vector3 direction = endPos - startPos;
        ramp.transform.rotation = Quaternion.LookRotation(direction, Vector3.up);

        ramp.transform.localScale = rampSize;
        ramp.layer = 9; // Environment layer

        // Apply physics properties
        ramp.GetComponent<Collider>().material = new PhysicMaterial
        {
            staticFriction = 0.8f
        };

        // Make static
        ramp.isStatic = true;
    }

    // Get next waypoint in sequence
    public Transform GetNextWaypoint(int currentIndex)
    {
        if (waypoints.Count > 0)
        {
            return waypoints[(currentIndex + 1) % waypoints.Count];
        }
        return null;
    }

    // Get closest waypoint to a position
    public Transform GetClosestWaypoint(Vector3 position)
    {
        Transform closest = null;
        float closestDistance = float.MaxValue;

        foreach (Transform waypoint in waypoints)
        {
            float distance = Vector3.Distance(position, waypoint.position);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closest = waypoint;
            }
        }

        return closest;
    }
}
```

## Environment Testing and Validation

### 1. Environment Validation Script

```csharp
// EnvironmentValidator.cs - Validate environment for humanoid use

using UnityEngine;

public class EnvironmentValidator : MonoBehaviour
{
    [Header("Validation Settings")]
    public float minimumWalkwayWidth = 0.8f; // Minimum width for humanoid passage
    public float maximumStepHeight = 0.2f;   // Maximum step height humanoid can handle
    public float safeCeilingHeight = 2.5f;   // Minimum ceiling height

    [Header("Validation Results")]
    [TextArea]
    public string validationReport = "";

    [ContextMenu("Validate Environment")]
    public void ValidateEnvironment()
    {
        validationReport = "Environment Validation Report:\n\n";

        // Check walkway widths
        bool walkwaysValid = CheckWalkwayWidths();
        validationReport += $"Walkway Widths: {(walkwaysValid ? "PASS" : "FAIL")}\n";

        // Check step heights
        bool stepsValid = CheckStepHeights();
        validationReport += $"Step Heights: {(stepsValid ? "PASS" : "FAIL")}\n";

        // Check ceiling heights
        bool ceilingValid = CheckCeilingHeights();
        validationReport += $"Ceiling Heights: {(ceilingValid ? "PASS" : "FAIL")}\n";

        // Check for physics issues
        bool physicsValid = CheckPhysicsSetup();
        validationReport += $"Physics Setup: {(physicsValid ? "PASS" : "FAIL")}\n";

        Debug.Log(validationReport);
    }

    bool CheckWalkwayWidths()
    {
        // For simplicity, this checks a few key areas
        // In practice, you'd want more comprehensive pathfinding validation
        Collider[] obstacles = Physics.OverlapBox(
            transform.position,
            new Vector3(20f, 2f, 20f), // Check area around environment
            Quaternion.identity,
            1 << 10 // Only check obstacle layer
        );

        // Check if any obstacles are too narrow for humanoid passage
        foreach (Collider obstacle in obstacles)
        {
            Vector3 size = obstacle.bounds.size;
            if (size.x < minimumWalkwayWidth && size.z < minimumWalkwayWidth)
            {
                validationReport += $"  - Narrow obstacle detected: {obstacle.name}\n";
                return false;
            }
        }

        return true;
    }

    bool CheckStepHeights()
    {
        // Check for steps that are too high
        RaycastHit[] hits = Physics.RaycastAll(
            transform.position + Vector3.up * 10f, // Ray from above
            Vector3.down,
            20f, // Ray distance
            1 << 9 // Environment layer
        );

        // Group hits by x,z position to find height differences
        for (int i = 0; i < hits.Length - 1; i++)
        {
            for (int j = i + 1; j < hits.Length; j++)
            {
                // If points are close horizontally but have large height difference
                float horizontalDistance = Vector2.Distance(
                    new Vector2(hits[i].point.x, hits[i].point.z),
                    new Vector2(hits[j].point.x, hits[j].point.z)
                );

                if (horizontalDistance < 1f) // Close horizontally
                {
                    float heightDiff = Mathf.Abs(hits[i].point.y - hits[j].point.y);
                    if (heightDiff > maximumStepHeight)
                    {
                        validationReport += $"  - Large step detected: {heightDiff:F2}m\n";
                        return false;
                    }
                }
            }
        }

        return true;
    }

    bool CheckCeilingHeights()
    {
        // Check for low ceilings that could hit humanoid head
        RaycastHit hit;
        if (Physics.Raycast(
            transform.position,
            Vector3.up,
            out hit,
            4f, // Check up to 4m
            1 << 9 // Environment layer
        ))
        {
            float ceilingHeight = hit.distance;
            if (ceilingHeight < safeCeilingHeight)
            {
                validationReport += $"  - Low ceiling: {ceilingHeight:F2}m at {hit.point}\n";
                return false;
            }
        }

        return true;
    }

    bool CheckPhysicsSetup()
    {
        // Check if required physics materials are assigned
        if (Physics.GetIgnoreLayerCollision(8, 9)) // Robot vs Environment
        {
            validationReport += "  - Robot may not collide with environment\n";
            return false;
        }

        // Check if static objects are properly marked
        int staticObjects = 0;
        Collider[] allColliders = FindObjectsOfType<Collider>();
        foreach (Collider col in allColliders)
        {
            if (col.transform.root == transform && col.GetComponent<Rigidbody>() == null)
            {
                staticObjects++;
            }
        }

        validationReport += $"  - Found {staticObjects} static objects\n";
        return staticObjects > 0; // Should have some static objects
    }
}
```

## Next Steps

Now that you have created comprehensive environments for humanoid robots in Unity, let's explore how to establish proper communication between Unity and ROS 2 systems. In the next section, we'll cover ROS communication implementation specifically for Unity environments.