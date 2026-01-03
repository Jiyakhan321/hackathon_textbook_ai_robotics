---
sidebar_position: 3
---

# Photorealistic Simulation and Synthetic Data Generation

## Overview

Photorealistic simulation is crucial for developing robust AI systems for humanoid robots. NVIDIA Isaac Sim provides advanced rendering capabilities that enable the generation of synthetic data indistinguishable from real-world data. This section covers creating realistic environments, configuring lighting and materials, and implementing synthetic data generation pipelines for humanoid robotics applications.

## Environment Design for Humanoid Robots

### 1. Realistic Indoor Environments

Creating indoor environments that closely match real-world spaces where humanoid robots operate:

```python
# indoor_environment.py - Create realistic indoor environments
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
import numpy as np

def create_realistic_indoor_environment(world: World):
    """
    Create a realistic indoor environment for humanoid robot testing
    """
    stage = omni.usd.get_context().get_stage()

    # Create main room structure
    create_main_room(stage)

    # Add furniture and objects
    add_furniture(stage)

    # Configure realistic lighting
    configure_realistic_lighting(stage)

    # Add textured surfaces
    add_realistic_materials(stage)

    print("Realistic indoor environment created")

def create_main_room(stage):
    """
    Create a main room with realistic dimensions and features
    """
    # Room dimensions (10m x 8m x 3m)
    room_length = 10.0
    room_width = 8.0
    room_height = 3.0

    # Floor
    floor_path = Sdf.Path("/World/Floor")
    floor_xform = UsdGeom.Xform.Define(stage, floor_path)
    floor_geom = UsdGeom.Cube.Define(stage, floor_path.AppendChild("Geometry"))
    floor_geom.CreateSizeAttr(1.0)
    floor_xform.AddTranslateOp().Set((0, 0, -0.05))
    floor_xform.AddScaleOp().Set((room_length, room_width, 0.1))

    # Walls
    wall_thickness = 0.2
    wall_configs = [
        {"position": (0, room_width/2, room_height/2), "scale": (room_length, wall_thickness, room_height)},
        {"position": (0, -room_width/2, room_height/2), "scale": (room_length, wall_thickness, room_height)},
        {"position": (room_length/2, 0, room_height/2), "scale": (wall_thickness, room_width, room_height)},
        {"position": (-room_length/2, 0, room_height/2), "scale": (wall_thickness, room_width, room_height)},
    ]

    for i, config in enumerate(wall_configs):
        wall_path = Sdf.Path(f"/World/Wall_{i}")
        wall_xform = UsdGeom.Xform.Define(stage, wall_path)
        wall_geom = UsdGeom.Cube.Define(stage, wall_path.AppendChild("Geometry"))
        wall_geom.CreateSizeAttr(1.0)
        wall_xform.AddTranslateOp().Set(config["position"])
        wall_xform.AddScaleOp().Set(config["scale"])

    # Ceiling
    ceiling_path = Sdf.Path("/World/Ceiling")
    ceiling_xform = UsdGeom.Xform.Define(stage, ceiling_path)
    ceiling_geom = UsdGeom.Cube.Define(stage, ceiling_path.AppendChild("Geometry"))
    ceiling_geom.CreateSizeAttr(1.0)
    ceiling_xform.AddTranslateOp().Set((0, 0, room_height))
    ceiling_xform.AddScaleOp().Set((room_length, room_width, 0.1))

def add_furniture(stage):
    """
    Add realistic furniture for humanoid interaction
    """
    furniture_configs = [
        # Dining table
        {
            "name": "DiningTable",
            "type": "cube",
            "position": (2, -2, 0.75),
            "scale": (1.5, 0.8, 0.8),
            "color": (0.8, 0.6, 0.4)  # Wood color
        },
        # Chair
        {
            "name": "Chair",
            "type": "cube",
            "position": (2, -3.2, 0.45),
            "scale": (0.5, 0.5, 0.9),
            "color": (0.3, 0.3, 0.3)  # Dark color
        },
        # Sofa
        {
            "name": "Sofa",
            "type": "cube",
            "position": (-3, 2, 0.4),
            "scale": (2.0, 0.8, 0.8),
            "color": (0.2, 0.2, 0.6)  # Blue color
        },
        # Coffee table
        {
            "name": "CoffeeTable",
            "type": "cube",
            "position": (-1, 0, 0.45),
            "scale": (0.8, 0.8, 0.9),
            "color": (0.7, 0.5, 0.3)  # Wood color
        },
        # Kitchen counter
        {
            "name": "KitchenCounter",
            "type": "cube",
            "position": (4, 3, 0.9),
            "scale": (2.0, 0.6, 0.9),
            "color": (0.9, 0.9, 0.9)  # White color
        }
    ]

    for i, config in enumerate(furniture_configs):
        furn_path = Sdf.Path(f"/World/Furniture_{config['name']}_{i}")
        furn_xform = UsdGeom.Xform.Define(stage, furn_path)
        furn_geom = UsdGeom.Cube.Define(stage, furn_path.AppendChild("Geometry"))
        furn_geom.CreateSizeAttr(1.0)
        furn_xform.AddTranslateOp().Set(config["position"])
        furn_xform.AddScaleOp().Set(config["scale"])

def configure_realistic_lighting(stage):
    """
    Configure realistic indoor lighting with multiple light sources
    """
    # Main ceiling light (over dining table area)
    main_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/MainLight"))
    main_light.CreateIntensityAttr(3000)
    main_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.94))  # Warm white
    main_light.AddRotateXYZOp().Set((60, 0, 0))  # Angle from ceiling

    # Ambient light to fill shadows
    ambient_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/AmbientLight"))
    ambient_light.CreateIntensityAttr(200)
    ambient_light.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))  # Cool white ambient

    # Task lighting (over kitchen counter)
    task_light = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/TaskLight"))
    task_light.CreateIntensityAttr(1000)
    task_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.98))
    task_light.AddTranslateOp().Set((4, 3, 2.0))

    # Window light simulation (if needed)
    window_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/WindowLight"))
    window_light.CreateIntensityAttr(1000)
    window_light.CreateColorAttr(Gf.Vec3f(0.95, 1.0, 1.0))
    window_light.AddRotateXYZOp().Set((30, 45, 0))

def add_realistic_materials(stage):
    """
    Add realistic materials to surfaces for better rendering
    """
    # Create material paths
    material_configs = [
        {"name": "FloorMaterial", "color": (0.7, 0.7, 0.7), "roughness": 0.8},
        {"name": "WallMaterial", "color": (0.95, 0.95, 0.95), "roughness": 0.9},
        {"name": "WoodMaterial", "color": (0.8, 0.6, 0.4), "roughness": 0.3},
        {"name": "MetalMaterial", "color": (0.7, 0.7, 0.8), "roughness": 0.1},
    ]

    for config in material_configs:
        material_path = Sdf.Path(f"/World/Materials/{config['name']}")
        material = UsdShade.Material.Define(stage, material_path)

        # Create PBR shader
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
        shader.CreateIdAttr("OmniPBR")

        # Set material properties
        shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*config["color"])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            config["roughness"]
        )

        # Connect shader to material
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")
```

### 2. Dynamic Environment Features

Creating environments that can change for varied training data:

```python
# dynamic_environment.py - Create environments with dynamic elements
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, Sdf, UsdGeom
import numpy as np
import random

class DynamicEnvironmentManager:
    """
    Manager for dynamic elements in simulation environments
    """
    def __init__(self, world: World):
        self.world = world
        self.stage = omni.usd.get_context().get_stage()
        self.dynamic_objects = []
        self.lighting_conditions = []

    def add_dynamic_objects(self):
        """
        Add objects that can be moved/changed during simulation
        """
        # Random objects that can be placed differently each time
        object_types = [
            {"name": "Box", "size": (0.2, 0.2, 0.2), "color": (1.0, 0.0, 0.0)},
            {"name": "Sphere", "size": (0.15, 0.15, 0.15), "color": (0.0, 1.0, 0.0)},
            {"name": "Cylinder", "size": (0.1, 0.1, 0.3), "color": (0.0, 0.0, 1.0)},
        ]

        # Place 5-10 random objects in the environment
        num_objects = random.randint(5, 10)
        for i in range(num_objects):
            obj_config = random.choice(object_types)
            position = (
                random.uniform(-4, 4),  # X position
                random.uniform(-3, 3),  # Y position
                obj_config["size"][2] / 2 + 0.01  # Z position (on floor)
            )

            obj_path = Sdf.Path(f"/World/DynamicObject_{i}")

            if obj_config["name"] == "Sphere":
                geom = UsdGeom.Sphere.Define(self.stage, obj_path)
                geom.CreateRadiusAttr(obj_config["size"][0])
            elif obj_config["name"] == "Cylinder":
                geom = UsdGeom.Cylinder.Define(self.stage, obj_path)
                geom.CreateRadiusAttr(obj_config["size"][0])
                geom.CreateHeightAttr(obj_config["size"][2])
            else:  # Box
                geom = UsdGeom.Cube.Define(self.stage, obj_path)
                geom.CreateSizeAttr(1.0)
                xform = UsdGeom.Xformable(geom.GetPrim())
                xform.AddScaleOp().Set(obj_config["size"])

            # Position the object
            xform = UsdGeom.Xformable(geom.GetPrim())
            xform.AddTranslateOp().Set(position)

            self.dynamic_objects.append({
                "path": obj_path,
                "type": obj_config["name"],
                "original_position": position
            })

    def change_lighting_conditions(self):
        """
        Randomly change lighting conditions for domain randomization
        """
        # Get existing lights
        main_light = self.stage.GetPrimAtPath("/World/MainLight")
        ambient_light = self.stage.GetPrimAtPath("/World/AmbientLight")
        task_light = self.stage.GetPrimAtPath("/World/TaskLight")

        if main_light and main_light.IsValid():
            # Randomize main light intensity and color
            intensity = random.uniform(2000, 5000)
            color_var = random.uniform(-0.1, 0.1)
            color = Gf.Vec3f(
                max(0.5, min(1.0, 1.0 + color_var)),
                max(0.5, min(1.0, 0.98 + color_var * 0.5)),
                max(0.5, min(1.0, 0.94 + color_var * 0.5))
            )

            main_light.GetAttribute("inputs:intensity").Set(intensity)
            main_light.GetAttribute("inputs:color").Set(color)

        if ambient_light and ambient_light.IsValid():
            # Randomize ambient light
            intensity = random.uniform(100, 500)
            ambient_light.GetAttribute("inputs:intensity").Set(intensity)

    def randomize_environment(self):
        """
        Randomize the environment for synthetic data generation
        """
        # Move dynamic objects to new positions
        for obj in self.dynamic_objects:
            new_position = (
                random.uniform(-4, 4),
                random.uniform(-3, 3),
                obj["original_position"][2]  # Keep same height
            )

            geom = UsdGeom.Xformable(self.stage.GetPrimAtPath(obj["path"]))
            geom.AddTranslateOp().Set(new_position)

        # Change lighting conditions
        self.change_lighting_conditions()

        print(f"Environment randomized with {len(self.dynamic_objects)} objects moved")

    def reset_environment(self):
        """
        Reset environment to original state
        """
        for obj in self.dynamic_objects:
            geom = UsdGeom.Xformable(self.stage.GetPrimAtPath(obj["path"]))
            geom.AddTranslateOp().Set(obj["original_position"])

        print("Environment reset to original state")
```

## Synthetic Data Generation Pipeline

### 1. Camera Setup for Data Collection

Configuring cameras for synthetic data generation:

```python
# camera_setup.py - Set up cameras for synthetic data collection
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.sensor import Camera
from pxr import Gf, Sdf
import numpy as np

def setup_data_collection_cameras(world: World):
    """
    Set up multiple cameras for comprehensive data collection
    """
    # Head-mounted camera (front-facing)
    head_camera = Camera(
        prim_path="/World/HumanoidRobot/Head/Camera",
        frequency=30,  # 30 Hz
        resolution=(640, 480)
    )

    # Chest-mounted camera (wide-angle)
    chest_camera = Camera(
        prim_path="/World/HumanoidRobot/Chest/Camera",
        frequency=30,
        resolution=(800, 600),
        viewport_position=(0, 0, 0.1),  # Slightly offset
        clipping_range=(0.1, 10.0)
    )

    # Eye-level camera (for human interaction)
    eye_camera = Camera(
        prim_path="/World/HumanoidRobot/Head/EyeCamera",
        frequency=60,  # Higher frequency for interaction
        resolution=(1280, 720)
    )

    # Add cameras to world
    world.scene.add(head_camera)
    world.scene.add(chest_camera)
    world.scene.add(eye_camera)

    return head_camera, chest_camera, eye_camera

def capture_synthetic_data(world: World, cameras, data_dir: str, num_samples: int = 1000):
    """
    Capture synthetic data from multiple cameras
    """
    import os
    import cv2
    from datetime import datetime

    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/rgb", exist_ok=True)
    os.makedirs(f"{data_dir}/depth", exist_ok=True)
    os.makedirs(f"{data_dir}/segmentation", exist_ok=True)

    env_manager = DynamicEnvironmentManager(world)

    for i in range(num_samples):
        # Randomize environment
        env_manager.randomize_environment()

        # Step the simulation
        world.step(render=True)

        # Capture data from each camera
        for j, camera in enumerate(cameras):
            # Get RGB data
            rgb_data = camera.get_rgb()
            if rgb_data is not None:
                rgb_image = cv2.cvtColor(rgb_data, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(f"{data_dir}/rgb/sample_{i:04d}_cam_{j}.png", rgb_image)

            # Get depth data
            depth_data = camera.get_depth()
            if depth_data is not None:
                # Normalize depth for visualization
                depth_normalized = ((depth_data - depth_data.min()) /
                                  (depth_data.max() - depth_data.min()) * 255).astype(np.uint8)
                cv2.imwrite(f"{data_dir}/depth/sample_{i:04d}_cam_{j}.png", depth_normalized)

            # Get segmentation data (if available)
            seg_data = camera.get_semantic_segmentation()
            if seg_data is not None:
                cv2.imwrite(f"{data_dir}/segmentation/sample_{i:04d}_cam_{j}.png", seg_data.astype(np.uint8))

        if i % 100 == 0:
            print(f"Captured {i}/{num_samples} synthetic data samples")

    print(f"Synthetic data collection completed. {num_samples} samples saved to {data_dir}")
```

### 2. Domain Randomization Techniques

Implementing domain randomization for robust perception:

```python
# domain_randomization.py - Implement domain randomization for synthetic data
import random
import numpy as np
from pxr import Gf, Sdf, UsdShade
import omni
from omni.isaac.core.utils.prims import get_prim_at_path

class DomainRandomizer:
    """
    Class to implement domain randomization techniques
    """
    def __init__(self, stage):
        self.stage = stage
        self.materials = []
        self.lights = []

    def randomize_textures(self):
        """
        Randomize textures and materials in the environment
        """
        # Randomize floor texture
        floor_material = self.stage.GetPrimAtPath("/World/Materials/FloorMaterial")
        if floor_material and floor_material.IsValid():
            # Randomize color slightly
            base_color = Gf.Vec3f(
                random.uniform(0.5, 0.9),
                random.uniform(0.5, 0.9),
                random.uniform(0.5, 0.9)
            )
            floor_material.GetAttribute("inputs:diffuse_color").Set(base_color)

            # Randomize roughness
            roughness = random.uniform(0.5, 1.0)
            floor_material.GetAttribute("inputs:roughness").Set(roughness)

        # Randomize wall texture
        wall_material = self.stage.GetPrimAtPath("/World/Materials/WallMaterial")
        if wall_material and wall_material.IsValid():
            wall_color = Gf.Vec3f(
                random.uniform(0.8, 1.0),
                random.uniform(0.8, 1.0),
                random.uniform(0.8, 1.0)
            )
            wall_material.GetAttribute("inputs:diffuse_color").Set(wall_color)

    def randomize_lighting(self):
        """
        Randomize lighting conditions
        """
        # Get all lights in the scene
        lights = ["/World/MainLight", "/World/AmbientLight", "/World/TaskLight", "/World/WindowLight"]

        for light_path in lights:
            light_prim = self.stage.GetPrimAtPath(light_path)
            if light_prim and light_prim.IsValid():
                # Randomize intensity
                base_intensity = light_prim.GetAttribute("inputs:intensity").Get()
                if base_intensity:
                    randomized_intensity = base_intensity * random.uniform(0.5, 2.0)
                    light_prim.GetAttribute("inputs:intensity").Set(randomized_intensity)

                # Randomize color temperature slightly
                base_color = light_prim.GetAttribute("inputs:color").Get()
                if base_color:
                    color_var = random.uniform(-0.1, 0.1)
                    new_color = Gf.Vec3f(
                        max(0.1, min(1.0, base_color[0] + color_var)),
                        max(0.1, min(1.0, base_color[1] + color_var * 0.5)),
                        max(0.1, min(1.0, base_color[2] + color_var * 0.5))
                    )
                    light_prim.GetAttribute("inputs:color").Set(new_color)

    def randomize_object_appearances(self):
        """
        Randomize appearances of objects in the scene
        """
        # Find all dynamic objects and randomize their materials
        for i in range(20):  # Check first 20 dynamic objects
            obj_path = f"/World/DynamicObject_{i}"
            obj_prim = self.stage.GetPrimAtPath(obj_path)
            if obj_prim and obj_prim.IsValid():
                # Create or modify material for this object
                material_path = f"/World/Materials/DynamicObject_{i}_Material"
                material = UsdShade.Material.Define(self.stage, Sdf.Path(material_path))

                shader = UsdShade.Shader.Define(self.stage, Sdf.Path(f"{material_path}/Shader"))
                shader.CreateIdAttr("OmniPBR")

                # Randomize color
                color = Gf.Vec3f(
                    random.uniform(0.2, 1.0),
                    random.uniform(0.2, 1.0),
                    random.uniform(0.2, 1.0)
                )
                shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(color)

                # Randomize material properties
                roughness = random.uniform(0.1, 0.9)
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)

                metallic = random.uniform(0.0, 0.3)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

                # Connect shader to material
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")

    def apply_domain_randomization(self):
        """
        Apply all domain randomization techniques
        """
        self.randomize_textures()
        self.randomize_lighting()
        self.randomize_object_appearances()

        print("Domain randomization applied to environment")

# Integration with environment manager
class EnhancedDynamicEnvironmentManager(DynamicEnvironmentManager):
    """
    Enhanced environment manager with domain randomization
    """
    def __init__(self, world: World):
        super().__init__(world)
        self.domain_randomizer = DomainRandomizer(self.stage)

    def randomize_environment(self):
        """
        Randomize the environment with domain randomization
        """
        # Move dynamic objects
        super().randomize_environment()

        # Apply domain randomization
        self.domain_randomizer.apply_domain_randomization()

        print("Environment randomized with domain randomization")
```

## Performance Optimization for Synthetic Data

### 1. Efficient Rendering Settings

Optimizing Isaac Sim for fast synthetic data generation:

```python
# performance_optimization.py - Optimize rendering for synthetic data generation
import omni
from omni import ui
from omni.isaac.core.utils.settings import set_carb_setting

def optimize_rendering_for_synthetic_data():
    """
    Optimize Isaac Sim settings for efficient synthetic data generation
    """
    # Set rendering quality for performance
    set_carb_setting(omni.appwindow.get_default_app_window().get_settings(),
                     "/app/renderer/enabled", True)

    # Reduce anti-aliasing for faster rendering
    set_carb_setting(omni.appwindow.get_default_app_window().get_settings(),
                     "/rtx/aa/op", 0)  # No anti-aliasing for synthetic data

    # Disable post-processing effects that aren't needed for synthetic data
    set_carb_setting(omni.appwindow.get_default_app_window().get_settings(),
                     "/rtx/post/dlss/enable", False)
    set_carb_setting(omni.appwindow.get_default_app_window().get_settings(),
                     "/rtx/post/fx/enable", False)

    # Optimize for synthetic data (reduce quality settings that don't affect ML)
    set_carb_setting(omni.appwindow.get_default_app_window().get_settings(),
                     "/renderer/quality", 0)  # Lowest quality setting

    print("Rendering optimized for synthetic data generation")

def configure_camera_for_performance(cameras):
    """
    Configure cameras for optimal performance during data collection
    """
    for camera in cameras:
        # Reduce unnecessary processing
        camera.set_fov(60)  # Standard field of view

        # Optimize for the specific task
        if "depth" in camera.name:
            # Ensure depth camera has appropriate settings
            camera.set_resolution((640, 480))  # Lower resolution for performance
        else:
            # RGB cameras can use higher resolution if needed
            camera.set_resolution((640, 480))

def batch_synthetic_data_collection(world: World, cameras, data_dir: str,
                                  num_batches: int = 10, samples_per_batch: int = 100):
    """
    Collect synthetic data in batches for better performance
    """
    import os
    import cv2
    import numpy as np

    os.makedirs(data_dir, exist_ok=True)

    env_manager = EnhancedDynamicEnvironmentManager(world)

    for batch in range(num_batches):
        batch_dir = f"{data_dir}/batch_{batch:03d}"
        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(f"{batch_dir}/rgb", exist_ok=True)
        os.makedirs(f"{batch_dir}/depth", exist_ok=True)

        for i in range(samples_per_batch):
            # Randomize environment with domain randomization
            env_manager.randomize_environment()

            # Step simulation
            world.step(render=True)

            # Capture data from all cameras
            for cam_idx, camera in enumerate(cameras):
                # Get RGB data
                rgb_data = camera.get_rgb()
                if rgb_data is not None:
                    rgb_image = cv2.cvtColor(rgb_data, cv2.COLOR_RGBA2BGR)
                    cv2.imwrite(f"{batch_dir}/rgb/sample_{batch:03d}_{i:04d}_cam_{cam_idx}.png", rgb_image)

                # Get depth data
                depth_data = camera.get_depth()
                if depth_data is not None:
                    depth_normalized = ((depth_data - depth_data.min()) /
                                      (depth_data.max() - depth_data.min()) * 255).astype(np.uint8)
                    cv2.imwrite(f"{batch_dir}/depth/sample_{batch:03d}_{i:04d}_cam_{cam_idx}.png", depth_normalized)

        print(f"Completed batch {batch + 1}/{num_batches}")

    print(f"Synthetic data collection completed: {num_batches * samples_per_batch} samples")
```

## Quality Assurance for Synthetic Data

### 1. Data Validation

Validating the quality of synthetic data:

```python
# data_validation.py - Validate synthetic data quality
import cv2
import numpy as np
import os
from typing import List, Tuple

def validate_synthetic_data(data_dir: str) -> dict:
    """
    Validate the quality and completeness of synthetic data
    """
    validation_results = {
        "total_samples": 0,
        "valid_rgb_samples": 0,
        "valid_depth_samples": 0,
        "rgb_quality_issues": [],
        "depth_quality_issues": [],
        "completeness_score": 0.0
    }

    # Check RGB data
    rgb_dir = f"{data_dir}/rgb"
    if os.path.exists(rgb_dir):
        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        validation_results["total_samples"] = len(rgb_files)

        for file in rgb_files:
            file_path = os.path.join(rgb_dir, file)
            img = cv2.imread(file_path)

            if img is not None:
                validation_results["valid_rgb_samples"] += 1

                # Check for common quality issues
                if img.size == 0:
                    validation_results["rgb_quality_issues"].append(f"{file}: Empty image")
                elif np.mean(img) < 10:  # Too dark
                    validation_results["rgb_quality_issues"].append(f"{file}: Too dark")
                elif np.mean(img) > 245:  # Too bright
                    validation_results["rgb_quality_issues"].append(f"{file}: Too bright")
                elif len(np.unique(img)) < 100:  # Not enough variation
                    validation_results["rgb_quality_issues"].append(f"{file}: Low variation")

    # Check depth data
    depth_dir = f"{data_dir}/depth"
    if os.path.exists(depth_dir):
        depth_files = [f for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for file in depth_files:
            file_path = os.path.join(depth_dir, file)
            depth_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            if depth_img is not None:
                validation_results["valid_depth_samples"] += 1

                # Check depth quality
                if np.all(depth_img == 0) or np.all(depth_img == 255):
                    validation_results["depth_quality_issues"].append(f"{file}: Invalid depth values")
                elif np.max(depth_img) == np.min(depth_img):  # No variation
                    validation_results["depth_quality_issues"].append(f"{file}: No depth variation")

    # Calculate completeness score
    expected_samples = validation_results["total_samples"]
    if expected_samples > 0:
        validation_results["completeness_score"] = (
            (validation_results["valid_rgb_samples"] + validation_results["valid_depth_samples"]) /
            (expected_samples * 2)  # Assuming RGB and depth for each sample
        )

    return validation_results

def print_validation_report(results: dict):
    """
    Print a formatted validation report
    """
    print("=== Synthetic Data Validation Report ===")
    print(f"Total samples: {results['total_samples']}")
    print(f"Valid RGB samples: {results['valid_rgb_samples']}")
    print(f"Valid depth samples: {results['valid_depth_samples']}")
    print(f"Completeness score: {results['completeness_score']:.2%}")

    if results['rgb_quality_issues']:
        print(f"\nRGB Quality Issues ({len(results['rgb_quality_issues'])}):")
        for issue in results['rgb_quality_issues'][:5]:  # Show first 5
            print(f"  - {issue}")
        if len(results['rgb_quality_issues']) > 5:
            print(f"  ... and {len(results['rgb_quality_issues']) - 5} more")

    if results['depth_quality_issues']:
        print(f"\nDepth Quality Issues ({len(results['depth_quality_issues'])}):")
        for issue in results['depth_quality_issues'][:5]:  # Show first 5
            print(f"  - {issue}")
        if len(results['depth_quality_issues']) > 5:
            print(f"  ... and {len(results['depth_quality_issues']) - 5} more")

    print("========================================")
```

## Next Steps

With photorealistic simulation and synthetic data generation properly configured, you're ready to move on to implementing Isaac ROS perception packages. The next section will cover hardware-accelerated perception using Isaac ROS, building on the realistic simulation environment you've created for your humanoid robot applications.