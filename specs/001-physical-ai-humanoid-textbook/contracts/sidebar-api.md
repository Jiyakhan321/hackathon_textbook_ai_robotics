# Sidebar API Contract: Physical AI & Humanoid Robotics Textbook

## Purpose
Define the structure and organization of the textbook navigation sidebar to ensure consistent user experience across all modules.

## Contract Definition

### Sidebar Structure
The sidebar must follow this exact order for optimal learning progression:

1. **Module 1: ROS 2** - Robotic Nervous System
   - Introduction to ROS 2 concepts
   - Nodes, Topics, Services
   - Python integration (rclpy)
   - URDF for humanoids

2. **Module 2: Gazebo & Unity** - Digital Twin
   - Physics simulation
   - Sensors (LiDAR, Depth Camera, IMU)
   - Unity high-fidelity human-robot interaction

3. **Module 3: NVIDIA Isaac (3D)** - AI-Robot Brain
   - 3D humanoid robot rendering (URDF/GLTF)
   - Sensors visualization
   - Path planning & navigation animations

4. **Module 4: Vision-Language-Action (3D + Voice)** - VLA
   - Voice-to-action using Whisper or Web Speech API
   - Cognitive planning (LLM → ROS actions)
   - Robot action animations

5. **Capstone Project (3D + Voice + LLM)**
   - Voice command → LLM planning → robot executes task
   - Visual feedback: path, object detection, manipulation

### Requirements
- Each module must have a clear, descriptive title in the sidebar
- The order must be maintained to support learning progression (basic to advanced)
- Interactive 3D modules (3, 4, and Capstone) must be clearly differentiated
- All entries must be properly nested under their respective module sections
- The sidebar must be responsive and accessible

### Validation Criteria
- Users can navigate to any module using the sidebar
- The order matches the pedagogical sequence: conceptual foundation → simulation → AI integration → advanced interaction → synthesis
- All 3D interactive modules are correctly identified in the navigation
- No broken links or missing entries in the sidebar