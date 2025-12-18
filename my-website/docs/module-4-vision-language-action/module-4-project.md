---
sidebar_position: 5
---

# Module 4 Project: Vision-Language-Action Humanoid Robot

## Project Overview

In this capstone project, we'll build a complete Vision-Language-Action system for a humanoid robot that can understand voice commands, plan actions using LLMs, navigate to targets, and manipulate objects. This integrates all components from Modules 1-4 into a cohesive autonomous system.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HUMANOID ROBOT SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   VOICE INPUT   │  │   VISION        │  │   ACTION        │  │
│  │   PROCESSING    │  │   PROCESSING    │  │   EXECUTION     │  │
│  │                 │  │                 │  │                 │  │
│  │ • Speech-to-    │  │ • RGB-D Camera  │  │ • Navigation    │  │
│  │   Text          │  │ • Object Detec. │  │ • Manipulation  │  │
│  │ • Noise Filter  │  │ • 3D Tracking   │  │ • Grasping      │  │
│  └─────────┬───────┘  └───────┬─────────┘  └─────────┬───────┘  │
│            │                  │                      │          │
│            └──────────────────┼──────────────────────┘          │
│                               │                                  │
│            ┌─────────────────┐ │  ┌──────────────────────────┐  │
│            │   LLM PLANNER   │ ◄──►  COGNITIVE CONTROL      │  │
│            │                 │ │  │   (Behavior Manager)     │  │
│            │ • Task Planning │ │  │ • State Machine          │  │
│            │ • Action Seq.   │ │  │ • Failure Recovery       │  │
│            │ • Context       │ │  │ • Safety Monitoring      │  │
│            │   Reasoning     │ │  └──────────────────────────┘  │
│            └─────────┬───────┘ │                                │
│                      │         │                                │
│                      ▼         ▼                                │
│            ┌─────────────────────────────────────────────────┐  │
│            │              ROS 2 MIDDLEWARE                   │  │
│            │  • Topics, Services, Actions                    │  │
│            │  • Distributed Computing Framework            │  │
│            │  • Hardware Abstraction Layer                 │  │
│            └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
VOICE COMMAND → [STT] → TEXT → [LLM] → ACTION PLAN → [NAVIGATION] → [MANIPULATION]
      ↑                                              ↓              ↓
[ERROR HANDLING] ← [FEEDBACK] ← [PERCEPTION] ← [MONITORING] ← [EXECUTION]
```

## Implementation Steps

### Step 1: Voice Command Processing

First, we'll implement the voice command processing pipeline that converts speech to actionable commands:

```python
import speech_recognition as sr
import asyncio
import json
from typing import Dict, List, Optional

class VoiceCommandProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Configure recognizer
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Command vocabulary
        self.command_patterns = [
            r"pick up the (\w+)",
            r"go to the (\w+)",
            r"bring me the (\w+)",
            r"move the (\w+) to the (\w+)",
            r"find the (\w+)",
        ]

    async def listen_for_command(self) -> Optional[str]:
        """Listen for voice commands and return recognized text."""
        try:
            with self.microphone as source:
                print("Listening for command...")
                audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=5.0)

            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text

        except sr.WaitTimeoutError:
            print("No speech detected within timeout")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return None

    def parse_command(self, text: str) -> Dict:
        """Parse voice command into structured action."""
        import re

        text_lower = text.lower()

        # Handle pickup command
        pickup_match = re.search(r"pick up the (\w+)", text_lower)
        if pickup_match:
            return {
                "action": "pickup",
                "object": pickup_match.group(1),
                "location": None
            }

        # Handle navigation command
        go_to_match = re.search(r"go to the (\w+)", text_lower)
        if go_to_match:
            return {
                "action": "navigate",
                "object": None,
                "location": go_to_match.group(1)
            }

        # Handle bring command
        bring_match = re.search(r"bring me the (\w+)", text_lower)
        if bring_match:
            return {
                "action": "bring",
                "object": bring_match.group(1),
                "location": "user"
            }

        # Handle move command
        move_match = re.search(r"move the (\w+) to the (\w+)", text_lower)
        if move_match:
            return {
                "action": "move",
                "object": move_match.group(1),
                "location": move_match.group(2)
            }

        # Default unknown command
        return {
            "action": "unknown",
            "object": None,
            "location": None,
            "raw_text": text
        }

class VoiceControlNode:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        self.node = Node('voice_control_node')
        self.command_pub = self.node.create_publisher(String, '/voice_commands', 10)

        # Timer for continuous listening
        self.timer = self.node.create_timer(1.0, self.check_voice_input)
        self.processor = VoiceCommandProcessor()

    async def check_voice_input(self):
        """Continuously check for voice input."""
        command_text = await self.processor.listen_for_command()

        if command_text:
            parsed_command = self.processor.parse_command(command_text)

            # Publish command
            cmd_msg = String()
            cmd_msg.data = json.dumps(parsed_command)
            self.command_pub.publish(cmd_msg)
```

### Step 2: LLM Cognitive Planning

Next, we'll implement the LLM-based cognitive planning system that interprets commands and generates action sequences:

```python
import openai
import json
from typing import Dict, List, Any
import asyncio

class LLMParticipant:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.system_prompt = """
        You are an AI planning system for a humanoid robot. Your role is to interpret high-level commands
        and break them down into executable action sequences. Consider:

        1. Object locations and accessibility
        2. Navigation requirements
        3. Manipulation prerequisites
        4. Safety constraints
        5. Environmental context

        Respond with JSON containing the action sequence.
        """

    async def plan_action_sequence(self, command: Dict, environment_state: Dict) -> List[Dict]:
        """Generate action sequence from command and environment state."""

        user_prompt = f"""
        Command: {command}
        Environment: {environment_state}

        Generate a step-by-step action plan as JSON array with elements like:
        {{
            "step": 1,
            "action": "navigation|perception|manipulation",
            "description": "...",
            "parameters": {{}},
            "success_criteria": "..."
        }}
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            # Parse the response
            content = response.choices[0].message.content.strip()

            # Extract JSON from response (handle potential markdown formatting)
            if content.startswith("```json"):
                content = content[7:-3]  # Remove ```json and ```
            elif content.startswith("```"):
                content = content[3:-3]   # Remove ``` and ```

            plan = json.loads(content)
            return plan

        except Exception as e:
            print(f"Error in LLM planning: {e}")
            # Fallback plan
            return self.generate_fallback_plan(command)

    def generate_fallback_plan(self, command: Dict) -> List[Dict]:
        """Generate a basic fallback plan if LLM fails."""
        if command.get('action') == 'pickup':
            return [
                {
                    "step": 1,
                    "action": "navigation",
                    "description": "Navigate to object location",
                    "parameters": {"target_object": command.get('object')},
                    "success_criteria": "Robot at object location"
                },
                {
                    "step": 2,
                    "action": "perception",
                    "description": "Locate specific object",
                    "parameters": {"object_name": command.get('object')},
                    "success_criteria": "Object detected and localized"
                },
                {
                    "step": 3,
                    "action": "manipulation",
                    "description": "Grasp the object",
                    "parameters": {"object_id": command.get('object')},
                    "success_criteria": "Object successfully grasped"
                }
            ]

        return []

class CognitivePlannerNode:
    def __init__(self, llm_api_key: str):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        import threading

        self.node = Node('cognitive_planner_node')

        # Publishers and subscribers
        self.command_sub = self.node.create_subscription(
            String, '/voice_commands', self.command_callback, 10
        )
        self.action_plan_pub = self.node.create_publisher(
            String, '/action_plan', 10
        )
        self.environment_sub = self.node.create_subscription(
            String, '/environment_state', self.environment_callback, 10
        )

        # Initialize LLM planner
        self.llm_planner = LLMParticipant(llm_api_key)
        self.current_environment = {}

        # Thread for async LLM calls
        self.planning_thread = None

    def command_callback(self, msg):
        """Handle incoming voice commands."""
        try:
            command = json.loads(msg.data)

            # Start planning in background thread
            self.planning_thread = threading.Thread(
                target=self.async_plan_wrapper,
                args=(command,)
            )
            self.planning_thread.start()

        except json.JSONDecodeError:
            print(f"Invalid JSON in command: {msg.data}")

    def environment_callback(self, msg):
        """Update environment state."""
        try:
            self.current_environment = json.loads(msg.data)
        except json.JSONDecodeError:
            print(f"Invalid JSON in environment: {msg.data}")

    def async_plan_wrapper(self, command):
        """Wrapper for async planning."""
        import asyncio

        async def plan():
            plan = await self.llm_planner.plan_action_sequence(
                command, self.current_environment
            )

            # Publish action plan
            plan_msg = String()
            plan_msg.data = json.dumps(plan)
            self.action_plan_pub.publish(plan_msg)

        # Run async planning
        asyncio.run(plan())
```

### Step 3: Navigation System

Now we'll implement the navigation system that moves the robot to target locations:

```python
import numpy as np
from typing import Tuple, List, Dict
import math

class NavigationSystem:
    def __init__(self):
        # Robot properties
        self.robot_radius = 0.3  # meters
        self.max_speed = 0.5     # m/s
        self.rotation_speed = 0.5  # rad/s

        # Map and localization
        self.map_resolution = 0.05  # meters per pixel
        self.local_map = None
        self.global_map = None
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta

    def update_robot_pose(self, pose: np.ndarray):
        """Update current robot pose."""
        self.robot_pose = pose

    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find path from start to goal using A* algorithm."""
        import heapq

        def heuristic(pos1, pos2):
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        def get_neighbors(pos):
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    if self.is_valid_position(new_pos):
                        neighbors.append(new_pos)
            return neighbors

        def is_valid_position(pos):
            # Check map bounds and obstacles
            if (pos[0] < 0 or pos[1] < 0 or
                pos[0] >= self.local_map.shape[0] or pos[1] >= self.local_map.shape[1]):
                return False

            # Check if cell is free (assuming 0 = free, 100 = obstacle)
            return self.local_map[pos[0], pos[1]] < 50

        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(current))
                    current = came_from[current]
                path.append(self.grid_to_world(start_grid))
                return path[::-1]

            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def world_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int((pos[0] - self.robot_pose[0]) / self.map_resolution)
        grid_y = int((pos[1] - self.robot_pose[1]) / self.map_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        world_x = pos[0] * self.map_resolution + self.robot_pose[0]
        world_y = pos[1] * self.map_resolution + self.robot_pose[1]
        return (world_x, world_y)

    def follow_path(self, path: List[Tuple[float, float]]) -> bool:
        """Follow the planned path."""
        for waypoint in path:
            if not self.navigate_to_waypoint(waypoint):
                return False  # Navigation failed
        return True

    def navigate_to_waypoint(self, target: Tuple[float, float]) -> bool:
        """Navigate to a single waypoint."""
        target_x, target_y = target
        current_x, current_y = self.robot_pose[0], self.robot_pose[1]

        # Calculate distance and angle to target
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)

        if distance < 0.1:  # Close enough
            return True

        target_angle = math.atan2(target_y - current_y, target_x - current_x)
        angle_diff = target_angle - self.robot_pose[2]

        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Rotate towards target
        if abs(angle_diff) > 0.1:
            self.rotate_robot(angle_diff)

        # Move forward
        self.move_forward(min(distance, 0.1))  # Move in small increments

        return True

    def rotate_robot(self, angle: float):
        """Rotate robot by specified angle."""
        # Simulate rotation
        print(f"Rotating robot by {angle} radians")
        self.robot_pose[2] += angle

    def move_forward(self, distance: float):
        """Move robot forward by specified distance."""
        # Simulate forward movement
        print(f"Moving forward {distance} meters")
        self.robot_pose[0] += distance * math.cos(self.robot_pose[2])
        self.robot_pose[1] += distance * math.sin(self.robot_pose[2])

class NavigationNode:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import OccupancyGrid

        self.node = Node('navigation_node')

        # Publishers and subscribers
        self.goal_sub = self.node.create_subscription(
            PoseStamped, '/navigation_goal', self.goal_callback, 10
        )
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.map_sub = self.node.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        # Navigation system
        self.nav_system = NavigationSystem()
        self.current_goal = None

    def map_callback(self, msg):
        """Update map data."""
        import numpy as np

        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape(height, width)
        self.nav_system.local_map = data

    def goal_callback(self, msg):
        """Handle navigation goal."""
        target_x = msg.pose.position.x
        target_y = msg.pose.position.y

        # Update robot pose (would come from localization)
        current_pose = np.array([0.0, 0.0, 0.0])  # Placeholder
        self.nav_system.update_robot_pose(current_pose)

        # Plan and execute navigation
        path = self.nav_system.find_path(
            (current_pose[0], current_pose[1]),
            (target_x, target_y)
        )

        if path:
            success = self.nav_system.follow_path(path)
            print(f"Navigation {'successful' if success else 'failed'}")
        else:
            print("No path found to goal")
```

### Step 4: Object Manipulation System

Now we'll implement the object manipulation system for grasping and manipulating objects:

```python
import numpy as np
from typing import Dict, List, Tuple
import math

class ManipulationSystem:
    def __init__(self):
        # Robot arm properties
        self.arm_joints = 6  # Number of joints
        self.joint_limits = [(-2.96, 2.96)] * 6  # Joint limits in radians
        self.reach = 1.2  # Maximum reach in meters
        self.gripper_max_aperture = 0.1  # Maximum gripper opening
        self.gripper_min_aperture = 0.01  # Minimum gripper opening

        # Current state
        self.current_joints = np.zeros(self.arm_joints)
        self.end_effector_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x, y, z, roll, pitch, yaw
        self.gripper_opening = self.gripper_max_aperture

    def calculate_inverse_kinematics(self, target_pose: np.ndarray) -> np.ndarray:
        """Calculate joint angles for target end-effector pose."""
        # Simplified analytical IK for demonstration
        # In practice, use numerical methods or robot-specific solvers

        target_x, target_y, target_z = target_pose[:3]

        # Calculate distance from base to target
        dist_xy = math.sqrt(target_x**2 + target_y**2)
        dist_xyz = math.sqrt(dist_xy**2 + target_z**2)

        if dist_xyz > self.reach:
            # Target out of reach, return current configuration
            return self.current_joints.copy()

        # Simplified calculation (real implementation would be more complex)
        joint_angles = np.zeros(self.arm_joints)

        # Base joint (shoulder pan)
        joint_angles[0] = math.atan2(target_y, target_x)

        # Shoulder lift
        shoulder_height = 0.2  # Fixed shoulder height
        arm_length = self.reach / 2  # Simplified arm model

        joint_angles[1] = math.atan2(target_z - shoulder_height, dist_xy)

        # Elbow joint
        joint_angles[2] = 0.0  # Simplified

        # Wrist joints
        joint_angles[3] = target_pose[3]  # Roll
        joint_angles[4] = target_pose[4]  # Pitch
        joint_angles[5] = target_pose[5]  # Yaw

        # Validate joint limits
        for i, angle in enumerate(joint_angles):
            min_lim, max_lim = self.joint_limits[i]
            joint_angles[i] = max(min_lim, min(max_lim, angle))

        return joint_angles

    def plan_grasp_pose(self, object_info: Dict) -> np.ndarray:
        """Plan optimal grasp pose for object."""
        obj_position = np.array(object_info['position'])
        obj_dimensions = np.array(object_info['dimensions'])

        # Calculate grasp position (slightly above object center)
        grasp_x = obj_position[0]
        grasp_y = obj_position[1]
        grasp_z = obj_position[2] + obj_dimensions[2] / 2 + 0.05  # 5cm above object

        # Determine grasp orientation based on object shape
        if obj_dimensions[0] > obj_dimensions[1] and obj_dimensions[0] > obj_dimensions[2]:
            # Longest dimension is X, grasp along Y axis
            roll = 0.0
            pitch = 0.0
            yaw = math.pi / 2  # 90 degrees
        elif obj_dimensions[1] > obj_dimensions[2]:
            # Longest dimension is Y, grasp along X axis
            roll = 0.0
            pitch = 0.0
            yaw = 0.0
        else:
            # Longest dimension is Z, grasp from top
            roll = 0.0
            pitch = math.pi / 2  # 90 degrees pitch
            yaw = 0.0

        return np.array([grasp_x, grasp_y, grasp_z, roll, pitch, yaw])

    def execute_grasp(self, object_info: Dict) -> bool:
        """Execute grasp operation on object."""
        try:
            # Plan grasp pose
            grasp_pose = self.plan_grasp_pose(object_info)

            # Calculate approach pose (above object)
            approach_pose = grasp_pose.copy()
            approach_pose[2] += 0.1  # 10cm above object

            # Move to approach position
            if not self.move_to_pose(approach_pose):
                return False

            # Move to grasp position
            if not self.move_to_pose(grasp_pose):
                return False

            # Close gripper
            object_width = min(object_info['dimensions'])
            grip_aperture = max(self.gripper_min_aperture, object_width - 0.01)
            self.close_gripper(grip_aperture)

            # Verify grasp success (simulated)
            grasp_successful = self.verify_grasp(object_info)

            if grasp_successful:
                # Lift object slightly
                lift_pose = grasp_pose.copy()
                lift_pose[2] += 0.05  # Lift 5cm
                self.move_to_pose(lift_pose)

                print(f"Successfully grasped {object_info.get('name', 'object')}")
                return True
            else:
                print("Grasp failed")
                return False

        except Exception as e:
            print(f"Error during grasp execution: {e}")
            return False

    def move_to_pose(self, target_pose: np.ndarray) -> bool:
        """Move end effector to target pose."""
        try:
            # Calculate required joint angles
            target_joints = self.calculate_inverse_kinematics(target_pose)

            if not self.validate_joints(target_joints):
                print("Target pose unreachable due to joint limits")
                return False

            # Simulate movement (in real system, send commands to controllers)
            self.current_joints = target_joints
            self.end_effector_pose = target_pose

            print(f"Moved to pose: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
            return True

        except Exception as e:
            print(f"Error moving to pose: {e}")
            return False

    def validate_joints(self, joints: np.ndarray) -> bool:
        """Validate joint angles are within limits."""
        for i, angle in enumerate(joints):
            min_lim, max_lim = self.joint_limits[i]
            if angle < min_lim or angle > max_lim:
                return False
        return True

    def close_gripper(self, aperture: float):
        """Close gripper to specified aperture."""
        self.gripper_opening = max(self.gripper_min_aperture, min(aperture, self.gripper_max_aperture))
        print(f"Gripper closed to {self.gripper_opening:.3f}m")

    def verify_grasp(self, object_info: Dict) -> bool:
        """Verify that grasp was successful."""
        # Simulated grasp verification
        # In real system, use force/torque sensors or visual feedback
        import random
        return random.random() > 0.2  # 80% success rate for simulation

class ManipulationNode:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String

        self.node = Node('manipulation_node')

        # Publishers and subscribers
        self.grasp_sub = self.node.create_subscription(
            String, '/manipulation_command', self.grasp_callback, 10
        )
        self.object_sub = self.node.create_subscription(
            String, '/object_detections', self.object_callback, 10
        )

        # Manipulation system
        self.manipulator = ManipulationSystem()
        self.known_objects = {}

    def object_callback(self, msg):
        """Update known objects."""
        try:
            objects = json.loads(msg.data)
            for obj in objects:
                self.known_objects[obj.get('class')] = obj
        except json.JSONDecodeError:
            print(f"Invalid JSON in object detections: {msg.data}")

    def grasp_callback(self, msg):
        """Handle grasp commands."""
        try:
            command = json.loads(msg.data)
            object_name = command.get('object_name')

            if object_name in self.known_objects:
                object_info = self.known_objects[object_name]
                success = self.manipulator.execute_grasp(object_info)

                # Publish result
                result_msg = String()
                result_msg.data = json.dumps({
                    'success': success,
                    'object': object_name,
                    'action': 'grasp'
                })

                # Would publish to result topic
                # self.result_pub.publish(result_msg)

            else:
                print(f"Unknown object: {object_name}")

        except json.JSONDecodeError:
            print(f"Invalid JSON in grasp command: {msg.data}")
```

### Step 5: System Integration and Behavior Manager

Finally, we'll create the behavior manager that orchestrates all components:

```python
import asyncio
import threading
from typing import Dict, List
import time

class BehaviorManager:
    def __init__(self):
        # System state
        self.current_behavior = "idle"
        self.action_queue = []
        self.environment_state = {}
        self.robot_state = {
            'position': [0, 0, 0],
            'orientation': [0, 0, 0, 1],
            'gripper': 'open',
            'holding_object': None
        }

        # Component references (would be connected to actual nodes)
        self.voice_processor = VoiceCommandProcessor()
        self.llm_planner = LLMParticipant(api_key="your-api-key")
        self.navigation_system = NavigationSystem()
        self.manipulation_system = ManipulationSystem()

        # Safety monitors
        self.emergency_stop = False
        self.collision_detected = False

    async def start_system(self):
        """Start the integrated system."""
        print("Starting Vision-Language-Action System...")

        # Start main control loop
        while not self.emergency_stop:
            await self.control_cycle()
            await asyncio.sleep(0.1)  # 10Hz control loop

    async def control_cycle(self):
        """Main control cycle."""
        try:
            # Process any pending actions
            if self.action_queue:
                current_action = self.action_queue[0]

                # Execute action based on type
                if current_action['action'] == 'navigation':
                    await self.execute_navigation(current_action)
                elif current_action['action'] == 'manipulation':
                    await self.execute_manipulation(current_action)
                elif current_action['action'] == 'perception':
                    await self.execute_perception(current_action)

                # Remove completed action
                if self.is_action_complete(current_action):
                    self.action_queue.pop(0)

            # Monitor system health
            self.monitor_safety()

        except Exception as e:
            print(f"Error in control cycle: {e}")

    async def execute_navigation(self, action: Dict):
        """Execute navigation action."""
        target_location = action['parameters'].get('target_object')

        if target_location:
            # Find object location from environment state
            target_pose = self.find_object_location(target_location)
            if target_pose:
                # Navigate to object
                path = self.navigation_system.find_path(
                    self.robot_state['position'][:2],
                    target_pose[:2]
                )

                if path:
                    success = self.navigation_system.follow_path(path)
                    if success:
                        # Update robot position
                        self.robot_state['position'] = target_pose
                        print(f"Navigation to {target_location} completed")
                else:
                    print(f"Could not navigate to {target_location}: no path found")

    async def execute_manipulation(self, action: Dict):
        """Execute manipulation action."""
        object_name = action['parameters'].get('object_id')

        if object_name:
            # Get object information
            object_info = self.get_object_info(object_name)
            if object_info:
                success = self.manipulation_system.execute_grasp(object_info)

                if success:
                    self.robot_state['holding_object'] = object_name
                    self.robot_state['gripper'] = 'closed'
                    print(f"Manipulation of {object_name} completed")
            else:
                print(f"Object {object_name} not found")

    async def execute_perception(self, action: Dict):
        """Execute perception action."""
        object_name = action['parameters'].get('object_name')

        # Update environment state with current perceptions
        self.update_environment_state()

        if object_name:
            # Verify object exists and is accessible
            if object_name in self.environment_state:
                print(f"Object {object_name} confirmed in environment")
            else:
                print(f"Object {object_name} not found in environment")

    def is_action_complete(self, action: Dict) -> bool:
        """Check if action is complete."""
        # Simple completion check - in real system, use feedback
        if action['action'] == 'navigation':
            target_pos = action['parameters'].get('target_object')
            if target_pos:
                target_pose = self.find_object_location(target_pos)
                if target_pose:
                    current_pos = self.robot_state['position']
                    distance = math.sqrt(sum((a-b)**2 for a, b in
                                           zip(current_pos[:2], target_pose[:2])))
                    return distance < 0.3  # Within 30cm

        return False

    def find_object_location(self, object_name: str) -> List[float]:
        """Find location of object in environment."""
        # In real system, query perception system
        if object_name in self.environment_state:
            return self.environment_state[object_name]['position']
        return None

    def get_object_info(self, object_name: str) -> Dict:
        """Get detailed information about an object."""
        if object_name in self.environment_state:
            return self.environment_state[object_name]
        return None

    def update_environment_state(self):
        """Update environment state from perception system."""
        # In real system, subscribe to perception topics
        pass

    def monitor_safety(self):
        """Monitor safety conditions."""
        # Check for collisions, emergency stops, etc.
        if self.collision_detected:
            print("Safety violation detected! Stopping all motion.")
            self.emergency_stop = True

    async def process_voice_command(self, command_text: str):
        """Process voice command through full pipeline."""
        # Parse command
        parsed_command = self.voice_processor.parse_command(command_text)

        # Update environment state
        self.update_environment_state()

        # Plan action sequence
        action_plan = await self.llm_planner.plan_action_sequence(
            parsed_command, self.environment_state
        )

        # Queue actions for execution
        for action in action_plan:
            self.action_queue.append(action)

        print(f"Queued {len(action_plan)} actions for execution")

class IntegratedSystemNode:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        self.node = Node('integrated_system_node')

        # Publishers and subscribers
        self.voice_cmd_sub = self.node.create_subscription(
            String, '/voice_commands', self.voice_command_callback, 10
        )
        self.action_plan_sub = self.node.create_subscription(
            String, '/action_plan', self.action_plan_callback, 10
        )
        self.environment_sub = self.node.create_subscription(
            String, '/environment_state', self.environment_callback, 10
        )

        # Behavior manager
        self.behavior_manager = BehaviorManager()

        # Start system in background
        self.system_thread = threading.Thread(
            target=self.run_system
        )
        self.system_thread.daemon = True
        self.system_thread.start()

    def voice_command_callback(self, msg):
        """Handle voice commands."""
        try:
            command_data = json.loads(msg.data)
            command_text = command_data.get('raw_text', '')

            if command_text:
                # Process command in async context
                asyncio.run(
                    self.behavior_manager.process_voice_command(command_text)
                )

        except json.JSONDecodeError:
            print(f"Invalid JSON in voice command: {msg.data}")

    def action_plan_callback(self, msg):
        """Handle received action plans."""
        try:
            action_plan = json.loads(msg.data)

            # Add to behavior manager queue
            for action in action_plan:
                self.behavior_manager.action_queue.append(action)

        except json.JSONDecodeError:
            print(f"Invalid JSON in action plan: {msg.data}")

    def environment_callback(self, msg):
        """Update environment state."""
        try:
            env_state = json.loads(msg.data)
            self.behavior_manager.environment_state = env_state
        except json.JSONDecodeError:
            print(f"Invalid JSON in environment state: {msg.data}")

    def run_system(self):
        """Run the integrated system."""
        import asyncio
        asyncio.run(self.behavior_manager.start_system())

def main():
    """Main entry point for the integrated system."""
    import rclpy

    rclpy.init()

    # Initialize all nodes
    voice_node = VoiceControlNode()
    planner_node = CognitivePlannerNode(llm_api_key="your-openai-api-key")
    nav_node = NavigationNode()
    manip_node = ManipulationNode()
    system_node = IntegratedSystemNode()

    # Spin all nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(voice_node.node)
    executor.add_node(planner_node.node)
    executor.add_node(nav_node.node)
    executor.add_node(manip_node.node)
    executor.add_node(system_node.node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 6: Configuration Files

Create the necessary configuration files for the system:

**config/vision_action_system.yaml:**
```yaml
# Vision-Action System Configuration

voice_processing:
  sample_rate: 16000
  chunk_size: 1024
  threshold: 300
  timeout: 5.0
  phrase_time_limit: 5.0

llm_planning:
  api_endpoint: "https://api.openai.com/v1/chat/completions"
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 1000
  timeout: 30

navigation:
  resolution: 0.05
  robot_radius: 0.3
  max_speed: 0.5
  rotation_speed: 0.5
  obstacle_threshold: 50

manipulation:
  joint_limits:
    - [-2.96, 2.96]
    - [-2.96, 2.96]
    - [-2.96, 2.96]
    - [-2.96, 2.96]
    - [-2.96, 2.96]
    - [-2.96, 2.96]
  reach: 1.2
  gripper_max_aperture: 0.1
  gripper_min_aperture: 0.01

behavior_manager:
  control_frequency: 10.0
  safety_monitoring: true
  emergency_stop_timeout: 5.0
```

**launch/vision_action_system.launch.py:**
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('vision_action_system'), 'config')

    return LaunchDescription([
        Node(
            package='vision_action_system',
            executable='voice_control_node',
            name='voice_control_node',
            parameters=[os.path.join(config_dir, 'vision_action_system.yaml')],
            output='screen'
        ),

        Node(
            package='vision_action_system',
            executable='cognitive_planner_node',
            name='cognitive_planner_node',
            parameters=[os.path.join(config_dir, 'vision_action_system.yaml')],
            output='screen'
        ),

        Node(
            package='vision_action_system',
            executable='navigation_node',
            name='navigation_node',
            parameters=[os.path.join(config_dir, 'vision_action_system.yaml')],
            output='screen'
        ),

        Node(
            package='vision_action_system',
            executable='manipulation_node',
            name='manipulation_node',
            parameters=[os.path.join(config_dir, 'vision_action_system.yaml')],
            output='screen'
        ),

        Node(
            package='vision_action_system',
            executable='integrated_system_node',
            name='integrated_system_node',
            parameters=[os.path.join(config_dir, 'vision_action_system.yaml')],
            output='screen'
        )
    ])
```

## Testing and Validation

### Unit Tests

Create comprehensive tests for each component:

```python
import unittest
import numpy as np
from unittest.mock import Mock, patch

class TestVisionActionSystem(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.vision_processor = MultiModalFusion()
        self.llm_planner = LLMParticipant(api_key="test-key")
        self.nav_system = NavigationSystem()
        self.manip_system = ManipulationSystem()
        self.behavior_manager = BehaviorManager()

    def test_object_detection(self):
        """Test object detection functionality."""
        # Mock RGB and depth images
        mock_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_depth = np.random.rand(480, 640).astype(np.float32)

        # Test detection
        detections = self.vision_processor.fused_detection(mock_rgb, mock_depth)

        # Assertions
        self.assertIsInstance(detections, list)
        # Additional assertions based on expected behavior

    def test_navigation_pathfinding(self):
        """Test navigation pathfinding."""
        start = (0.0, 0.0)
        goal = (5.0, 5.0)

        path = self.nav_system.find_path(start, goal)

        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)

    def test_grasp_planning(self):
        """Test grasp planning."""
        object_info = {
            'position': [1.0, 1.0, 0.5],
            'dimensions': [0.1, 0.1, 0.1],
            'name': 'cube'
        }

        grasp_pose = self.manip_system.plan_grasp_pose(object_info)

        self.assertEqual(len(grasp_pose), 6)  # x, y, z, roll, pitch, yaw

    @patch('openai.ChatCompletion.acreate')
    def test_llm_planning(self, mock_create):
        """Test LLM planning."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '[{"step": 1, "action": "navigation", "description": "Go to object", "parameters": {}, "success_criteria": "At location"}]'

        mock_create.return_value = mock_response

        command = {"action": "pickup", "object": "red_cube", "location": None}
        env_state = {"objects": [{"name": "red_cube", "position": [1, 1, 0]}]}

        plan = self.llm_planner.plan_action_sequence(command, env_state)

        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the full system."""

    def test_voice_to_action_pipeline(self):
        """Test complete voice-to-action pipeline."""
        # This would test the full flow from voice command to action execution
        pass

if __name__ == '__main__':
    unittest.main()
```

### Performance Benchmarks

Monitor system performance with benchmarks:

```python
import time
import psutil
import threading
from collections import deque

class PerformanceMonitor:
    def __init__(self):
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)

        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory_percent)

    def measure_latency(self, func, *args, **kwargs):
        """Measure function execution latency."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.latency_history.append(latency)

        return result, latency

    def get_statistics(self):
        """Get current performance statistics."""
        stats = {
            'cpu_avg': sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
            'memory_avg': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            'latency_avg': sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0,
            'latency_min': min(self.latency_history) if self.latency_history else 0,
            'latency_max': max(self.latency_history) if self.latency_history else 0
        }
        return stats

# Usage example
perf_monitor = PerformanceMonitor()
perf_monitor.start_monitoring()

# Measure specific operations
result, latency = perf_monitor.measure_latency(
    lambda: self.vision_processor.fused_detection(rgb_img, depth_img)
)
```

## Deployment and Operation

### System Setup

1. **Hardware Requirements:**
   - RGB-D camera (Intel RealSense, Kinect, etc.)
   - Microphone array for voice input
   - Humanoid robot platform with manipulator arms
   - Computer with sufficient processing power (GPU recommended)

2. **Software Dependencies:**
   - ROS 2 (Humble Hawksbill or later)
   - Python 3.8+
   - OpenCV
   - PyTorch/TensorFlow
   - OpenAI API client
   - Speech recognition libraries

3. **Installation:**
```bash
# Install ROS 2 dependencies
sudo apt update
sudo apt install ros-humble-desktop-full
source /opt/ros/humble/setup.bash

# Install Python packages
pip install openai opencv-python speechrecognition torch torchvision numpy scipy

# Build the workspace
colcon build
source install/setup.bash
```

### Launching the System

```bash
# Launch the complete system
ros2 launch vision_action_system vision_action_system.launch.py

# Or launch individual components
ros2 run vision_action_system voice_control_node
ros2 run vision_action_system cognitive_planner_node
```

## Troubleshooting

### Common Issues

1. **Voice Recognition Problems:**
   - Check microphone permissions and configuration
   - Adjust ambient noise threshold
   - Verify internet connectivity for cloud-based STT

2. **Navigation Failures:**
   - Ensure map is properly loaded
   - Check localization accuracy
   - Verify obstacle detection parameters

3. **Manipulation Failures:**
   - Calibrate camera-robot coordinate transformation
   - Verify object detection accuracy
   - Check gripper calibration

### Debugging Tips

- Enable detailed logging for each component
- Monitor ROS 2 topics and services
- Use visualization tools (RViz) to inspect perception results
- Implement graceful error handling and recovery mechanisms

## Summary

This Vision-Language-Action system demonstrates the integration of multiple complex technologies to create an intelligent humanoid robot capable of understanding natural language commands, perceiving its environment, and executing meaningful actions. The system follows a modular architecture with clear interfaces between components, making it extensible and maintainable.

Key achievements include:
- Voice command processing with natural language understanding
- Multi-modal perception combining RGB and depth information
- LLM-powered cognitive planning for complex task decomposition
- Safe navigation and manipulation capabilities
- Real-time system integration using ROS 2

The project serves as a foundation for advanced robotics applications and can be extended with additional capabilities such as multi-robot coordination, learning from demonstration, or enhanced safety features.