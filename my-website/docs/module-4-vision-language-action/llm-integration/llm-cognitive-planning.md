---
sidebar_position: 3
---

# LLM Integration for Cognitive Planning

## Overview

Large Language Models (LLMs) enable humanoid robots to understand complex natural language commands and convert them into executable action plans. This section covers integrating LLMs with ROS 2 systems for cognitive planning, task decomposition, and safe action sequencing. The integration allows robots to interpret high-level commands and break them down into specific robotic actions.

## LLM Integration Architecture

### 1. ROS 2 LLM Interface

Creating a robust interface between LLMs and ROS 2 systems:

```python
# llm_interface.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
import openai
import json
import asyncio
import threading
from queue import Queue
from datetime import datetime

class LLMInterface(Node):
    """
    Interface between LLMs and ROS 2 systems for cognitive planning
    """
    def __init__(self):
        super().__init__('llm_interface')

        # LLM configuration
        self.llm_model = "gpt-3.5-turbo"  # Can be configured
        self.max_tokens = 500
        self.temperature = 0.3  # Lower for more deterministic responses

        # Context management
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges

        # Create subscribers
        self.command_sub = self.create_subscription(
            String,
            '/validated_command',
            self.command_callback,
            10
        )

        self.system_status_sub = self.create_subscription(
            String,
            '/system_status',
            self.system_status_callback,
            10
        )

        # Create publishers
        self.plan_pub = self.create_publisher(String, '/action_plan', 10)
        self.response_pub = self.create_publisher(String, '/llm_response', 10)
        self.status_pub = self.create_publisher(String, '/llm_status', 10)

        # Request queue for handling multiple requests
        self.request_queue = Queue(maxsize=5)
        self.is_processing = False

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_requests)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info('LLM Interface initialized')

    def command_callback(self, msg):
        """
        Handle incoming validated commands
        """
        command = msg.data.strip()
        self.get_logger().info(f'Received command for LLM processing: "{command}"')

        # Add to processing queue
        try:
            self.request_queue.put_nowait({
                'type': 'command',
                'data': command,
                'timestamp': self.get_clock().now().to_msg()
            })
        except:
            self.get_logger().warn('Request queue full, dropping command')

    def system_status_callback(self, msg):
        """
        Handle system status updates for context
        """
        status = msg.data
        # Add to conversation history for context
        self.conversation_history.append({
            'role': 'system',
            'content': f'System status: {status}',
            'timestamp': self.get_clock().now().to_msg()
        })

    def process_requests(self):
        """
        Process LLM requests in background thread
        """
        while rclpy.ok():
            try:
                request = self.request_queue.get(timeout=1.0)

                if request['type'] == 'command':
                    self.process_command(request['data'])

                self.request_queue.task_done()

            except:
                continue

    def process_command(self, command):
        """
        Process a command through LLM
        """
        if self.is_processing:
            self.get_logger().warn('LLM is busy, skipping command')
            return

        self.is_processing = True
        self.update_status("PROCESSING")

        try:
            # Prepare context for LLM
            context = self.build_context(command)

            # Call LLM
            response = self.call_llm(context)

            # Process and publish response
            if response:
                self.publish_response(response)
                self.process_llm_response(response)

        except Exception as e:
            self.get_logger().error(f'Error processing command with LLM: {e}')
            error_response = {
                'error': str(e),
                'command': command
            }
            error_msg = String()
            error_msg.data = json.dumps(error_response)
            self.response_pub.publish(error_msg)

        finally:
            self.is_processing = False
            self.update_status("IDLE")

    def build_context(self, command):
        """
        Build context for LLM including robot capabilities and environment
        """
        # Get recent conversation history
        recent_history = self.conversation_history[-3:] if len(self.conversation_history) > 3 else self.conversation_history

        # Build context messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cognitive planning assistant for a humanoid robot. "
                    "Your role is to interpret natural language commands and break them down "
                    "into specific, executable actions. The robot has capabilities including "
                    "navigation, object manipulation, and basic interaction. "
                    "Respond in JSON format with a plan that can be executed by the robot. "
                    "Include safety checks and validation for each action."
                )
            }
        ]

        # Add recent history
        for history_item in recent_history:
            messages.append({
                "role": history_item['role'],
                "content": history_item['content']
            })

        # Add current command
        messages.append({
            "role": "user",
            "content": f"Command: {command}"
        })

        return messages

    def call_llm(self, messages):
        """
        Call the LLM with prepared context
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}  # Request JSON response
            )

            content = response.choices[0].message.content
            self.get_logger().info(f'LLM response: {content}')

            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": content,
                "timestamp": self.get_clock().now().to_msg()
            })

            # Keep history within limits
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            return content

        except Exception as e:
            self.get_logger().error(f'Error calling LLM: {e}')
            return None

    def process_llm_response(self, response):
        """
        Process LLM response and convert to action plan
        """
        try:
            # Parse JSON response
            response_data = json.loads(response)

            # Convert to action plan if it's a plan
            if 'actions' in response_data or 'plan' in response_data:
                plan_data = response_data.get('plan', response_data.get('actions', []))
                plan_json = json.dumps(plan_data)

                # Publish action plan
                plan_msg = String()
                plan_msg.data = plan_json
                self.plan_pub.publish(plan_msg)

                self.get_logger().info(f'Published action plan with {len(plan_data)} actions')

        except json.JSONDecodeError:
            self.get_logger().error('LLM response is not valid JSON')
        except Exception as e:
            self.get_logger().error(f'Error processing LLM response: {e}')

    def publish_response(self, response):
        """
        Publish LLM response
        """
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

    def update_status(self, status):
        """
        Update and publish status
        """
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    llm_interface = LLMInterface()

    try:
        rclpy.spin(llm_interface)
    except KeyboardInterrupt:
        llm_interface.get_logger().info('Shutting down LLM interface')
    finally:
        llm_interface.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Task Decomposition System

Implementing a system for decomposing complex tasks into executable actions:

```python
# task_decomposer.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
import json
import re
from enum import Enum
from typing import List, Dict, Any

class TaskType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    SAFETY = "safety"

class TaskDecomposer(Node):
    """
    Decompose complex tasks into executable action sequences
    """
    def __init__(self):
        super().__init__('task_decomposer')

        # Create subscribers
        self.plan_sub = self.create_subscription(
            String,
            '/action_plan',
            self.plan_callback,
            10
        )

        # Create publishers
        self.decomposed_plan_pub = self.create_publisher(String, '/decomposed_plan', 10)
        self.action_sequence_pub = self.create_publisher(String, '/action_sequence', 10)

        # Task decomposition rules
        self.decomposition_rules = {
            'navigation': self.decompose_navigation,
            'manipulation': self.decompose_manipulation,
            'perception': self.decompose_perception,
            'complex': self.decompose_complex_task
        }

        # Known locations and objects
        self.known_locations = {
            'kitchen': {'x': 2.0, 'y': 3.0, 'z': 0.0},
            'living_room': {'x': -1.0, 'y': 1.0, 'z': 0.0},
            'bedroom': {'x': 4.0, 'y': -2.0, 'z': 0.0},
            'office': {'x': -3.0, 'y': -1.0, 'z': 0.0}
        }

        self.known_objects = {
            'water': 'drink',
            'coffee': 'drink',
            'book': 'read',
            'cup': 'container',
            'bottle': 'container',
            'apple': 'food'
        }

        self.get_logger().info('Task Decomposer initialized')

    def plan_callback(self, msg):
        """
        Process incoming action plan and decompose it
        """
        try:
            plan_data = json.loads(msg.data)

            # Decompose the plan
            decomposed_plan = self.decompose_plan(plan_data)

            # Publish decomposed plan
            decomposed_msg = String()
            decomposed_msg.data = json.dumps(decomposed_plan)
            self.decomposed_plan_pub.publish(decomposed_msg)

            # Create and publish action sequence
            action_sequence = self.create_action_sequence(decomposed_plan)
            sequence_msg = String()
            sequence_msg.data = json.dumps(action_sequence)
            self.action_sequence_pub.publish(sequence_msg)

            self.get_logger().info(f'Decomposed plan with {len(action_sequence)} actions')

        except Exception as e:
            self.get_logger().error(f'Error decomposing plan: {e}')

    def decompose_plan(self, plan_data):
        """
        Decompose a plan into primitive actions
        """
        if isinstance(plan_data, list):
            # Handle list of tasks
            decomposed_tasks = []
            for task in plan_data:
                decomposed_tasks.extend(self.decompose_task(task))
            return decomposed_tasks
        else:
            # Handle single task
            return self.decompose_task(plan_data)

    def decompose_task(self, task):
        """
        Decompose a single task into primitive actions
        """
        task_type = self.identify_task_type(task)

        if task_type in self.decomposition_rules:
            return self.decomposition_rules[task_type](task)
        else:
            # Default decomposition for unknown tasks
            return self.decompose_unknown_task(task)

    def identify_task_type(self, task):
        """
        Identify the type of task from the command
        """
        task_str = json.dumps(task).lower() if isinstance(task, dict) else str(task).lower()

        # Navigation tasks
        if any(keyword in task_str for keyword in ['go to', 'move to', 'navigate', 'walk to', 'go', 'move', 'walk']):
            return 'navigation'

        # Manipulation tasks
        if any(keyword in task_str for keyword in ['pick', 'grasp', 'grab', 'take', 'lift', 'place', 'put', 'release', 'drop']):
            return 'manipulation'

        # Perception tasks
        if any(keyword in task_str for keyword in ['find', 'look', 'see', 'locate', 'search', 'identify', 'recognize']):
            return 'perception'

        # Complex multi-step tasks
        if any(keyword in task_str for keyword in ['and', 'then', 'after', 'before', 'while']):
            return 'complex'

        # Default to unknown
        return 'complex'

    def decompose_navigation(self, task):
        """
        Decompose navigation tasks
        """
        if isinstance(task, dict):
            target_location = task.get('location', task.get('target', ''))
        else:
            target_location = str(task)

        # Extract target location
        location = self.extract_location(target_location)

        if location:
            return [
                {
                    'action': 'move_base',
                    'type': 'navigation',
                    'target_location': location,
                    'description': f'Navigate to {location}'
                }
            ]
        else:
            return [
                {
                    'action': 'ask_for_location',
                    'type': 'communication',
                    'description': 'Request clarification of destination'
                }
            ]

    def decompose_manipulation(self, task):
        """
        Decompose manipulation tasks
        """
        if isinstance(task, dict):
            obj = task.get('object', task.get('item', ''))
            action = task.get('action', '')
        else:
            task_str = str(task)
            obj = self.extract_object(task_str)
            action = self.extract_action(task_str)

        actions = []

        if obj:
            # Navigate to object if needed
            actions.append({
                'action': 'find_object',
                'type': 'perception',
                'object': obj,
                'description': f'Locate {obj}'
            })

            # Move to object
            actions.append({
                'action': 'approach_object',
                'type': 'navigation',
                'object': obj,
                'description': f'Move to {obj}'
            })

            # Grasp object
            if 'pick' in action.lower() or 'grasp' in action.lower() or 'take' in action.lower():
                actions.append({
                    'action': 'grasp_object',
                    'type': 'manipulation',
                    'object': obj,
                    'description': f'Grasp {obj}'
                })
            elif 'place' in action.lower() or 'put' in action.lower():
                actions.append({
                    'action': 'release_object',
                    'type': 'manipulation',
                    'object': obj,
                    'description': f'Release {obj}'
                })

        return actions

    def decompose_perception(self, task):
        """
        Decompose perception tasks
        """
        if isinstance(task, dict):
            obj = task.get('object', task.get('item', ''))
        else:
            obj = self.extract_object(str(task))

        return [
            {
                'action': 'search_for_object',
                'type': 'perception',
                'object': obj,
                'description': f'Search for {obj}'
            },
            {
                'action': 'object_recognition',
                'type': 'perception',
                'object': obj,
                'description': f'Recognize {obj}'
            }
        ]

    def decompose_complex_task(self, task):
        """
        Decompose complex multi-step tasks
        """
        if isinstance(task, dict):
            return self.decompose_complex_dict_task(task)
        else:
            return self.decompose_complex_string_task(str(task))

    def decompose_complex_dict_task(self, task):
        """
        Decompose complex task from dictionary
        """
        actions = []

        # Look for subtasks
        if 'subtasks' in task:
            for subtask in task['subtasks']:
                actions.extend(self.decompose_task(subtask))
        elif 'steps' in task:
            for step in task['steps']:
                actions.extend(self.decompose_task(step))

        return actions

    def decompose_complex_string_task(self, task_str):
        """
        Decompose complex task from string
        """
        actions = []

        # Split on common conjunctions
        subtasks = re.split(r'\band\b|\bthen\b|\bafter\b|\bbefore\b', task_str, flags=re.IGNORECASE)

        for subtask in subtasks:
            subtask = subtask.strip()
            if subtask:
                actions.extend(self.decompose_task(subtask))

        return actions

    def decompose_unknown_task(self, task):
        """
        Decompose unknown task type
        """
        return [
            {
                'action': 'request_clarification',
                'type': 'communication',
                'task': str(task),
                'description': f'Cannot understand task: {task}'
            }
        ]

    def extract_location(self, text):
        """
        Extract known location from text
        """
        text_lower = text.lower()
        for location in self.known_locations:
            if location in text_lower:
                return location
        return None

    def extract_object(self, text):
        """
        Extract known object from text
        """
        text_lower = text.lower()
        for obj in self.known_objects:
            if obj in text_lower:
                return obj
        return None

    def extract_action(self, text):
        """
        Extract action verb from text
        """
        # Common action verbs for manipulation
        action_verbs = [
            'pick', 'grasp', 'grab', 'take', 'lift', 'place', 'put', 'release', 'drop',
            'move', 'go', 'walk', 'navigate', 'find', 'look', 'see', 'locate'
        ]

        text_lower = text.lower()
        for verb in action_verbs:
            if verb in text_lower:
                return verb

        return 'unknown'

    def create_action_sequence(self, decomposed_plan):
        """
        Create a sequence of actions from decomposed plan
        """
        sequence = []

        for action in decomposed_plan:
            sequence.append({
                'id': len(sequence) + 1,
                'action': action.get('action', ''),
                'type': action.get('type', ''),
                'parameters': action,
                'description': action.get('description', ''),
                'dependencies': [],  # Add dependencies if needed
                'timeout': 30.0,     # Default timeout
                'retry_count': 3     # Default retry count
            })

        return sequence

def main(args=None):
    rclpy.init(args=args)
    decomposer = TaskDecomposer()

    try:
        rclpy.spin(decomposer)
    except KeyboardInterrupt:
        decomposer.get_logger().info('Shutting down task decomposer')
    finally:
        decomposer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3. Action Planning and Validation

Implementing safe action planning with validation:

```python
# action_planner_validator.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import json
import numpy as np
from typing import Dict, Any

class ActionPlannerValidator(Node):
    """
    Plan actions with safety validation and environment awareness
    """
    def __init__(self):
        super().__init__('action_planner_validator')

        # Create subscribers
        self.action_sequence_sub = self.create_subscription(
            String,
            '/action_sequence',
            self.action_sequence_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Create publishers
        self.validated_plan_pub = self.create_publisher(String, '/validated_action_plan', 10)
        self.safety_check_pub = self.create_publisher(Bool, '/safety_check_result', 10)

        # Environment data
        self.map_data = None
        self.scan_data = None
        self.map_resolution = 0.05
        self.robot_radius = 0.4  # 40cm safety margin

        # Safety parameters
        self.min_distance_to_obstacle = 0.5  # 50cm minimum
        self.max_navigation_distance = 20.0  # 20m maximum

        self.get_logger().info('Action Planner Validator initialized')

    def action_sequence_callback(self, msg):
        """
        Process incoming action sequence and validate it
        """
        try:
            action_sequence = json.loads(msg.data)

            # Validate each action in the sequence
            validated_sequence = []
            all_safe = True

            for i, action in enumerate(action_sequence):
                is_safe, safety_issues = self.validate_action(action)

                if is_safe:
                    validated_sequence.append(action)
                else:
                    self.get_logger().warn(f'Unsafe action at index {i}: {safety_issues}')
                    all_safe = False

                    # Try to find alternative for unsafe action
                    alternative = self.find_alternative_action(action)
                    if alternative:
                        validated_sequence.append(alternative)
                        all_safe = True

            # Publish validated plan
            validated_msg = String()
            validated_msg.data = json.dumps(validated_sequence)
            self.validated_plan_pub.publish(validated_msg)

            # Publish safety status
            safety_msg = Bool()
            safety_msg.data = all_safe
            self.safety_check_pub.publish(safety_msg)

            self.get_logger().info(f'Validated action sequence: {len(validated_sequence)} safe actions')

        except Exception as e:
            self.get_logger().error(f'Error validating action sequence: {e}')

    def validate_action(self, action):
        """
        Validate a single action for safety
        """
        action_type = action.get('type', '')
        safety_issues = []

        if action_type == 'navigation':
            safety_issues.extend(self.validate_navigation_action(action))
        elif action_type == 'manipulation':
            safety_issues.extend(self.validate_manipulation_action(action))
        elif action_type == 'perception':
            safety_issues.extend(self.validate_perception_action(action))

        return len(safety_issues) == 0, safety_issues

    def validate_navigation_action(self, action):
        """
        Validate navigation action for safety
        """
        safety_issues = []

        # Check target location
        target_location = action.get('target_location', {})
        if not target_location:
            safety_issues.append('No target location specified')

        # If we have map data, check path safety
        if self.map_data and isinstance(target_location, dict):
            try:
                # Convert target to map coordinates
                target_x = target_location.get('x', 0)
                target_y = target_location.get('y', 0)

                # Check if target is reachable
                target_map_x = int((target_x - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
                target_map_y = int((target_y - self.map_data.info.origin.position.y) / self.map_data.info.resolution)

                if (0 <= target_map_x < self.map_data.info.width and
                    0 <= target_map_y < self.map_data.info.height):
                    # Check if target cell is free
                    map_index = target_map_y * self.map_data.info.width + target_map_x
                    if self.map_data.data[map_index] > 50:  # Occupancy threshold
                        safety_issues.append(f'Target location is occupied: {target_location}')
                else:
                    safety_issues.append(f'Target location out of map bounds: {target_location}')
            except Exception as e:
                safety_issues.append(f'Error validating navigation target: {e}')

        # Check navigation distance
        if 'distance' in action:
            distance = action['distance']
            if distance > self.max_navigation_distance:
                safety_issues.append(f'Navigation distance too long: {distance}m > {self.max_navigation_distance}m')

        return safety_issues

    def validate_manipulation_action(self, action):
        """
        Validate manipulation action for safety
        """
        safety_issues = []

        # Check object existence and accessibility
        obj = action.get('object', '')
        if not obj:
            safety_issues.append('No object specified for manipulation')

        # Check if object is within robot reach
        # This would require checking current robot state and object position
        # For now, assume validation is done elsewhere

        return safety_issues

    def validate_perception_action(self, action):
        """
        Validate perception action for safety
        """
        safety_issues = []

        # Check if robot can safely observe the area
        # This might involve checking scan data for obstacles in viewing direction

        return safety_issues

    def find_alternative_action(self, unsafe_action):
        """
        Find alternative action for unsafe action
        """
        action_type = unsafe_action.get('type', '')

        if action_type == 'navigation':
            # Try to find a nearby safe location
            return self.find_alternative_navigation(unsafe_action)
        elif action_type == 'manipulation':
            # Suggest alternative manipulation approach
            return self.find_alternative_manipulation(unsafe_action)

        return None

    def find_alternative_navigation(self, action):
        """
        Find alternative navigation target
        """
        # For now, return a safe version of the action
        # In practice, this would search for nearby safe locations
        alternative = action.copy()
        alternative['safety_status'] = 'alternative_found'
        return alternative

    def find_alternative_manipulation(self, action):
        """
        Find alternative manipulation approach
        """
        alternative = action.copy()
        alternative['safety_status'] = 'alternative_approach'
        return alternative

    def map_callback(self, msg):
        """
        Store map data for navigation validation
        """
        self.map_data = msg
        self.map_resolution = msg.info.resolution

    def scan_callback(self, msg):
        """
        Store laser scan data for obstacle detection
        """
        self.scan_data = msg

def main(args=None):
    rclpy.init(args=args)
    validator = ActionPlannerValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Shutting down action planner validator')
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Planning and Reasoning

### 1. Hierarchical Task Planning

Implementing hierarchical task planning for complex scenarios:

```python
# hierarchical_planner.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
import json
import networkx as nx
from typing import Dict, List, Tuple

class HierarchicalPlanner(Node):
    """
    Hierarchical task planning system for complex humanoid tasks
    """
    def __init__(self):
        super().__init__('hierarchical_planner')

        # Create subscribers
        self.high_level_plan_sub = self.create_subscription(
            String,
            '/high_level_plan',
            self.high_level_plan_callback,
            10
        )

        # Create publishers
        self.hierarchical_plan_pub = self.create_publisher(String, '/hierarchical_plan', 10)
        self.task_dependency_pub = self.create_publisher(String, '/task_dependencies', 10)

        # Task hierarchy database
        self.task_database = {
            'assembly': {
                'preconditions': ['parts_available', 'workspace_clear'],
                'subtasks': ['fetch_parts', 'position_parts', 'assemble'],
                'postconditions': ['assembly_complete']
            },
            'navigation': {
                'preconditions': ['path_clear', 'destination_known'],
                'subtasks': ['localize', 'plan_path', 'execute_path'],
                'postconditions': ['at_destination']
            },
            'manipulation': {
                'preconditions': ['object_detected', 'gripper_free'],
                'subtasks': ['approach_object', 'grasp', 'transport', 'place'],
                'postconditions': ['object_placed']
            }
        }

        self.get_logger().info('Hierarchical Planner initialized')

    def high_level_plan_callback(self, msg):
        """
        Process high-level plan and create hierarchical structure
        """
        try:
            high_level_plan = json.loads(msg.data)

            # Create hierarchical plan
            hierarchical_plan = self.create_hierarchical_plan(high_level_plan)

            # Create task dependency graph
            dependency_graph = self.create_dependency_graph(hierarchical_plan)

            # Publish hierarchical plan
            plan_msg = String()
            plan_msg.data = json.dumps(hierarchical_plan)
            self.hierarchical_plan_pub.publish(plan_msg)

            # Publish dependency graph
            dep_msg = String()
            dep_msg.data = json.dumps(dependency_graph)
            self.task_dependency_pub.publish(dep_msg)

            self.get_logger().info(f'Created hierarchical plan with {len(hierarchical_plan)} tasks')

        except Exception as e:
            self.get_logger().error(f'Error creating hierarchical plan: {e}')

    def create_hierarchical_plan(self, high_level_plan):
        """
        Create hierarchical structure from high-level plan
        """
        hierarchical_plan = []

        for task in high_level_plan:
            if isinstance(task, dict):
                task_name = task.get('name', task.get('action', ''))
            else:
                task_name = str(task)

            # Decompose high-level task into subtasks
            subtasks = self.decompose_high_level_task(task_name)

            # Create hierarchical task structure
            hierarchical_task = {
                'id': f"task_{len(hierarchical_plan)}",
                'name': task_name,
                'type': 'high_level',
                'subtasks': subtasks,
                'dependencies': [],
                'priority': 1,
                'timeout': 60.0
            }

            hierarchical_plan.append(hierarchical_task)

        return hierarchical_plan

    def decompose_high_level_task(self, task_name):
        """
        Decompose high-level task into subtasks
        """
        if task_name.lower() in self.task_database:
            return self.create_subtasks_from_database(task_name)
        else:
            return self.create_generic_subtasks(task_name)

    def create_subtasks_from_database(self, task_name):
        """
        Create subtasks based on predefined task database
        """
        task_info = self.task_database[task_name.lower()]
        subtasks = []

        for i, subtask_name in enumerate(task_info['subtasks']):
            subtask = {
                'id': f"{task_name.lower()}_subtask_{i}",
                'name': subtask_name,
                'type': 'primitive',
                'preconditions': task_info['preconditions'],
                'postconditions': [],
                'parameters': {},
                'timeout': 30.0,
                'priority': 2 + i
            }
            subtasks.append(subtask)

        return subtasks

    def create_generic_subtasks(self, task_name):
        """
        Create generic subtasks for unknown task types
        """
        # Generic decomposition based on task patterns
        if 'move' in task_name.lower() or 'go' in task_name.lower():
            return self.create_navigation_subtasks()
        elif 'pick' in task_name.lower() or 'grasp' in task_name.lower():
            return self.create_manipulation_subtasks()
        else:
            return [self.create_generic_subtask(task_name)]

    def create_navigation_subtasks(self):
        """
        Create navigation subtasks
        """
        return [
            {
                'id': 'nav_localize',
                'name': 'localize',
                'type': 'primitive',
                'preconditions': ['amcl_running'],
                'postconditions': ['position_known'],
                'parameters': {},
                'timeout': 10.0,
                'priority': 2
            },
            {
                'id': 'nav_plan_path',
                'name': 'plan_path',
                'type': 'primitive',
                'preconditions': ['map_available', 'goal_set'],
                'postconditions': ['path_planned'],
                'parameters': {},
                'timeout': 10.0,
                'priority': 3
            },
            {
                'id': 'nav_execute_path',
                'name': 'execute_path',
                'type': 'primitive',
                'preconditions': ['path_planned'],
                'postconditions': ['at_goal'],
                'parameters': {},
                'timeout': 60.0,
                'priority': 4
            }
        ]

    def create_manipulation_subtasks(self):
        """
        Create manipulation subtasks
        """
        return [
            {
                'id': 'manip_approach',
                'name': 'approach_object',
                'type': 'primitive',
                'preconditions': ['object_detected'],
                'postconditions': ['at_object'],
                'parameters': {},
                'timeout': 15.0,
                'priority': 2
            },
            {
                'id': 'manip_grasp',
                'name': 'grasp_object',
                'type': 'primitive',
                'preconditions': ['at_object', 'gripper_open'],
                'postconditions': ['object_grasped'],
                'parameters': {},
                'timeout': 10.0,
                'priority': 3
            },
            {
                'id': 'manip_transport',
                'name': 'transport_object',
                'type': 'primitive',
                'preconditions': ['object_grasped'],
                'postconditions': ['at_destination'],
                'parameters': {},
                'timeout': 30.0,
                'priority': 4
            },
            {
                'id': 'manip_place',
                'name': 'place_object',
                'type': 'primitive',
                'preconditions': ['at_destination'],
                'postconditions': ['object_released'],
                'parameters': {},
                'timeout': 10.0,
                'priority': 5
            }
        ]

    def create_generic_subtask(self, task_name):
        """
        Create a generic subtask
        """
        return {
            'id': f'generic_{task_name.lower().replace(" ", "_")}',
            'name': task_name,
            'type': 'primitive',
            'preconditions': [],
            'postconditions': [],
            'parameters': {},
            'timeout': 30.0,
            'priority': 2
        }

    def create_dependency_graph(self, hierarchical_plan):
        """
        Create task dependency graph
        """
        G = nx.DiGraph()

        # Add nodes for all tasks
        for task in hierarchical_plan:
            G.add_node(task['id'], task=task)

            # Add nodes for subtasks
            for subtask in task.get('subtasks', []):
                G.add_node(subtask['id'], task=subtask)
                # Add dependency: subtask depends on parent
                G.add_edge(task['id'], subtask['id'])

        # Create dependency structure
        dependency_graph = {
            'nodes': [],
            'edges': []
        }

        for node_id in G.nodes():
            node_data = G.nodes[node_id]['task']
            dependency_graph['nodes'].append({
                'id': node_id,
                'task': node_data
            })

        for edge in G.edges():
            dependency_graph['edges'].append({
                'source': edge[0],
                'target': edge[1]
            })

        return dependency_graph

def main(args=None):
    rclpy.init(args=args)
    hierarchical_planner = HierarchicalPlanner()

    try:
        rclpy.spin(hierarchical_planner)
    except KeyboardInterrupt:
        hierarchical_planner.get_logger().info('Shutting down hierarchical planner')
    finally:
        hierarchical_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## ROS 2 Action Integration

### 1. ROS 2 Action Server for LLM Plans

Creating an action server to execute LLM-generated plans:

```python
# llm_action_server.py
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import json
from threading import Lock
import time

from humanoid_interfaces.action import ExecutePlan  # Custom action definition

class LLMActionServer(Node):
    """
    ROS 2 Action Server for executing LLM-generated plans
    """
    def __init__(self):
        super().__init__('llm_action_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            ExecutePlan,
            'execute_llm_plan',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self._mutex = Lock()
        self.current_plan = None
        self.is_executing = False

        # Action execution interface
        self.action_interfaces = {
            'move_base': self.execute_navigation,
            'grasp_object': self.execute_manipulation,
            'find_object': self.execute_perception,
            'ask_for_location': self.execute_communication
        }

        self.get_logger().info('LLM Action Server initialized')

    def execute_callback(self, goal_handle):
        """
        Execute the LLM-generated plan
        """
        self.get_logger().info('Received execute plan goal')

        # Parse the plan from goal
        try:
            plan_data = json.loads(goal_handle.request.plan_json)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid plan JSON')
            goal_handle.abort()
            result = ExecutePlan.Result()
            result.success = False
            result.message = 'Invalid plan JSON'
            return result

        # Check if we can execute the plan
        if self.is_executing:
            self.get_logger().warn('Action server already executing, aborting')
            goal_handle.abort()
            result = ExecutePlan.Result()
            result.success = False
            result.message = 'Server already executing'
            return result

        with self._mutex:
            self.is_executing = True
            self.current_plan = plan_data

        # Execute each action in the plan
        feedback = ExecutePlan.Feedback()
        result = ExecutePlan.Result()

        try:
            for i, action in enumerate(plan_data):
                # Check if goal was cancelled
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = 'Goal cancelled'
                    return result

                # Update feedback
                feedback.current_action = f"Executing: {action.get('name', 'unknown')}"
                feedback.progress = float(i + 1) / len(plan_data) * 100.0
                goal_handle.publish_feedback(feedback)

                # Execute the action
                action_name = action.get('action', '')
                if action_name in self.action_interfaces:
                    success, message = self.action_interfaces[action_name](action)
                    if not success:
                        self.get_logger().error(f'Action failed: {message}')
                        goal_handle.abort()
                        result.success = False
                        result.message = f'Action failed: {message}'
                        return result
                else:
                    self.get_logger().error(f'Unknown action: {action_name}')
                    goal_handle.abort()
                    result.success = False
                    result.message = f'Unknown action: {action_name}'
                    return result

        except Exception as e:
            self.get_logger().error(f'Error executing plan: {e}')
            goal_handle.abort()
            result.success = False
            result.message = f'Execution error: {str(e)}'
            return result

        finally:
            with self._mutex:
                self.is_executing = False
                self.current_plan = None

        # Plan completed successfully
        goal_handle.succeed()
        result.success = True
        result.message = 'Plan executed successfully'
        return result

    def execute_navigation(self, action):
        """
        Execute navigation action
        """
        self.get_logger().info(f'Executing navigation: {action}')
        # In real implementation, this would call navigation stack
        time.sleep(2.0)  # Simulate execution time
        return True, "Navigation completed"

    def execute_manipulation(self, action):
        """
        Execute manipulation action
        """
        self.get_logger().info(f'Executing manipulation: {action}')
        # In real implementation, this would call manipulation stack
        time.sleep(3.0)  # Simulate execution time
        return True, "Manipulation completed"

    def execute_perception(self, action):
        """
        Execute perception action
        """
        self.get_logger().info(f'Executing perception: {action}')
        # In real implementation, this would call perception stack
        time.sleep(1.0)  # Simulate execution time
        return True, "Perception completed"

    def execute_communication(self, action):
        """
        Execute communication action
        """
        self.get_logger().info(f'Executing communication: {action}')
        # In real implementation, this would call text-to-speech or similar
        time.sleep(1.0)  # Simulate execution time
        return True, "Communication completed"

def main(args=None):
    rclpy.init(args=args)

    action_server = LLMActionServer()

    # Use MultiThreadedExecutor to handle callbacks in separate threads
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        action_server.get_logger().info('Shutting down LLM action server')
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Next Steps

With the LLM integration and cognitive planning system properly implemented, you're ready to move on to integrating vision systems with action planning. The next section will cover connecting computer vision perception with the cognitive planning system to enable robots to recognize objects, understand scenes, and execute vision-guided actions.