---
sidebar_position: 3
---

# LLM Integration for Action Planning

## Overview

Large Language Model (LLM) integration is the cognitive core of Vision-Language-Action (VLA) systems for humanoid robots. This component bridges natural language understanding with robotic action planning, enabling humanoid robots to interpret complex voice commands and decompose them into executable robotic tasks.

LLM integration transforms high-level human instructions into structured action plans that consider environmental constraints, robot capabilities, and safety requirements. This module covers the implementation of prompt engineering, context management, and cognitive planning systems that connect voice recognition with physical execution.

## Learning Objectives

By the end of this section, you will be able to:
- Design effective prompts for robotics task decomposition
- Implement context management for ongoing conversations
- Create cognitive planning pipelines that convert natural language to robot actions
- Integrate safety validation layers with LLM outputs
- Connect LLM outputs to ROS 2 action execution systems

## Prerequisites

Before implementing LLM integration for action planning, ensure you have:
- Completed Module 3 (AI-Robot Brain) focusing on navigation and perception
- Voice recognition system from Section 2 of this module
- Basic understanding of LLM APIs (OpenAI GPT, Anthropic Claude, or open-source alternatives)
- ROS 2 action client/server implementation knowledge
- Familiarity with task planning and execution frameworks

## LLM Selection and Setup

### Choosing the Right LLM for Robotics

Different LLM architectures offer varying capabilities for robotics applications:

```python
import openai
import anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMProviderManager:
    """Manage different LLM providers for robotics applications"""

    def __init__(self):
        self.providers = {
            'openai': self._setup_openai,
            'anthropic': self._setup_anthropic,
            'open_source': self._setup_open_source
        }
        self.current_provider = None

    def _setup_openai(self, api_key: str):
        """Setup OpenAI GPT for robotics tasks"""
        openai.api_key = api_key
        return {
            'client': openai.OpenAI(api_key=api_key),
            'model': 'gpt-4-turbo',
            'max_tokens': 2048,
            'temperature': 0.3
        }

    def _setup_anthropic(self, api_key: str):
        """Setup Anthropic Claude for robotics tasks"""
        return {
            'client': anthropic.Anthropic(api_key=api_key),
            'model': 'claude-3-5-sonnet-20241022',
            'max_tokens': 2048,
            'temperature': 0.3
        }

    def _setup_open_source(self, model_name: str):
        """Setup open-source LLM for robotics tasks"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return {
            'tokenizer': tokenizer,
            'model': model,
            'device': model.device
        }
```

### Environment Configuration

Configure your environment for LLM integration:

```bash
# Create environment variables file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LLM_PROVIDER=openai  # or anthropic, open_source
LLM_MODEL=gpt-4-turbo
EOF

# Install required dependencies
pip install openai anthropic transformers torch accelerate
pip install python-dotenv  # For environment variable management
```

## Prompt Engineering for Robotics

### Structured Task Decomposition Prompts

Effective prompt engineering is crucial for converting natural language to robotic actions:

```python
class RoboticsPromptEngineer:
    """Engineer effective prompts for robotics task decomposition"""

    def __init__(self):
        self.system_prompt = """
        You are an AI assistant specialized in robotics task planning. Your role is to convert natural language commands into structured robotic action plans.

        ## Capabilities
        - Navigate to locations (indoors/outdoors)
        - Manipulate objects (pick/place/grasp)
        - Interact with humans (follow, greet, assist)
        - Perform household tasks (cleaning, organizing)
        - Safety and obstacle avoidance

        ## Constraints
        - Always validate safety before action
        - Consider robot physical limitations
        - Handle ambiguous commands gracefully
        - Prioritize human safety above all
        """

    def create_task_decomposition_prompt(self, user_command: str, environment_context: dict):
        """Create a structured prompt for task decomposition"""

        prompt = f"""
        {self.system_prompt}

        ## Environment Context
        - Current robot position: {environment_context.get('position', 'unknown')}
        - Nearby objects: {environment_context.get('objects', [])}
        - Available actions: {environment_context.get('capabilities', [])}
        - Safety constraints: {environment_context.get('safety_constraints', [])}

        ## User Command
        "{user_command}"

        ## Response Format
        Provide your response in the following JSON format:
        {{
            "original_command": "...",
            "intent_classification": "...",
            "decomposed_tasks": [
                {{
                    "task_id": "...",
                    "description": "...",
                    "action_type": "...",
                    "parameters": {{}},
                    "estimated_duration": "...",
                    "safety_check_required": true,
                    "dependencies": []
                }}
            ],
            "confidence_score": 0.0-1.0,
            "clarification_needed": false,
            "reasoning": "..."
        }}

        ## Examples
        User: "Go to the kitchen and bring me a cup of water"
        Response: {{
            "original_command": "Go to the kitchen and bring me a cup of water",
            "intent_classification": "fetch_object_with_navigation",
            "decomposed_tasks": [
                {{
                    "task_id": "nav_to_kitchen",
                    "description": "Navigate to kitchen location",
                    "action_type": "navigation",
                    "parameters": {{"target_location": "kitchen"}},
                    "estimated_duration": "2-3 minutes",
                    "safety_check_required": true,
                    "dependencies": []
                }},
                {{
                    "task_id": "locate_cup",
                    "description": "Locate and identify cup in kitchen",
                    "action_type": "object_detection",
                    "parameters": {{"object_type": "cup"}},
                    "estimated_duration": "30 seconds",
                    "safety_check_required": true,
                    "dependencies": ["nav_to_kitchen"]
                }}
            ],
            "confidence_score": 0.85,
            "clarification_needed": false,
            "reasoning": "Command involves navigation to kitchen and object manipulation to fetch water."
        }}
        """

        return prompt

    def create_safety_validation_prompt(self, proposed_actions: list, environment_state: dict):
        """Create safety validation prompts for proposed actions"""

        safety_prompt = f"""
        Evaluate the following proposed robotic actions for safety compliance:

        ## Environment State
        - Human presence: {environment_state.get('humans_nearby', 'unknown')}
        - Obstacles: {environment_state.get('obstacles', [])}
        - Robot health: {environment_state.get('robot_status', 'normal')}
        - Safety zones: {environment_state.get('safety_zones', [])}

        ## Proposed Actions
        {proposed_actions}

        ## Safety Requirements
        1. No actions that could harm humans
        2. No actions that could damage robot or environment
        3. Compliance with operational constraints
        4. Emergency stop capabilities maintained

        ## Response Format
        {{
            "actions_safe": true/false,
            "safety_issues": [...],
            "recommended_modifications": [...],
            "risk_assessment": "low/medium/high",
            "validation_reasoning": "..."
        }}
        """

        return safety_prompt
```

### Context-Aware Prompting

Implement context management for ongoing conversations:

```python
import json
from datetime import datetime, timedelta

class ContextManager:
    """Manage conversation and environmental context for LLM interactions"""

    def __init__(self, max_context_length: int = 10):
        self.conversation_history = []
        self.environment_memory = {}
        self.object_locations = {}
        self.task_states = {}
        self.max_context_length = max_context_length

    def add_conversation_turn(self, user_input: str, ai_response: str, timestamp: datetime = None):
        """Add a conversation turn to history"""
        if timestamp is None:
            timestamp = datetime.now()

        turn = {
            'timestamp': timestamp.isoformat(),
            'user_input': user_input,
            'ai_response': ai_response,
            'turn_id': len(self.conversation_history)
        }

        self.conversation_history.append(turn)

        # Maintain context window size
        if len(self.conversation_history) > self.max_context_length:
            self.conversation_history = self.conversation_history[-self.max_context_length:]

    def update_environment_memory(self, updates: dict):
        """Update environmental context"""
        self.environment_memory.update(updates)
        self._cleanup_old_memory()

    def update_object_location(self, object_name: str, location: dict, confidence: float = 1.0):
        """Update known object locations"""
        self.object_locations[object_name] = {
            'location': location,
            'confidence': confidence,
            'last_seen': datetime.now().isoformat(),
            'tracking_id': f"{object_name}_{len(self.object_locations)}"
        }

    def get_context_for_prompt(self) -> dict:
        """Get current context for LLM prompting"""
        return {
            'conversation_history': self.conversation_history[-3:],  # Last 3 turns
            'environment_state': self.environment_memory,
            'known_objects': self.object_locations,
            'ongoing_tasks': self.task_states,
            'current_time': datetime.now().isoformat()
        }

    def _cleanup_old_memory(self):
        """Remove outdated environmental memory"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=1)  # Keep 1 hour of memory

        # Clean up object locations older than 1 hour
        old_objects = []
        for obj_name, obj_data in self.object_locations.items():
            obj_time = datetime.fromisoformat(obj_data['last_seen'])
            if obj_time < cutoff_time:
                old_objects.append(obj_name)

        for obj in old_objects:
            del self.object_locations[obj]
```

## Cognitive Planning Implementation

### Task Decomposition Engine

Implement the core task decomposition system:

```python
import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    NAVIGATION = "navigation"
    OBJECT_DETECTION = "object_detection"
    MANIPULATION = "manipulation"
    INTERACTION = "interaction"
    MONITORING = "monitoring"
    SAFETY_CHECK = "safety_check"

@dataclass
class Task:
    task_id: str
    description: str
    action_type: ActionType
    parameters: Dict[str, Any]
    estimated_duration: str
    safety_check_required: bool
    dependencies: List[str]
    priority: int = 1
    confidence_threshold: float = 0.7

class TaskDecompositionEngine:
    """Core engine for decomposing natural language commands into robotic tasks"""

    def __init__(self, llm_client, prompt_engineer: RoboticsPromptEngineer):
        self.llm_client = llm_client
        self.prompt_engineer = prompt_engineer
        self.context_manager = ContextManager()

    async def decompose_command(self, user_command: str, environment_context: dict) -> List[Task]:
        """Decompose a user command into executable tasks"""

        # Create prompt with context
        prompt = self.prompt_engineer.create_task_decomposition_prompt(
            user_command, environment_context
        )

        # Call LLM for task decomposition
        response = await self._call_llm(prompt)

        # Parse and validate response
        parsed_response = self._parse_task_response(response)

        # Convert to Task objects
        tasks = self._convert_to_tasks(parsed_response['decomposed_tasks'])

        # Store in context
        self.context_manager.add_conversation_turn(
            user_command,
            json.dumps([t.__dict__ for t in tasks], indent=2)
        )

        return tasks

    def _parse_task_response(self, llm_response: str) -> dict:
        """Parse LLM response into structured task format"""
        try:
            # Try to find JSON in response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = llm_response[start_idx:end_idx]
                parsed = json.loads(json_str)

                # Validate required fields
                required_fields = ['decomposed_tasks', 'confidence_score']
                for field in required_fields:
                    if field not in parsed:
                        raise ValueError(f"Missing required field: {field}")

                return parsed
            else:
                raise ValueError("No valid JSON found in LLM response")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}")

    def _convert_to_tasks(self, task_dicts: List[dict]) -> List[Task]:
        """Convert dictionary representations to Task objects"""
        tasks = []
        for task_dict in task_dicts:
            try:
                task = Task(
                    task_id=task_dict['task_id'],
                    description=task_dict['description'],
                    action_type=ActionType(task_dict['action_type']),
                    parameters=task_dict.get('parameters', {}),
                    estimated_duration=task_dict.get('estimated_duration', 'unknown'),
                    safety_check_required=task_dict.get('safety_check_required', True),
                    dependencies=task_dict.get('dependencies', []),
                    priority=task_dict.get('priority', 1),
                    confidence_threshold=task_dict.get('confidence_threshold', 0.7)
                )
                tasks.append(task)
            except (KeyError, ValueError) as e:
                print(f"Error converting task: {e}, skipping: {task_dict}")
                continue

        return tasks

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with proper error handling and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that responds in JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"LLM call failed, retrying... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(1)
```

### Safety and Validation Layer

Implement safety validation for LLM-generated actions:

```python
class SafetyValidator:
    """Validate LLM-generated actions for safety compliance"""

    def __init__(self, llm_client, prompt_engineer: RoboticsPromptEngineer):
        self.llm_client = llm_client
        self.prompt_engineer = prompt_engineer
        self.safety_rules = self._load_safety_rules()

    def _load_safety_rules(self) -> dict:
        """Load predefined safety rules"""
        return {
            "human_safety": [
                "Maintain safe distance from humans (minimum 1 meter)",
                "Avoid sudden movements near humans",
                "Stop immediately if human enters danger zone"
            ],
            "robot_safety": [
                "Don't exceed joint angle limits",
                "Monitor temperature and power consumption",
                "Avoid collisions with obstacles"
            ],
            "environmental_safety": [
                "Respect restricted areas",
                "Handle fragile objects carefully",
                "Follow navigation constraints"
            ]
        }

    async def validate_action_plan(self, tasks: List[Task], environment_state: dict) -> dict:
        """Validate action plan against safety requirements"""

        # Create safety validation prompt
        safety_prompt = self.prompt_engineer.create_safety_validation_prompt(
            [task.__dict__ for task in tasks],
            environment_state
        )

        # Call LLM for safety evaluation
        response = await self._call_llm(safety_prompt)

        # Parse safety response
        try:
            safety_analysis = json.loads(response)
        except json.JSONDecodeError:
            safety_analysis = {
                "actions_safe": False,
                "safety_issues": ["Could not parse safety analysis"],
                "recommended_modifications": [],
                "risk_assessment": "high",
                "validation_reasoning": "Failed to parse LLM safety response"
            }

        # Apply internal safety rules
        internal_validation = self._apply_internal_safety_rules(tasks, environment_state)

        # Combine results
        combined_result = {
            "llm_analysis": safety_analysis,
            "internal_validation": internal_validation,
            "overall_safe": safety_analysis["actions_safe"] and internal_validation["passed"],
            "blocking_issues": self._find_blocking_issues(safety_analysis, internal_validation)
        }

        return combined_result

    def _apply_internal_safety_rules(self, tasks: List[Task], environment_state: dict) -> dict:
        """Apply internal safety rules to task list"""
        issues = []
        passed = True

        for task in tasks:
            if task.action_type == ActionType.NAVIGATION:
                # Check navigation safety
                target_loc = task.parameters.get('target_location')
                if target_loc in environment_state.get('restricted_areas', []):
                    issues.append(f"Navigation to restricted area: {target_loc}")
                    passed = False

            elif task.action_type == ActionType.MANIPULATION:
                # Check manipulation safety
                obj_type = task.parameters.get('object_type')
                if obj_type in environment_state.get('fragile_objects', []):
                    issues.append(f"Manipulation of fragile object: {obj_type}")

        return {
            "passed": passed,
            "issues": issues,
            "rules_applied": len(self.safety_rules)
        }

    def _find_blocking_issues(self, llm_analysis: dict, internal_validation: dict) -> list:
        """Find issues that block action execution"""
        blocking_issues = []

        # Add LLM-identified safety issues
        if not llm_analysis.get("actions_safe", True):
            blocking_issues.extend(llm_analysis.get("safety_issues", []))

        # Add internal validation issues
        if not internal_validation["passed"]:
            blocking_issues.extend(internal_validation["issues"])

        return blocking_issues

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for safety validation"""
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a safety validator for robotic systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Warning: Safety validation LLM call failed: {e}")
            # Return a safe default response
            return json.dumps({
                "actions_safe": False,
                "safety_issues": ["Could not contact safety validator"],
                "recommended_modifications": [],
                "risk_assessment": "high",
                "validation_reasoning": "Safety validation service unavailable"
            })
```

## ROS 2 Integration

### Action Client Interface

Connect LLM outputs to ROS 2 action execution:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from builtin_interfaces.msg import Duration

from nav2_msgs.action import NavigateToPose
from moveit_msgs.action import MoveGroup
from manipulation_msgs.action import PickObject, PlaceObject

class LLMActionExecutor(Node):
    """Execute LLM-generated tasks through ROS 2 action interfaces"""

    def __init__(self):
        super().__init__('llm_action_executor')

        # Action clients for different capabilities
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.moveit_client = ActionClient(self, MoveGroup, 'move_group')
        self.pick_client = ActionClient(self, PickObject, 'pick_object')
        self.place_client = ActionClient(self, PlaceObject, 'place_object')

        # Publishers for status updates
        self.status_pub = self.create_publisher(String, 'llm_execution_status', 10)

        # Task queue for execution
        self.execution_queue = []
        self.current_task = None

        # Timer for task execution loop
        self.timer = self.create_timer(0.1, self._execution_loop)

    def execute_task_list(self, tasks: List[Task]):
        """Execute a list of tasks generated by LLM"""
        self.execution_queue.extend(tasks)
        self.get_logger().info(f"Added {len(tasks)} tasks to execution queue")

    def _execution_loop(self):
        """Main execution loop for processing tasks"""
        if self.current_task is None and self.execution_queue:
            # Start next task
            self.current_task = self.execution_queue.pop(0)
            self._execute_current_task()
        elif self.current_task:
            # Check if current task is complete
            if self._is_task_complete():
                self._complete_current_task()

    def _execute_current_task(self):
        """Execute the current task based on its type"""
        if not self.current_task:
            return

        task = self.current_task
        self.get_logger().info(f"Executing task: {task.description}")

        if task.action_type == ActionType.NAVIGATION:
            self._execute_navigation_task(task)
        elif task.action_type == ActionType.OBJECT_DETECTION:
            self._execute_detection_task(task)
        elif task.action_type == ActionType.MANIPULATION:
            self._execute_manipulation_task(task)
        elif task.action_type == ActionType.INTERACTION:
            self._execute_interaction_task(task)
        elif task.action_type == ActionType.SAFETY_CHECK:
            self._execute_safety_check_task(task)

    def _execute_navigation_task(self, task: Task):
        """Execute navigation task"""
        goal_msg = NavigateToPose.Goal()

        # Set target pose from task parameters
        pose_param = task.parameters.get('target_pose')
        if pose_param:
            goal_msg.pose.header.frame_id = pose_param.get('frame_id', 'map')
            goal_msg.pose.pose.position.x = pose_param['position']['x']
            goal_msg.pose.pose.position.y = pose_param['position']['y']
            goal_msg.pose.pose.position.z = pose_param['position']['z']
            goal_msg.pose.pose.orientation.w = pose_param['orientation']['w']
        else:
            # Use location name lookup
            location_name = task.parameters.get('target_location')
            location_pose = self._lookup_location(location_name)
            if location_pose:
                goal_msg.pose = location_pose

        # Send navigation goal
        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._navigation_completed)

    def _execute_manipulation_task(self, task: Task):
        """Execute manipulation task"""
        if task.parameters.get('action') == 'pick':
            goal_msg = PickObject.Goal()
            goal_msg.object_name = task.parameters.get('object_name')
            goal_msg.object_pose = task.parameters.get('object_pose')

            self.pick_client.wait_for_server()
            future = self.pick_client.send_goal_async(goal_msg)
            future.add_done_callback(self._pick_completed)

        elif task.parameters.get('action') == 'place':
            goal_msg = PlaceObject.Goal()
            goal_msg.target_pose = task.parameters.get('target_pose')

            self.place_client.wait_for_server()
            future = self.place_client.send_goal_async(goal_msg)
            future.add_done_callback(self._place_completed)

    def _is_task_complete(self) -> bool:
        """Check if current task is complete"""
        # Implementation depends on task type and feedback
        # This is a simplified version
        return True  # Placeholder - implement based on actual task feedback

    def _complete_current_task(self):
        """Complete current task and move to next"""
        if self.current_task:
            self.get_logger().info(f"Completed task: {self.current_task.description}")
            self.current_task = None

    def _lookup_location(self, location_name: str) -> PoseStamped:
        """Lookup predefined location by name"""
        # This would typically come from a map or configuration
        locations = {
            'kitchen': PoseStamped(),
            'living_room': PoseStamped(),
            'bedroom': PoseStamped(),
            # ... other locations
        }
        return locations.get(location_name)

    def _navigation_completed(self, future):
        """Handle navigation completion"""
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Navigation goal accepted')

    def _pick_completed(self, future):
        """Handle pick completion"""
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Pick goal accepted')

    def _place_completed(self, future):
        """Handle place completion"""
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Place goal accepted')
```

### Main Integration Node

Create the main integration node that connects all components:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped

class VLAIntegrationNode(Node):
    """Main integration node for Vision-Language-Action system"""

    def __init__(self):
        super().__init__('vla_integration_node')

        # Initialize components
        self.llm_client = self._initialize_llm_client()
        self.prompt_engineer = RoboticsPromptEngineer()
        self.task_decomposer = TaskDecompositionEngine(self.llm_client, self.prompt_engineer)
        self.safety_validator = SafetyValidator(self.llm_client, self.prompt_engineer)
        self.action_executor = LLMActionExecutor()

        # Subscriptions
        self.voice_command_sub = self.create_subscription(
            String, 'voice_command', self.voice_command_callback, 10)
        self.environment_sub = self.create_subscription(
            String, 'environment_state', self.environment_callback, 10)
        self.robot_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, 'amcl_pose', self.pose_callback, 10)

        # Publishers
        self.response_pub = self.create_publisher(String, 'vla_response', 10)
        self.task_status_pub = self.create_publisher(String, 'task_status', 10)

        # Internal state
        self.current_environment = {}
        self.robot_pose = None

        self.get_logger().info("VLA Integration Node initialized")

    def _initialize_llm_client(self):
        """Initialize LLM client based on environment configuration"""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        provider = os.getenv('LLM_PROVIDER', 'openai')

        if provider == 'openai':
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            return openai.AsyncOpenAI()
        elif provider == 'anthropic':
            import anthropic
            return anthropic.AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def process_voice_command(self, command: str):
        """Process a voice command through the full VLA pipeline"""
        try:
            # Get current environment context
            env_context = await self._get_environment_context()

            # Decompose command into tasks
            tasks = await self.task_decomposer.decompose_command(command, env_context)

            # Validate safety
            safety_result = await self.safety_validator.validate_action_plan(tasks, env_context)

            if not safety_result['overall_safe']:
                # Handle unsafe commands
                response_msg = String()
                response_msg.data = f"Command '{command}' is unsafe to execute: {safety_result['blocking_issues']}"
                self.response_pub.publish(response_msg)
                return

            # Execute safe tasks
            self.action_executor.execute_task_list(tasks)

            # Publish success response
            response_msg = String()
            response_msg.data = f"Processing command: {command}. Executing {len(tasks)} tasks."
            self.response_pub.publish(response_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing voice command: {e}")
            error_msg = String()
            error_msg.data = f"Error processing command: {str(e)}"
            self.response_pub.publish(error_msg)

    async def _get_environment_context(self) -> dict:
        """Get current environment context for LLM prompting"""
        return {
            'position': self.robot_pose,
            'objects': self.current_environment.get('objects', []),
            'capabilities': ['navigation', 'manipulation', 'interaction'],
            'safety_constraints': ['human_proximity', 'fragile_objects'],
            'restricted_areas': self.current_environment.get('restricted_areas', [])
        }

    def voice_command_callback(self, msg: String):
        """Handle incoming voice commands"""
        command = msg.data
        self.get_logger().info(f"Received voice command: {command}")

        # Process in separate thread to avoid blocking
        import threading
        thread = threading.Thread(target=lambda: rclpy.spin_until_future_complete(
            self, self.process_voice_command(command)))
        thread.start()

    def environment_callback(self, msg: String):
        """Handle environment updates"""
        try:
            env_data = json.loads(msg.data)
            self.current_environment.update(env_data)
        except json.JSONDecodeError:
            self.get_logger().warn("Invalid environment data received")

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """Handle robot pose updates"""
        self.robot_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'orientation': {
                'x': msg.pose.pose.orientation.x,
                'y': msg.pose.pose.orientation.y,
                'z': msg.pose.pose.orientation.z,
                'w': msg.pose.pose.orientation.w
            }
        }

def main(args=None):
    rclpy.init(args=args)
    node = VLAIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Implementation Steps

### 1. Set Up the LLM Integration Package

First, create the ROS 2 package structure:

```bash
# Create package
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python vla_llm_integration
cd vla_llm_integration
mkdir -p vla_llm_integration/config
```

### 2. Install Dependencies

Create and populate the requirements file:

```bash
# Create requirements.txt
cat > vla_llm_integration/requirements.txt << EOF
openai>=1.0.0
anthropic>=0.5.0
transformers>=4.35.0
torch>=2.0.0
python-dotenv>=1.0.0
numpy>=1.21.0
EOF
```

### 3. Configure the System

Create a launch file to bring up the VLA system:

```xml
<!-- vla_llm_integration/launch/vla_system.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('vla_llm_integration'), 'config')

    return LaunchDescription([
        Node(
            package='vla_llm_integration',
            executable='vla_integration_node',
            name='vla_integration_node',
            parameters=[],
            output='screen'
        ),
        Node(
            package='voice_recognition',
            executable='whisper_speech_recognizer',
            name='whisper_speech_recognizer',
            output='screen'
        )
    ])
```

### 4. Testing the Integration

Create a test script to verify the integration:

```python
#!/usr/bin/env python3
# test_vla_integration.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class VLATestClient(Node):
    def __init__(self):
        super().__init__('vla_test_client')
        self.publisher = self.create_publisher(String, 'voice_command', 10)
        self.timer = self.create_timer(5.0, self.send_test_commands)
        self.command_count = 0

    def send_test_commands(self):
        """Send test commands to VLA system"""
        test_commands = [
            "Navigate to the kitchen",
            "Find the red cup",
            "Bring me a glass of water"
        ]

        if self.command_count < len(test_commands):
            cmd = String()
            cmd.data = test_commands[self.command_count]
            self.publisher.publish(cmd)
            self.get_logger().info(f"Sent test command: {cmd.data}")
            self.command_count += 1

def main(args=None):
    rclpy.init(args=args)
    test_client = VLATestClient()

    try:
        rclpy.spin(test_client)
    except KeyboardInterrupt:
        pass
    finally:
        test_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices and Considerations

### 1. Prompt Optimization

- Use few-shot examples to guide LLM behavior
- Include explicit safety constraints in prompts
- Provide clear output formatting requirements
- Test prompts with edge cases and ambiguous commands

### 2. Safety First Approach

- Always validate LLM outputs before execution
- Implement multiple safety layers (LLM analysis + internal rules)
- Maintain emergency stop capabilities
- Log all safety decisions for audit purposes

### 3. Performance Optimization

- Cache LLM responses for repeated commands
- Use local models for low-latency responses when possible
- Implement proper error handling and retry logic
- Monitor token usage and costs

### 4. Context Management

- Maintain conversation history for context awareness
- Regularly clean up old context to prevent memory bloat
- Synchronize context between different system components
- Handle context conflicts gracefully

## Troubleshooting

### Common Issues

1. **LLM Rate Limiting**: Implement proper rate limiting and retry logic
2. **Context Length Limits**: Manage context size to stay within token limits
3. **Safety Validation Failures**: Fine-tune safety rules based on environment
4. **Task Decomposition Errors**: Improve prompt engineering for better parsing

### Debugging Tips

- Enable detailed logging for LLM interactions
- Monitor token usage and response times
- Validate JSON parsing of LLM responses
- Test with various command complexities

This LLM integration system provides the cognitive foundation for converting natural language commands into executable robotic actions, forming a crucial bridge between human communication and robot execution in Vision-Language-Action systems.