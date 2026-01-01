---
sidebar_position: 2
---

# Exercise 2: Robot Control Service for Humanoid Robot

## Overview

In this exercise, you'll create a ROS 2 service server and client that allows for remote control of a humanoid robot. Services provide a request-response communication pattern, which is ideal for control commands that require confirmation of completion.

## Learning Objectives

By completing this exercise, you will:
- Create a custom service definition for robot control
- Implement a service server that can control humanoid robot joints
- Create a service client to send control commands
- Understand the request-response pattern in ROS 2
- Practice error handling in service implementations

## Prerequisites

Before starting this exercise, ensure you have:
- ROS 2 Humble Hawksbill installed
- Basic Python programming knowledge
- Understanding of ROS 2 services and message definitions
- Completion of Exercise 1 (or basic understanding of publishers)

## Step 1: Creating Custom Service Definitions

First, let's create custom service definitions for humanoid robot control. Create the service definition files:

```bash
# In your package directory
mkdir -p humanoid_publisher_examples/srv
```

Create the service definition file for setting joint positions:

```yaml
# humanoid_publisher_examples/srv/SetJointPositions.srv
# Request
string[] joint_names
float64[] positions
float64 duration  # Time to reach target position in seconds

---
# Response
bool success
string message
float64[] achieved_positions
```

Create another service for executing predefined movements:

```yaml
# humanoid_publisher_examples/srv/ExecuteMovement.srv
# Request
string movement_type  # Options: 'wave', 'walk', 'sit', 'stand', 'dance'
float64 speed         # Speed multiplier (0.1 to 2.0)

---
# Response
bool success
string message
string executed_movement
```

## Step 2: Creating the Service Server

Create the service server that will handle robot control requests:

```python
# humanoid_publisher_examples/joint_control_server.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import math

# Import your custom service types (you'll need to generate them first)
# For now, we'll use a mock implementation
try:
    from humanoid_publisher_examples.srv import SetJointPositions, ExecuteMovement
except ImportError:
    # Mock service definitions for this example
    class SetJointPositions:
        class Request:
            def __init__(self):
                self.joint_names = []
                self.positions = []
                self.duration = 0.0

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
                self.achieved_positions = []

    class ExecuteMovement:
        class Request:
            def __init__(self):
                self.movement_type = ""
                self.speed = 1.0

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
                self.executed_movement = ""

class JointControlServer(Node):
    def __init__(self):
        super().__init__('joint_control_server')

        # Create publishers and services
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Create services
        self.set_positions_service = self.create_service(
            SetJointPositions,
            'set_joint_positions',
            self.handle_set_positions
        )

        self.execute_movement_service = self.create_service(
            ExecuteMovement,
            'execute_movement',
            self.handle_execute_movement
        )

        # Initialize robot state
        self.current_joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        self.current_positions = [0.0] * len(self.current_joint_names)
        self.target_positions = [0.0] * len(self.current_joint_names)
        self.moving = False

        # Timer for smooth joint movement
        self.movement_timer = self.create_timer(0.02, self.movement_callback)  # 50 Hz

        self.get_logger().info('Joint Control Server initialized')

    def handle_set_positions(self, request, response):
        """Handle set joint positions request"""
        self.get_logger().info(f'Received request to set {len(request.joint_names)} joint positions')

        try:
            # Validate request
            if len(request.joint_names) != len(request.positions):
                response.success = False
                response.message = 'Joint names and positions arrays must have the same length'
                return response

            if request.duration <= 0:
                response.success = False
                response.message = 'Duration must be positive'
                return response

            # Update target positions
            for i, joint_name in enumerate(request.joint_names):
                if joint_name in self.current_joint_names:
                    idx = self.current_joint_names.index(joint_name)
                    self.target_positions[idx] = request.positions[i]
                else:
                    response.success = False
                    response.message = f'Joint {joint_name} not found in robot model'
                    return response

            # Start smooth movement
            self.moving = True
            self.movement_start_time = self.get_clock().now().nanoseconds * 1e-9
            self.movement_duration = request.duration
            self.movement_start_positions = self.current_positions.copy()

            response.success = True
            response.message = f'Starting movement for {len(request.joint_names)} joints'
            response.achieved_positions = self.target_positions.copy()

            self.get_logger().info(f'Successfully started movement: {response.message}')

        except Exception as e:
            response.success = False
            response.message = f'Error processing request: {str(e)}'
            self.get_logger().error(response.message)

        return response

    def handle_execute_movement(self, request, response):
        """Handle execute predefined movement request"""
        self.get_logger().info(f'Received request to execute movement: {request.movement_type}')

        try:
            # Validate movement type
            valid_movements = ['wave', 'walk', 'sit', 'stand', 'dance', 'idle']
            if request.movement_type not in valid_movements:
                response.success = False
                response.message = f'Invalid movement type. Valid options: {valid_movements}'
                return response

            if request.speed < 0.1 or request.speed > 2.0:
                response.success = False
                response.message = 'Speed must be between 0.1 and 2.0'
                return response

            # Execute the movement pattern
            success, message = self.execute_movement_pattern(request.movement_type, request.speed)

            response.success = success
            response.message = message
            response.executed_movement = request.movement_type

            self.get_logger().info(f'Movement execution result: {message}')

        except Exception as e:
            response.success = False
            response.message = f'Error executing movement: {str(e)}'
            self.get_logger().error(response.message)

        return response

    def execute_movement_pattern(self, movement_type, speed):
        """Execute predefined movement patterns"""
        try:
            if movement_type == 'wave':
                # Set positions for waving motion
                self.target_positions[self.current_joint_names.index('right_shoulder_joint')] = 1.0
                self.target_positions[self.current_joint_names.index('right_elbow_joint')] = 1.0
                self.movement_duration = 2.0 / speed
            elif movement_type == 'walk':
                # Set positions for walking preparation
                self.target_positions[self.current_joint_names.index('left_hip_joint')] = 0.2
                self.target_positions[self.current_joint_names.index('right_hip_joint')] = 0.2
                self.movement_duration = 1.0 / speed
            elif movement_type == 'sit':
                # Set positions for sitting
                self.target_positions[self.current_joint_names.index('left_hip_joint')] = -1.0
                self.target_positions[self.current_joint_names.index('left_knee_joint')] = 1.5
                self.target_positions[self.current_joint_names.index('right_hip_joint')] = -1.0
                self.target_positions[self.current_joint_names.index('right_knee_joint')] = 1.5
                self.movement_duration = 3.0 / speed
            elif movement_type == 'stand':
                # Return to standing position
                for i in range(len(self.target_positions)):
                    self.target_positions[i] = 0.0
                self.movement_duration = 2.0 / speed
            elif movement_type == 'dance':
                # Set positions for a simple dance move
                self.target_positions[self.current_joint_names.index('left_shoulder_joint')] = 1.0
                self.target_positions[self.current_joint_names.index('right_shoulder_joint')] = 1.0
                self.target_positions[self.current_joint_names.index('left_hip_joint')] = 0.3
                self.target_positions[self.current_joint_names.index('right_hip_joint')] = -0.3
                self.movement_duration = 4.0 / speed
            elif movement_type == 'idle':
                # Return to neutral position
                for i in range(len(self.target_positions)):
                    self.target_positions[i] = 0.0
                self.movement_duration = 1.0 / speed

            # Start the movement
            self.moving = True
            self.movement_start_time = self.get_clock().now().nanoseconds * 1e-9
            self.movement_start_positions = self.current_positions.copy()

            return True, f'Started {movement_type} movement with speed {speed}'

        except ValueError as e:
            return False, f'Joint not found: {str(e)}'
        except Exception as e:
            return False, f'Error in movement pattern: {str(e)}'

    def movement_callback(self):
        """Timer callback to smoothly move joints to target positions"""
        if not self.moving:
            # Publish current positions if not moving
            self.publish_joint_states()
            return

        # Calculate movement progress
        current_time = self.get_clock().now().nanoseconds * 1e-9
        elapsed = current_time - self.movement_start_time

        if elapsed >= self.movement_duration:
            # Movement complete
            self.current_positions = self.target_positions.copy()
            self.moving = False
            self.get_logger().info('Movement completed')
        else:
            # Interpolate between start and target positions
            progress = elapsed / self.movement_duration
            # Use smooth interpolation (ease-in-out)
            smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))

            for i in range(len(self.current_positions)):
                self.current_positions[i] = (
                    self.movement_start_positions[i] +
                    smooth_progress * (self.target_positions[i] - self.movement_start_positions[i])
                )

        # Publish current joint states
        self.publish_joint_states()

    def publish_joint_states(self):
        """Publish the current joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.current_joint_names
        msg.position = self.current_positions
        msg.velocity = [0.0] * len(self.current_joint_names)
        msg.effort = [0.0] * len(self.current_joint_names)

        self.joint_publisher.publish(msg)

def main(args=None):
    """Main function to initialize and run the server"""
    rclpy.init(args=args)

    try:
        control_server = JointControlServer()
        control_server.get_logger().info('Starting joint control server...')

        rclpy.spin(control_server)

    except KeyboardInterrupt:
        print('\nShutting down joint control server...')
    finally:
        control_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 3: Creating the Service Client

Create a client that can call the services:

```python
# humanoid_publisher_examples/joint_control_client.py

import rclpy
from rclpy.node import Node
import time

# Import your custom service types
try:
    from humanoid_publisher_examples.srv import SetJointPositions, ExecuteMovement
except ImportError:
    # Mock service definitions for this example
    class SetJointPositions:
        class Request:
            def __init__(self):
                self.joint_names = []
                self.positions = []
                self.duration = 0.0

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
                self.achieved_positions = []

    class ExecuteMovement:
        class Request:
            def __init__(self):
                self.movement_type = ""
                self.speed = 1.0

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
                self.executed_movement = ""

class JointControlClient(Node):
    def __init__(self):
        super().__init__('joint_control_client')

        # Create clients for the services
        self.set_positions_client = self.create_client(
            SetJointPositions,
            'set_joint_positions'
        )

        self.execute_movement_client = self.create_client(
            ExecuteMovement,
            'execute_movement'
        )

        # Wait for services to be available
        while not self.set_positions_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set positions service not available, waiting again...')

        while not self.execute_movement_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Execute movement service not available, waiting again...')

        self.get_logger().info('Joint Control Client initialized and services available')

    def set_joint_positions(self, joint_names, positions, duration=2.0):
        """Send a request to set specific joint positions"""
        request = SetJointPositions.Request()
        request.joint_names = joint_names
        request.positions = positions
        request.duration = duration

        self.get_logger().info(f'Sending set positions request for {len(joint_names)} joints')

        # Make the service call
        future = self.set_positions_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response:
            if response.success:
                self.get_logger().info(f'Successfully set joint positions: {response.message}')
                self.get_logger().info(f'Achieved positions: {response.achieved_positions}')
            else:
                self.get_logger().error(f'Failed to set joint positions: {response.message}')
        else:
            self.get_logger().error('Service call failed')

        return response

    def execute_movement(self, movement_type, speed=1.0):
        """Send a request to execute a predefined movement"""
        request = ExecuteMovement.Request()
        request.movement_type = movement_type
        request.speed = speed

        self.get_logger().info(f'Sending execute movement request: {movement_type} at speed {speed}')

        # Make the service call
        future = self.execute_movement_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response:
            if response.success:
                self.get_logger().info(f'Successfully executed movement: {response.executed_movement}')
                self.get_logger().info(f'Result: {response.message}')
            else:
                self.get_logger().error(f'Failed to execute movement: {response.message}')
        else:
            self.get_logger().error('Service call failed')

        return response

    def demo_sequence(self):
        """Run a demonstration sequence of control commands"""
        self.get_logger().info('Starting demonstration sequence...')

        # 1. Move to standing position
        self.get_logger().info('Moving to standing position...')
        self.execute_movement('stand', speed=1.0)
        time.sleep(3)

        # 2. Wave with right arm
        self.get_logger().info('Waving with right arm...')
        self.execute_movement('wave', speed=1.0)
        time.sleep(3)

        # 3. Prepare to walk
        self.get_logger().info('Preparing to walk...')
        self.execute_movement('walk', speed=1.0)
        time.sleep(2)

        # 4. Sit down
        self.get_logger().info('Sitting down...')
        self.execute_movement('sit', speed=0.8)
        time.sleep(4)

        # 5. Stand up
        self.get_logger().info('Standing up...')
        self.execute_movement('stand', speed=1.0)
        time.sleep(3)

        # 6. Do a little dance
        self.get_logger().info('Dancing...')
        self.execute_movement('dance', speed=1.2)
        time.sleep(5)

        # 7. Return to idle
        self.get_logger().info('Returning to idle position...')
        self.execute_movement('idle', speed=1.0)

        self.get_logger().info('Demonstration sequence completed!')

def main(args=None):
    """Main function to initialize and run the client"""
    rclpy.init(args=args)

    try:
        client = JointControlClient()

        # Run the demonstration sequence
        client.demo_sequence()

    except KeyboardInterrupt:
        print('\nClient interrupted by user')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 4: Creating a Command-Line Interface Client

Create a more interactive client that allows manual control:

```python
# humanoid_publisher_examples/manual_control_client.py

import rclpy
from rclpy.node import Node
import sys
import select
import tty
import termios

try:
    from humanoid_publisher_examples.srv import SetJointPositions, ExecuteMovement
except ImportError:
    # Mock service definitions for this example
    class SetJointPositions:
        class Request:
            def __init__(self):
                self.joint_names = []
                self.positions = []
                self.duration = 0.0

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
                self.achieved_positions = []

    class ExecuteMovement:
        class Request:
            def __init__(self):
                self.movement_type = ""
                self.speed = 1.0

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
                self.executed_movement = ""

class ManualControlClient(Node):
    def __init__(self):
        super().__init__('manual_control_client')

        # Create clients for the services
        self.set_positions_client = self.create_client(
            SetJointPositions,
            'set_joint_positions'
        )

        self.execute_movement_client = self.create_client(
            ExecuteMovement,
            'execute_movement'
        )

        # Wait for services to be available
        while not self.set_positions_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set positions service not available, waiting again...')

        while not self.execute_movement_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Execute movement service not available, waiting again...')

        self.get_logger().info('Manual Control Client initialized')
        self.print_usage()

    def print_usage(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("HUMANOID ROBOT MANUAL CONTROL INTERFACE")
        print("="*60)
        print("Commands:")
        print("  w - Wave (right arm)")
        print("  W - Walk preparation")
        print("  s - Sit down")
        print("  S - Stand up")
        print("  d - Dance")
        print("  i - Idle position")
        print("  m - Move specific joints")
        print("  q - Quit")
        print("\nPress any command key to execute...")

    def execute_movement(self, movement_type, speed=1.0):
        """Execute a predefined movement"""
        request = ExecuteMovement.Request()
        request.movement_type = movement_type
        request.speed = speed

        future = self.execute_movement_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response and response.success:
            self.get_logger().info(f'✓ {movement_type.capitalize()} movement executed')
        else:
            error_msg = response.message if response else "Service call failed"
            self.get_logger().error(f'✗ Failed to execute {movement_type}: {error_msg}')

    def move_specific_joints(self):
        """Move specific joints to custom positions"""
        print("\nMoving specific joints...")
        print("Available joints: left/right_hip, left/right_knee, left/right_ankle, left/right_shoulder, left/right_elbow")

        try:
            joint_name = input("Enter joint name: ").strip()
            position = float(input("Enter position (in radians): "))
            duration = float(input("Enter duration (in seconds, default 2.0): ") or "2.0")

            request = SetJointPositions.Request()
            request.joint_names = [joint_name]
            request.positions = [position]
            request.duration = duration

            future = self.set_positions_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            response = future.result()
            if response and response.success:
                self.get_logger().info(f'✓ Moved {joint_name} to {position} radians')
            else:
                error_msg = response.message if response else "Service call failed"
                self.get_logger().error(f'✗ Failed to move joint: {error_msg}')

        except ValueError:
            self.get_logger().error("Invalid input. Please enter numeric values for position and duration.")
        except KeyboardInterrupt:
            print("\nJoint movement cancelled.")

    def run_interactive(self):
        """Run the interactive control loop"""
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())

            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)

                    if key.lower() == 'q':
                        print("\nQuitting...")
                        break
                    elif key.lower() == 'w':
                        self.execute_movement('wave')
                    elif key.lower() == 'w' and key.isupper():  # Shift+W
                        self.execute_movement('walk')
                    elif key.lower() == 's' and key.islower():  # s for sit
                        self.execute_movement('sit')
                    elif key.lower() == 's' and key.isupper():  # S for stand
                        self.execute_movement('stand')
                    elif key.lower() == 'd':
                        self.execute_movement('dance')
                    elif key.lower() == 'i':
                        self.execute_movement('idle')
                    elif key.lower() == 'm':
                        self.move_specific_joints()
                    else:
                        print(f"\nUnknown command: {key}. Press '?' for help.")
                        self.print_usage()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def main(args=None):
    """Main function to initialize and run the manual client"""
    rclpy.init(args=args)

    try:
        client = ManualControlClient()
        client.run_interactive()

    except KeyboardInterrupt:
        print('\nManual control interrupted by user')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 5: Updating setup.py

Add the new executables to your `setup.py`:

```python
# Add to entry_points in setup.py
entry_points={
    'console_scripts': [
        'simple_joint_publisher = humanoid_publisher_examples.simple_joint_publisher:main',
        'enhanced_joint_publisher = humanoid_publisher_examples.enhanced_joint_publisher:main',
        'joint_control_server = humanoid_publisher_examples.joint_control_server:main',
        'joint_control_client = humanoid_publisher_examples.joint_control_client:main',
        'manual_control_client = humanoid_publisher_examples.manual_control_client:main',
    ],
},
```

## Step 6: Building and Running

Build your package:

```bash
cd ~/humanoid_ws
colcon build --packages-select humanoid_publisher_examples
source install/setup.bash
```

## Step 7: Running the Service Server and Client

1. **Start the server** (in one terminal):
```bash
ros2 run humanoid_publisher_examples joint_control_server
```

2. **Run the demo client** (in another terminal):
```bash
ros2 run humanoid_publisher_examples joint_control_client
```

3. **Or run the manual client** (in another terminal):
```bash
ros2 run humanoid_publisher_examples manual_control_client
```

## Step 8: Testing the Services

You can also test the services using the command line:

```bash
# Test set_joint_positions service
ros2 service call /set_joint_positions humanoid_publisher_examples/srv/SetJointPositions "{
  joint_names: ['left_shoulder_joint', 'right_shoulder_joint'],
  positions: [0.5, 0.5],
  duration: 2.0
}"

# Test execute_movement service
ros2 service call /execute_movement humanoid_publisher_examples/srv/ExecuteMovement "{
  movement_type: 'wave',
  speed: 1.0
}"
```

## Step 9: Adding Error Handling and Validation

Enhance the server with better error handling:

```python
# humanoid_publisher_examples/robust_control_server.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import math
from threading import Lock

try:
    from humanoid_publisher_examples.srv import SetJointPositions, ExecuteMovement
except ImportError:
    # Mock service definitions
    class SetJointPositions:
        class Request:
            def __init__(self):
                self.joint_names = []
                self.positions = []
                self.duration = 0.0

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
                self.achieved_positions = []

    class ExecuteMovement:
        class Request:
            def __init__(self):
                self.movement_type = ""
                self.speed = 1.0

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
                self.executed_movement = ""

class RobustControlServer(Node):
    def __init__(self):
        super().__init__('robust_control_server')

        # Thread safety lock
        self.lock = Lock()

        # Create publishers and services
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', 10)

        self.set_positions_service = self.create_service(
            SetJointPositions,
            'set_joint_positions',
            self.handle_set_positions
        )

        self.execute_movement_service = self.create_service(
            ExecuteMovement,
            'execute_movement',
            self.handle_execute_movement
        )

        # Initialize robot state
        self.current_joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        self.current_positions = [0.0] * len(self.current_joint_names)
        self.target_positions = [0.0] * len(self.current_joint_names)
        self.moving = False
        self.active_movement = None

        # Timer for smooth joint movement
        self.movement_timer = self.create_timer(0.02, self.movement_callback)

        self.get_logger().info('Robust Control Server initialized')

    def handle_set_positions(self, request, response):
        """Handle set joint positions request with enhanced validation"""
        with self.lock:
            self.get_logger().info(f'Received request to set {len(request.joint_names)} joint positions')

            try:
                # Input validation
                if not request.joint_names or not request.positions:
                    response.success = False
                    response.message = 'Joint names and positions cannot be empty'
                    return response

                if len(request.joint_names) != len(request.positions):
                    response.success = False
                    response.message = f'Length mismatch: {len(request.joint_names)} names vs {len(request.positions)} positions'
                    return response

                if request.duration <= 0:
                    response.success = False
                    response.message = 'Duration must be positive'
                    return response

                if request.duration > 10.0:  # Maximum 10 seconds for safety
                    response.success = False
                    response.message = 'Duration cannot exceed 10 seconds for safety'
                    return response

                # Validate joint names and positions
                for i, joint_name in enumerate(request.joint_names):
                    if joint_name not in self.current_joint_names:
                        response.success = False
                        response.message = f'Invalid joint name: {joint_name}'
                        return response

                    # Validate position limits (example: -3.14 to 3.14 radians)
                    if abs(request.positions[i]) > 3.14:
                        response.success = False
                        response.message = f'Position out of range for {joint_name}: {request.positions[i]}'
                        return response

                # Update target positions
                for i, joint_name in enumerate(request.joint_names):
                    idx = self.current_joint_names.index(joint_name)
                    self.target_positions[idx] = request.positions[i]

                # Start smooth movement
                self.moving = True
                self.active_movement = 'custom'
                self.movement_start_time = self.get_clock().now().nanoseconds * 1e-9
                self.movement_duration = request.duration
                self.movement_start_positions = self.current_positions.copy()

                response.success = True
                response.message = f'Successfully started movement for {len(request.joint_names)} joints'
                response.achieved_positions = self.target_positions.copy()

                self.get_logger().info(f'Movement started: {response.message}')

            except Exception as e:
                response.success = False
                response.message = f'Unexpected error: {str(e)}'
                self.get_logger().error(f'Service error: {response.message}')
                self.get_logger().error(f'Error details: {type(e).__name__}: {e}')

            return response

    def handle_execute_movement(self, request, response):
        """Handle execute predefined movement with enhanced validation"""
        with self.lock:
            self.get_logger().info(f'Received request to execute movement: {request.movement_type}')

            try:
                # Validate input
                valid_movements = ['wave', 'walk', 'sit', 'stand', 'dance', 'idle', 'rest']
                if request.movement_type not in valid_movements:
                    response.success = False
                    response.message = f'Invalid movement type. Valid options: {valid_movements}'
                    return response

                if request.speed < 0.1 or request.speed > 3.0:
                    response.success = False
                    response.message = 'Speed must be between 0.1 and 3.0'
                    return response

                # Execute the movement pattern
                success, message = self.execute_movement_pattern(request.movement_type, request.speed)

                response.success = success
                response.message = message
                response.executed_movement = request.movement_type

                if success:
                    self.get_logger().info(f'Movement executed: {message}')
                else:
                    self.get_logger().error(f'Movement failed: {message}')

            except Exception as e:
                response.success = False
                response.message = f'Unexpected error in movement execution: {str(e)}'
                self.get_logger().error(f'Service error: {response.message}')

            return response

    def execute_movement_pattern(self, movement_type, speed):
        """Execute predefined movement patterns with safety checks"""
        try:
            # Define safe limits for each movement type
            if movement_type == 'wave':
                self.target_positions[self.current_joint_names.index('right_shoulder_joint')] = 1.0
                self.target_positions[self.current_joint_names.index('right_elbow_joint')] = 1.0
                self.movement_duration = 2.0 / speed
            elif movement_type == 'walk':
                self.target_positions[self.current_joint_names.index('left_hip_joint')] = 0.2
                self.target_positions[self.current_joint_names.index('right_hip_joint')] = 0.2
                self.movement_duration = 1.0 / speed
            elif movement_type == 'sit':
                # Safe sitting position
                self.target_positions[self.current_joint_names.index('left_hip_joint')] = -1.0
                self.target_positions[self.current_joint_names.index('left_knee_joint')] = 1.5
                self.target_positions[self.current_joint_names.index('right_hip_joint')] = -1.0
                self.target_positions[self.current_joint_names.index('right_knee_joint')] = 1.5
                self.movement_duration = 3.0 / speed
            elif movement_type == 'stand':
                # Return to neutral standing position
                for i in range(len(self.target_positions)):
                    self.target_positions[i] = 0.0
                self.movement_duration = 2.0 / speed
            elif movement_type == 'dance':
                # Safe dance position
                self.target_positions[self.current_joint_names.index('left_shoulder_joint')] = 0.8
                self.target_positions[self.current_joint_names.index('right_shoulder_joint')] = 0.8
                self.target_positions[self.current_joint_names.index('left_hip_joint')] = 0.2
                self.target_positions[self.current_joint_names.index('right_hip_joint')] = -0.2
                self.movement_duration = 4.0 / speed
            elif movement_type == 'idle' or movement_type == 'rest':
                # Return to safe rest position
                for i in range(len(self.target_positions)):
                    self.target_positions[i] = 0.0
                self.movement_duration = 1.0 / speed

            # Start the movement
            self.moving = True
            self.active_movement = movement_type
            self.movement_start_time = self.get_clock().now().nanoseconds * 1e-9
            self.movement_start_positions = self.current_positions.copy()

            return True, f'Started {movement_type} movement with speed {speed}'

        except ValueError as e:
            return False, f'Joint not found in robot model: {str(e)}'
        except Exception as e:
            return False, f'Error in movement pattern execution: {str(e)}'

    def movement_callback(self):
        """Timer callback to smoothly move joints with safety checks"""
        with self.lock:
            if not self.moving:
                # Publish current positions if not moving
                self.publish_joint_states()
                return

            # Calculate movement progress
            current_time = self.get_clock().now().nanoseconds * 1e-9
            elapsed = current_time - self.movement_start_time

            if elapsed >= self.movement_duration:
                # Movement complete
                self.current_positions = self.target_positions.copy()
                self.moving = False
                self.active_movement = None
                self.get_logger().info('Movement completed successfully')
            else:
                # Interpolate between start and target positions with smooth easing
                progress = min(elapsed / self.movement_duration, 1.0)  # Clamp to [0, 1]
                # Use smooth step function for natural movement
                smooth_progress = progress * progress * (3 - 2 * progress)

                for i in range(len(self.current_positions)):
                    self.current_positions[i] = (
                        self.movement_start_positions[i] +
                        smooth_progress * (self.target_positions[i] - self.movement_start_positions[i])
                    )

            # Publish current joint states
            self.publish_joint_states()

    def publish_joint_states(self):
        """Publish the current joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.current_joint_names
        msg.position = self.current_positions.copy()
        msg.velocity = [0.0] * len(self.current_joint_names)
        msg.effort = [0.0] * len(self.current_joint_names)

        self.joint_publisher.publish(msg)

def main(args=None):
    """Main function to initialize and run the robust server"""
    rclpy.init(args=args)

    try:
        control_server = RobustControlServer()
        control_server.get_logger().info('Starting robust joint control server...')

        rclpy.spin(control_server)

    except KeyboardInterrupt:
        print('\nShutting down robust control server...')
    finally:
        control_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 10: Adding the Robust Server to setup.py

Add the robust server to your `setup.py`:

```python
# Add to entry_points in setup.py
entry_points={
    'console_scripts': [
        'simple_joint_publisher = humanoid_publisher_examples.simple_joint_publisher:main',
        'enhanced_joint_publisher = humanoid_publisher_examples.enhanced_joint_publisher:main',
        'joint_control_server = humanoid_publisher_examples.joint_control_server:main',
        'robust_control_server = humanoid_publisher_examples.robust_control_server:main',
        'joint_control_client = humanoid_publisher_examples.joint_control_client:main',
        'manual_control_client = humanoid_publisher_examples.manual_control_client:main',
    ],
},
```

## Troubleshooting Tips

1. **Service not found**: Make sure the server is running before calling the client
2. **Import errors**: If you created custom .srv files, make sure to build them properly
3. **Permission errors**: Check file permissions on your Python scripts
4. **Connection timeouts**: Verify that both nodes are on the same ROS domain

## Next Steps

After completing this exercise, you should have a solid understanding of:
- Creating custom ROS 2 service definitions
- Implementing service servers for robot control
- Creating service clients to send commands
- Handling errors and validation in services
- Implementing smooth motion control for humanoid robots

In the next exercise, we'll explore creating URDF files for humanoid robots and integrating them with our control systems.