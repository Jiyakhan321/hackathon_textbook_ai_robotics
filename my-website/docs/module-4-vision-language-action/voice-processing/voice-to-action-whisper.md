---
sidebar_position: 2
---

# Voice-to-Action Using OpenAI Whisper

## Overview

Speech recognition is a critical component of natural human-robot interaction. This section covers implementing voice-to-action systems using OpenAI Whisper, focusing on real-time speech recognition and integration with ROS 2 for humanoid robot control. Whisper provides robust, multilingual speech recognition capabilities that are essential for creating intuitive voice interfaces.

## Speech Recognition Pipeline for Humanoid Robots

### 1. Audio Input and Preprocessing

Humanoid robots require specialized audio processing to handle the challenges of real-world environments:

```python
# audio_input_handler.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
import pyaudio
import numpy as np
import webrtcvad
from collections import deque
import threading
import time

class AudioInputHandler(Node):
    """
    Handle audio input for humanoid robot voice recognition
    """
    def __init__(self):
        super().__init__('audio_input_handler')

        # Audio configuration
        self.sample_rate = 16000  # Whisper works best at 16kHz
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16

        # Voice Activity Detection
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Aggressive mode for robotics

        # Audio buffers
        self.audio_buffer = deque(maxlen=30)  # 30 chunks for 1.9s of audio
        self.speech_segments = []
        self.is_listening = False

        # Publishers
        self.audio_pub = self.create_publisher(AudioData, '/audio_input', 10)
        self.speech_detected_pub = self.create_publisher(Bool, '/speech_detected', 10)
        self.voice_activity_pub = self.create_publisher(Bool, '/voice_activity', 10)

        # Initialize PyAudio
        self.audio_interface = pyaudio.PyAudio()

        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self.capture_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        # Create timer for voice activity detection
        self.vad_timer = self.create_timer(0.1, self.check_voice_activity)

        self.get_logger().info('Audio Input Handler initialized')

    def capture_audio(self):
        """
        Continuously capture audio from microphone
        """
        stream = self.audio_interface.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        self.get_logger().info('Audio capture started')

        while rclpy.ok():
            try:
                # Read audio chunk
                audio_chunk = stream.read(self.chunk_size, exception_on_overflow=False)

                # Convert to numpy array for processing
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

                # Add to buffer
                self.audio_buffer.append(audio_array)

                # Check for voice activity
                if self.is_voice_active(audio_chunk):
                    self.speech_segments.append(audio_chunk)
                    self.is_listening = True

                    # If we have accumulated enough speech, publish it
                    if len(self.speech_segments) > 50:  # ~3.2 seconds of speech
                        self.publish_speech_segment()
                        self.speech_segments = []
                else:
                    if self.is_listening and len(self.speech_segments) > 0:
                        # End of speech detected
                        self.publish_speech_segment()
                        self.speech_segments = []
                        self.is_listening = False

            except Exception as e:
                self.get_logger().error(f'Error in audio capture: {e}')
                time.sleep(0.1)

        stream.stop_stream()
        stream.close()

    def is_voice_active(self, audio_chunk):
        """
        Check if voice is active in audio chunk using VAD
        """
        try:
            # VAD requires 10, 20, or 30ms frames
            frame_duration = 20  # ms
            frame_size = int(self.sample_rate * frame_duration / 1000)

            # Split chunk into VAD-compatible frames
            for i in range(0, len(audio_chunk), frame_size):
                frame = audio_chunk[i:i+frame_size]
                if len(frame) == frame_size:
                    # VAD expects 16-bit PCM data
                    try:
                        is_speech = self.vad.is_speech(frame, self.sample_rate)
                        if is_speech:
                            return True
                    except:
                        continue
            return False
        except:
            return False

    def publish_speech_segment(self):
        """
        Publish accumulated speech segment
        """
        if not self.speech_segments:
            return

        # Combine all speech segments
        combined_audio = b''.join(self.speech_segments)

        # Create and publish audio message
        audio_msg = AudioData()
        audio_msg.data = combined_audio
        self.audio_pub.publish(audio_msg)

        # Publish speech detected flag
        speech_msg = Bool()
        speech_msg.data = True
        self.speech_detected_pub.publish(speech_msg)

        self.get_logger().info(f'Published speech segment: {len(combined_audio)} bytes')

    def check_voice_activity(self):
        """
        Periodically check for voice activity
        """
        voice_active = self.is_listening
        activity_msg = Bool()
        activity_msg.data = voice_active
        self.voice_activity_pub.publish(activity_msg)

def main(args=None):
    rclpy.init(args=args)
    audio_handler = AudioInputHandler()

    try:
        rclpy.spin(audio_handler)
    except KeyboardInterrupt:
        audio_handler.get_logger().info('Shutting down audio input handler')
    finally:
        audio_handler.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Whisper Integration with ROS 2

Implementing Whisper for real-time speech recognition:

```python
# whisper_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
import whisper
import torch
import numpy as np
import io
import wave
import tempfile
import os
from threading import Lock

class WhisperROSIntegration(Node):
    """
    Integrate OpenAI Whisper with ROS 2 for speech recognition
    """
    def __init__(self):
        super().__init__('whisper_integration')

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")  # Use "small" or "medium" for better accuracy
        self.get_logger().info('Whisper model loaded')

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_input',
            self.audio_callback,
            10
        )

        # Create publishers
        self.transcription_pub = self.create_publisher(String, '/transcription', 10)
        self.command_pub = self.create_publisher(String, '/voice_command', 10)
        self.processing_status_pub = self.create_publisher(Bool, '/whisper_processing', 10)

        # Processing state
        self.processing_lock = Lock()
        self.is_processing = False

        # Command keywords for humanoid robots
        self.command_keywords = [
            'move', 'go', 'walk', 'turn', 'stop', 'start',
            'pick', 'place', 'grab', 'release', 'take',
            'find', 'look', 'see', 'show', 'bring',
            'hello', 'hi', 'help', 'please', 'thank you'
        ]

        self.get_logger().info('Whisper Integration initialized')

    def audio_callback(self, msg):
        """
        Process incoming audio data with Whisper
        """
        if self.is_processing:
            self.get_logger().warn('Whisper is busy, dropping audio packet')
            return

        # Process audio in separate thread to avoid blocking
        processing_thread = threading.Thread(
            target=self.process_audio_with_whisper,
            args=(msg,)
        )
        processing_thread.start()

    def process_audio_with_whisper(self, audio_msg):
        """
        Process audio data using Whisper model
        """
        with self.processing_lock:
            self.is_processing = True

        # Publish processing status
        status_msg = Bool()
        status_msg.data = True
        self.processing_status_pub.publish(status_msg)

        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_msg.data, dtype=np.int16)

            # Normalize audio to float32
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Process with Whisper
            result = self.model.transcribe(
                audio_float,
                language='en',
                fp16=torch.cuda.is_available()  # Use fp16 if CUDA available
            )

            transcription = result['text'].strip()

            if transcription:
                self.get_logger().info(f'Transcribed: "{transcription}"')

                # Publish transcription
                trans_msg = String()
                trans_msg.data = transcription
                self.transcription_pub.publish(trans_msg)

                # Check if this is a command
                if self.is_command(transcription):
                    cmd_msg = String()
                    cmd_msg.data = transcription
                    self.command_pub.publish(cmd_msg)
                    self.get_logger().info(f'Command detected: "{transcription}"')

        except Exception as e:
            self.get_logger().error(f'Error in Whisper processing: {e}')

        finally:
            # Reset processing state
            with self.processing_lock:
                self.is_processing = False

            # Publish processing status
            status_msg = Bool()
            status_msg.data = False
            self.processing_status_pub.publish(status_msg)

    def is_command(self, transcription):
        """
        Check if transcription contains robot commands
        """
        transcription_lower = transcription.lower()

        # Check for command keywords
        for keyword in self.command_keywords:
            if keyword in transcription_lower:
                return True

        # Check for action verbs
        action_verbs = ['move', 'go', 'walk', 'turn', 'stop', 'start', 'pick', 'place']
        for verb in action_verbs:
            if verb in transcription_lower:
                return True

        return False

def main(args=None):
    rclpy.init(args=args)
    whisper_node = WhisperROSIntegration()

    try:
        rclpy.spin(whisper_node)
    except KeyboardInterrupt:
        whisper_node.get_logger().info('Shutting down Whisper integration')
    finally:
        whisper_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3. Voice Command Validation and Safety

Implementing safety checks for voice commands:

```python
# voice_command_validator.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
import json
import re
from enum import Enum

class SafetyLevel(Enum):
    SAFE = 0
    WARNING = 1
    DANGEROUS = 2

class VoiceCommandValidator(Node):
    """
    Validate voice commands for safety and correctness
    """
    def __init__(self):
        super().__init__('voice_command_validator')

        # Create subscribers
        self.command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        # Create publishers
        self.validated_command_pub = self.create_publisher(String, '/validated_command', 10)
        self.safety_status_pub = self.create_publisher(String, '/command_safety_status', 10)
        self.request_clarification_pub = self.create_publisher(String, '/request_clarification', 10)

        # Safety configuration
        self.dangerous_commands = [
            'shutdown', 'power off', 'emergency stop', 'danger', 'harm', 'damage'
        ]

        self.dangerous_objects = [
            'fire', 'hot', 'sharp', 'dangerous', 'poison', 'toxic'
        ]

        # Movement constraints
        self.max_distance = 10.0  # meters
        self.max_height = 2.0    # meters

        self.get_logger().info('Voice Command Validator initialized')

    def command_callback(self, msg):
        """
        Process incoming voice command with validation
        """
        command = msg.data.strip()

        self.get_logger().info(f'Validating command: "{command}"')

        # Analyze command for safety
        safety_level, safety_issues = self.analyze_safety(command)

        if safety_level == SafetyLevel.DANGEROUS:
            self.get_logger().error(f'Dangerous command blocked: {command}')
            self.publish_safety_status(f'DANGEROUS: {"; ".join(safety_issues)}')
            return
        elif safety_level == SafetyLevel.WARNING:
            self.get_logger().warn(f'Warning for command: {command} - {"; ".join(safety_issues)}')
            self.request_clarification(command, safety_issues)
        else:
            self.get_logger().info(f'Command validated: {command}')
            self.publish_safety_status('SAFE')
            self.validated_command_pub.publish(msg)

    def analyze_safety(self, command):
        """
        Analyze command for safety issues
        """
        safety_issues = []

        # Check for dangerous keywords
        for dangerous_cmd in self.dangerous_commands:
            if dangerous_cmd.lower() in command.lower():
                safety_issues.append(f'Dangerous keyword: {dangerous_cmd}')

        # Check for dangerous objects
        for dangerous_obj in self.dangerous_objects:
            if dangerous_obj.lower() in command.lower():
                safety_issues.append(f'Dangerous object: {dangerous_obj}')

        # Check for movement commands that exceed limits
        distance = self.extract_distance(command)
        if distance and distance > self.max_distance:
            safety_issues.append(f'Distance too far: {distance}m > {self.max_distance}m')

        height = self.extract_height(command)
        if height and height > self.max_height:
            safety_issues.append(f'Height too high: {height}m > {self.max_height}m')

        # Check for ambiguous commands
        if self.is_ambiguous(command):
            safety_issues.append('Command is ambiguous')

        if safety_issues:
            if any('DANGEROUS' in issue for issue in safety_issues):
                return SafetyLevel.DANGEROUS, safety_issues
            else:
                return SafetyLevel.WARNING, safety_issues

        return SafetyLevel.SAFE, []

    def extract_distance(self, command):
        """
        Extract distance from command using regex
        """
        # Look for patterns like "go 5 meters", "move 3 feet", etc.
        distance_patterns = [
            r'(\d+(?:\.\d+)?)\s*meters?',  # meters
            r'(\d+(?:\.\d+)?)\s*feet',     # feet
            r'(\d+(?:\.\d+)?)\s*steps?',   # steps
        ]

        for pattern in distance_patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                distance = float(match.group(1))
                # Convert feet to meters if needed
                if 'feet' in match.group(0).lower():
                    distance *= 0.3048
                return distance

        return None

    def extract_height(self, command):
        """
        Extract height from command
        """
        height_patterns = [
            r'(\d+(?:\.\d+)?)\s*meters?\s*(?:high|tall|up)',
            r'(\d+(?:\.\d+)?)\s*feet\s*(?:high|tall|up)',
        ]

        for pattern in height_patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                height = float(match.group(1))
                if 'feet' in match.group(0).lower():
                    height *= 0.3048
                return height

        return None

    def is_ambiguous(self, command):
        """
        Check if command is ambiguous
        """
        ambiguous_indicators = [
            'that', 'there', 'it', 'thing', 'object', 'stuff',
            'somewhere', 'anywhere', 'here', 'there'
        ]

        command_lower = command.lower()
        for indicator in ambiguous_indicators:
            if indicator in command_lower:
                # Check if there's specific context
                if not any(specific in command_lower for specific in ['kitchen', 'table', 'chair', 'door', 'window']):
                    return True

        return False

    def request_clarification(self, command, issues):
        """
        Request clarification for ambiguous or potentially unsafe commands
        """
        clarification_msg = String()
        clarification_msg.data = f'Command "{command}" needs clarification: {"; ".join(issues)}'
        self.request_clarification_pub.publish(clarification_msg)

    def publish_safety_status(self, status):
        """
        Publish safety status
        """
        status_msg = String()
        status_msg.data = status
        self.safety_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    validator = VoiceCommandValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Shutting down voice command validator')
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Real-time Voice Processing Pipeline

### 1. Complete Voice Processing System

Creating a complete voice processing pipeline:

```python
# voice_processing_pipeline.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import PoseStamped
import threading
import queue
import time
from collections import deque

class VoiceProcessingPipeline(Node):
    """
    Complete voice processing pipeline for humanoid robots
    """
    def __init__(self):
        super().__init__('voice_processing_pipeline')

        # Initialize components
        self.command_queue = queue.Queue(maxsize=10)
        self.response_queue = queue.Queue(maxsize=10)
        self.command_history = deque(maxlen=50)

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_input',
            self.audio_callback,
            10
        )

        self.transcription_sub = self.create_subscription(
            String,
            '/transcription',
            self.transcription_callback,
            10
        )

        self.validated_command_sub = self.create_subscription(
            String,
            '/validated_command',
            self.validated_command_callback,
            10
        )

        # Create publishers
        self.response_pub = self.create_publisher(String, '/robot_response', 10)
        self.status_pub = self.create_publisher(String, '/voice_pipeline_status', 10)

        # Pipeline state
        self.is_active = True
        self.pipeline_status = "IDLE"

        # Start processing threads
        self.command_processing_thread = threading.Thread(
            target=self.process_commands
        )
        self.command_processing_thread.daemon = True
        self.command_processing_thread.start()

        self.get_logger().info('Voice Processing Pipeline initialized')

    def audio_callback(self, msg):
        """
        Handle incoming audio data
        """
        self.update_status("RECEIVING_AUDIO")
        self.get_logger().debug('Audio received')

    def transcription_callback(self, msg):
        """
        Handle incoming transcriptions
        """
        self.update_status("PROCESSING_TRANSCRIPTION")
        self.get_logger().info(f'Transcription received: {msg.data}')

        # Add to command queue for processing
        try:
            self.command_queue.put_nowait(msg.data)
        except queue.Full:
            self.get_logger().warn('Command queue full, dropping transcription')

    def validated_command_callback(self, msg):
        """
        Handle validated commands
        """
        self.update_status("EXECUTING_COMMAND")
        self.get_logger().info(f'Executing validated command: {msg.data}')

        # Add to execution queue
        try:
            self.command_queue.put_nowait(f"EXECUTE: {msg.data}")
        except queue.Full:
            self.get_logger().warn('Execution queue full')

    def process_commands(self):
        """
        Process commands from queue
        """
        while self.is_active:
            try:
                command = self.command_queue.get(timeout=1.0)

                if command.startswith("EXECUTE: "):
                    actual_command = command[9:]  # Remove "EXECUTE: " prefix
                    self.execute_command(actual_command)
                else:
                    self.process_command(command)

                self.command_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error processing command: {e}')

    def process_command(self, command):
        """
        Process a voice command
        """
        self.get_logger().info(f'Processing command: {command}')

        # Add to history
        self.command_history.append({
            'command': command,
            'timestamp': time.time()
        })

        # Generate response
        response = self.generate_response(command)

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        self.get_logger().info(f'Response: {response}')

    def execute_command(self, command):
        """
        Execute a validated command
        """
        self.get_logger().info(f'Executing command: {command}')

        # This is where you would integrate with the robot's action system
        # For now, just log the execution
        self.update_status(f"EXECUTING: {command}")

        # Simulate command execution
        time.sleep(0.5)  # Simulate execution time

        # Publish completion
        response_msg = String()
        response_msg.data = f"Command '{command}' executed successfully"
        self.response_pub.publish(response_msg)

    def generate_response(self, command):
        """
        Generate appropriate response for a command
        """
        command_lower = command.lower()

        if any(greeting in command_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        elif any(help_request in command_lower for help_request in ['help', 'what can you do']):
            return "I can navigate, pick up objects, and respond to voice commands. What would you like me to do?"
        elif any(thanks in command_lower for thanks in ['thank you', 'thanks']):
            return "You're welcome! Is there anything else I can help with?"
        else:
            return f"I understand you said: '{command}'. How should I respond to this?"

    def update_status(self, status):
        """
        Update and publish pipeline status
        """
        self.pipeline_status = status
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    pipeline = VoiceProcessingPipeline()

    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pipeline.get_logger().info('Shutting down voice processing pipeline')
        pipeline.is_active = False
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### 1. Efficient Whisper Processing

Optimizing Whisper for real-time performance:

```python
# efficient_whisper_processor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
import whisper
import torch
import numpy as np
from threading import Lock, Thread
import time
from collections import deque

class EfficientWhisperProcessor(Node):
    """
    Efficient Whisper processor optimized for real-time humanoid applications
    """
    def __init__(self):
        super().__init__('efficient_whisper_processor')

        # Initialize Whisper model with optimizations
        self.get_logger().info('Loading optimized Whisper model...')

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("base").to(device)

        # Set model to evaluation mode
        self.model.eval()

        self.get_logger().info(f'Whisper model loaded on {device}')

        # Create subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_input',
            self.audio_callback,
            10
        )

        # Create publishers
        self.transcription_pub = self.create_publisher(String, '/transcription', 10)
        self.performance_pub = self.create_publisher(String, '/whisper_performance', 10)

        # Processing optimization
        self.processing_lock = Lock()
        self.is_processing = False
        self.processing_times = deque(maxlen=100)  # Track last 100 processing times

        # Audio processing parameters
        self.sample_rate = 16000
        self.max_audio_duration = 10.0  # Maximum 10 seconds for processing

        # Create timer for performance monitoring
        self.performance_timer = self.create_timer(5.0, self.report_performance)

        self.get_logger().info('Efficient Whisper Processor initialized')

    def audio_callback(self, msg):
        """
        Process incoming audio with optimized Whisper
        """
        if self.is_processing:
            self.get_logger().warn('Whisper is busy, dropping audio')
            return

        # Start processing in background thread
        processing_thread = Thread(
            target=self.process_audio_optimized,
            args=(msg,)
        )
        processing_thread.daemon = True
        processing_thread.start()

    def process_audio_optimized(self, audio_msg):
        """
        Optimized audio processing with Whisper
        """
        start_time = time.time()

        with self.processing_lock:
            if self.is_processing:
                return  # Another thread already started processing
            self.is_processing = True

        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_msg.data, dtype=np.int16)

            # Normalize and convert to float
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Limit audio duration to prevent long processing times
            max_samples = int(self.max_audio_duration * self.sample_rate)
            if len(audio_float) > max_samples:
                self.get_logger().info(f'Truncating audio from {len(audio_float)/self.sample_rate:.2f}s to {self.max_audio_duration}s')
                audio_float = audio_float[:max_samples]

            # Process with Whisper using optimized settings
            with torch.no_grad():  # Disable gradient computation for inference
                result = self.model.transcribe(
                    audio_float,
                    language='en',
                    task='transcribe',
                    temperature=0.0,  # Use greedy decoding for speed
                    best_of=1,        # Only use the best result
                    fp16=torch.cuda.is_available()
                )

            transcription = result['text'].strip()

            if transcription:
                self.get_logger().info(f'Transcribed: "{transcription}"')

                # Publish transcription
                trans_msg = String()
                trans_msg.data = transcription
                self.transcription_pub.publish(trans_msg)

        except Exception as e:
            self.get_logger().error(f'Error in optimized Whisper processing: {e}')

        finally:
            # Calculate and store processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Reset processing state
            with self.processing_lock:
                self.is_processing = False

    def report_performance(self):
        """
        Report performance metrics
        """
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            max_time = max(self.processing_times)
            min_time = min(self.processing_times)

            perf_msg = String()
            perf_msg.data = f'Whisper Performance - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s, Count: {len(self.processing_times)}'
            self.performance_pub.publish(perf_msg)

            self.get_logger().info(f'Performance: Avg {avg_time:.3f}s, Current load: {len(self.processing_times)}/100')

def main(args=None):
    rclpy.init(args=args)
    processor = EfficientWhisperProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Shutting down efficient Whisper processor')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files and Configuration

### 1. Complete Voice System Launch

Creating launch files for the complete voice system:

```python
# launch/voice_recognition_system.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    whisper_model = LaunchConfiguration('whisper_model')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    declare_whisper_model = DeclareLaunchArgument(
        'whisper_model',
        default_value='base',
        description='Whisper model size (tiny, base, small, medium, large)'
    )

    # Audio input handler
    audio_input_handler = Node(
        package='humanoid_voice_system',
        executable='audio_input_handler',
        name='audio_input_handler',
        parameters=[{
            'use_sim_time': use_sim_time,
            'sample_rate': 16000,
            'chunk_size': 1024,
            'channels': 1
        }],
        remappings=[
            ('/audio_input', '/audio_input'),
            ('/speech_detected', '/speech_detected'),
            ('/voice_activity', '/voice_activity')
        ]
    )

    # Whisper integration
    whisper_integration = Node(
        package='humanoid_voice_system',
        executable='whisper_integration',
        name='whisper_integration',
        parameters=[{
            'use_sim_time': use_sim_time,
            'whisper_model': whisper_model
        }],
        remappings=[
            ('/audio_input', '/audio_input'),
            ('/transcription', '/transcription'),
            ('/voice_command', '/voice_command'),
            ('/whisper_processing', '/whisper_processing')
        ]
    )

    # Voice command validator
    voice_validator = Node(
        package='humanoid_voice_system',
        executable='voice_command_validator',
        name='voice_command_validator',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        remappings=[
            ('/voice_command', '/voice_command'),
            ('/validated_command', '/validated_command'),
            ('/command_safety_status', '/command_safety_status'),
            ('/request_clarification', '/request_clarification')
        ]
    )

    # Voice processing pipeline
    voice_pipeline = Node(
        package='humanoid_voice_system',
        executable='voice_processing_pipeline',
        name='voice_processing_pipeline',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        remappings=[
            ('/audio_input', '/audio_input'),
            ('/transcription', '/transcription'),
            ('/validated_command', '/validated_command'),
            ('/robot_response', '/robot_response'),
            ('/voice_pipeline_status', '/voice_pipeline_status')
        ]
    )

    # Efficient Whisper processor (alternative to basic integration)
    efficient_whisper = Node(
        package='humanoid_voice_system',
        executable='efficient_whisper_processor',
        name='efficient_whisper_processor',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        remappings=[
            ('/audio_input', '/audio_input'),
            ('/transcription', '/transcription'),
            ('/whisper_performance', '/whisper_performance')
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_whisper_model,

        audio_input_handler,
        whisper_integration,
        voice_validator,
        voice_pipeline,
        efficient_whisper
    ])
```

## Next Steps

With the voice recognition and Whisper integration properly implemented, you're ready to move on to implementing cognitive planning using Large Language Models (LLMs) to convert natural language commands into robotic actions. The next section will cover connecting LLMs to ROS 2 systems for intelligent action planning and task decomposition.