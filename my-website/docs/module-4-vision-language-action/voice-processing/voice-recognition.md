---
sidebar_position: 2
---

# Voice Recognition and Processing for Humanoid Robots

## Overview

Voice recognition forms the foundation of natural human-robot interaction in VLA systems. This section covers implementing speech-to-text systems optimized for humanoid robot environments, including real-time processing, noise reduction, and context-aware recognition that enables robots to understand and respond to natural language commands.

Humanoid robots operate in diverse acoustic environments, requiring robust voice recognition systems that can handle background noise, reverberation, and multiple speakers while maintaining real-time performance for natural interaction.

## Audio Input and Preprocessing

### 1. Audio Capture Configuration

Setting up audio input for humanoid robots with optimal configuration:

```python
#!/usr/bin/env python3
# audio_capture.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
import pyaudio
import numpy as np
import threading
import queue
import time

class HumanoidAudioCapture(Node):
    """
    Audio capture system optimized for humanoid robot environments
    """
    def __init__(self):
        super().__init__('humanoid_audio_capture')

        # Audio parameters
        self.rate = 16000  # 16kHz sample rate (good for speech)
        self.chunk = 1024  # Number of frames per buffer
        self.channels = 1  # Mono audio
        self.format = pyaudio.paInt16  # 16-bit audio
        self.audio_queue = queue.Queue(maxsize=10)

        # Initialize PyAudio
        self.pyaudio_instance = pyaudio.PyAudio()

        # Publishers
        self.audio_pub = self.create_publisher(
            AudioData,
            '/audio/raw',
            10
        )

        self.speech_detected_pub = self.create_publisher(
            Bool,
            '/speech_detected',
            10
        )

        # Audio processing thread
        self.audio_thread = threading.Thread(target=self.audio_capture_loop)
        self.running = True

        # Voice activity detection parameters
        self.energy_threshold = 1000  # Adjust based on environment
        self.silence_threshold = 0.5  # 50% of energy threshold
        self.silence_duration = 1.0   # 1 second of silence to stop

        # Start audio capture
        self.audio_thread.start()

        self.get_logger().info('Humanoid Audio Capture initialized')

    def audio_capture_loop(self):
        """
        Continuous audio capture loop
        """
        # Open audio stream
        stream = self.pyaudio_instance.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        self.get_logger().info('Audio capture started')

        while self.running:
            try:
                # Read audio data
                data = stream.read(self.chunk, exception_on_overflow=False)

                # Convert to numpy array for processing
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # Check for speech activity
                energy = np.mean(np.abs(audio_array))
                speech_detected = energy > self.energy_threshold

                # Publish speech detection
                speech_msg = Bool()
                speech_msg.data = speech_detected
                self.speech_detected_pub.publish(speech_msg)

                # Publish audio data if queue not full
                if speech_detected and not self.audio_queue.full():
                    self.audio_queue.put(data)

                # Publish raw audio data periodically
                if not self.audio_queue.empty():
                    try:
                        audio_data = self.audio_queue.get_nowait()
                        audio_msg = AudioData()
                        audio_msg.data = audio_data
                        self.audio_pub.publish(audio_msg)
                    except queue.Empty:
                        pass

            except Exception as e:
                self.get_logger().error(f'Audio capture error: {e}')
                time.sleep(0.1)  # Brief pause before retry

        stream.stop_stream()
        stream.close()

    def destroy_node(self):
        """
        Clean shutdown of audio capture
        """
        self.running = False
        if self.audio_thread.is_alive():
            self.audio_thread.join()

        # Close PyAudio
        self.pyaudio_instance.terminate()

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    audio_capture = HumanoidAudioCapture()

    try:
        rclpy.spin(audio_capture)
    except KeyboardInterrupt:
        audio_capture.get_logger().info('Shutting down audio capture')
    finally:
        audio_capture.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Noise Reduction and Audio Enhancement

Implementing noise reduction for humanoid environments:

```python
#!/usr/bin/env python3
# noise_reduction.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import AudioData
from std_msgs.msg import String
import numpy as np
from scipy import signal
import webrtcvad
import collections

class AudioNoiseReducer(Node):
    """
    Noise reduction system for humanoid audio processing
    """
    def __init__(self):
        super().__init__('audio_noise_reducer')

        # Subscribe to raw audio
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio/raw',
            self.audio_callback,
            10
        )

        # Publish enhanced audio
        self.enhanced_audio_pub = self.create_publisher(
            AudioData,
            '/audio/enhanced',
            10
        )

        self.enhanced_text_pub = self.create_publisher(
            String,
            '/audio/enhanced_status',
            10
        )

        # Audio parameters
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

        # Initialize WebRTC VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Aggressive mode for noisy environments

        # Noise reduction parameters
        self.noise_floor = 0.01  # Minimum signal level
        self.snr_threshold = 10  # Signal-to-noise ratio threshold

        # Audio buffer for noise estimation
        self.noise_buffer = collections.deque(maxlen=100)
        self.signal_buffer = collections.deque(maxlen=10)

        self.get_logger().info('Audio Noise Reducer initialized')

    def audio_callback(self, msg):
        """
        Process incoming audio data
        """
        try:
            # Convert byte data to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize to [-1, 1]

            # Apply noise reduction
            enhanced_audio = self.reduce_noise(audio_data)

            # Detect voice activity
            is_speech = self.detect_voice_activity(enhanced_audio)

            if is_speech:
                # Convert back to int16 for publishing
                enhanced_int16 = (enhanced_audio * 32768).astype(np.int16)
                enhanced_bytes = enhanced_int16.tobytes()

                # Publish enhanced audio
                enhanced_msg = AudioData()
                enhanced_msg.data = enhanced_bytes
                self.enhanced_audio_pub.publish(enhanced_msg)

                # Publish status
                status_msg = String()
                status_msg.data = f"Enhanced audio with SNR: {self.estimate_snr(enhanced_audio):.2f}dB"
                self.enhanced_text_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error in audio callback: {e}')

    def reduce_noise(self, audio_data):
        """
        Apply noise reduction to audio data
        """
        # Simple spectral subtraction noise reduction
        # More sophisticated methods can be implemented

        # Convert to frequency domain
        fft_data = np.fft.fft(audio_data)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)

        # Estimate noise floor
        if len(self.noise_buffer) < 50:
            # Collect initial noise samples
            self.noise_buffer.extend(magnitude)
        else:
            # Update noise estimate
            current_noise = np.percentile(magnitude, 10)  # 10th percentile as noise estimate
            self.noise_buffer.append(current_noise)

        # Calculate average noise level
        if self.noise_buffer:
            avg_noise = np.mean(list(self.noise_buffer))
        else:
            avg_noise = 0.0

        # Apply spectral subtraction
        enhanced_magnitude = np.maximum(magnitude - avg_noise * 0.7, 0.1 * magnitude)

        # Reconstruct signal
        enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = np.real(np.fft.ifft(enhanced_fft))

        # Normalize to prevent clipping
        max_val = np.max(np.abs(enhanced_audio))
        if max_val > 0:
            enhanced_audio = enhanced_audio / max_val * 0.8  # 80% of max to prevent clipping

        return enhanced_audio

    def detect_voice_activity(self, audio_data):
        """
        Detect voice activity using WebRTC VAD
        """
        # WebRTC VAD works with specific frame sizes and rates
        # We'll do a simplified version here
        if len(audio_data) < self.frame_size:
            return False

        # Calculate energy-based voice activity detection
        energy = np.mean(np.abs(audio_data))
        threshold = self.estimate_energy_threshold()

        return energy > threshold

    def estimate_energy_threshold(self):
        """
        Estimate adaptive energy threshold for VAD
        """
        if len(self.signal_buffer) > 0:
            recent_energies = list(self.signal_buffer)
            # Use median to be robust to outliers
            median_energy = np.median(recent_energies)
            return median_energy * 0.5  # 50% of median as threshold
        else:
            return 0.01  # Default threshold

    def estimate_snr(self, audio_data):
        """
        Estimate signal-to-noise ratio
        """
        signal_power = np.mean(audio_data ** 2)
        if len(self.noise_buffer) > 0:
            noise_power = np.mean(list(self.noise_buffer)) ** 2
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                return snr
        return 0.0

def main(args=None):
    rclpy.init(args=args)
    noise_reducer = AudioNoiseReducer()

    try:
        rclpy.spin(noise_reducer)
    except KeyboardInterrupt:
        noise_reducer.get_logger().info('Shutting down noise reducer')
    finally:
        noise_reducer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## OpenAI Whisper Integration

### 1. Whisper Setup and Configuration

Setting up OpenAI Whisper for humanoid robot voice recognition:

```python
#!/usr/bin/env python3
# whisper_integration.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import AudioData
from std_msgs.msg import String
from rclpy.qos import QoSProfile
import torch
import whisper
import numpy as np
import io
import wave
from threading import Thread, Lock
from queue import Queue
import time

class WhisperSpeechRecognizer(Node):
    """
    Speech recognition using OpenAI Whisper
    """
    def __init__(self):
        super().__init__('whisper_speech_recognizer')

        # Subscribe to enhanced audio
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio/enhanced',
            self.audio_callback,
            QoSProfile(depth=10)
        )

        # Publishers
        self.transcript_pub = self.create_publisher(
            String,
            '/speech_to_text/transcript',
            10
        )

        self.confidence_pub = self.create_publisher(
            String,
            '/speech_to_text/confidence',
            10
        )

        # Audio processing parameters
        self.sample_rate = 16000
        self.chunk_duration = 2.0  # Process 2-second chunks
        self.min_silence_duration = 0.5  # Minimum speech duration to process

        # Audio buffer for accumulating audio chunks
        self.audio_buffer = []
        self.buffer_duration = 0.0

        # Whisper model
        self.model = None
        self.load_whisper_model()

        # Processing thread and queue
        self.audio_queue = Queue(maxsize=5)
        self.processing_thread = Thread(target=self.process_audio_queue)
        self.processing_thread.start()

        # Processing lock
        self.processing_lock = Lock()

        self.get_logger().info('Whisper Speech Recognizer initialized')

    def load_whisper_model(self):
        """
        Load Whisper model (use smaller model for real-time performance)
        """
        try:
            # Use 'tiny' or 'base' model for real-time performance
            # Use 'small' or 'medium' for better accuracy with more compute
            self.model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
            self.get_logger().info(f'Whisper model loaded on {"GPU" if torch.cuda.is_available() else "CPU"}')
        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {e}')
            # Fallback: try loading on CPU
            try:
                self.model = whisper.load_model("base", device="cpu")
                self.get_logger().info('Whisper model loaded on CPU')
            except Exception as e2:
                self.get_logger().error(f'Failed to load Whisper model on CPU: {e2}')

    def audio_callback(self, msg):
        """
        Process incoming audio data
        """
        if self.model is None:
            return

        try:
            # Convert byte data to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize

            # Add to buffer
            self.audio_buffer.extend(audio_data)
            self.buffer_duration = len(self.audio_buffer) / self.sample_rate

            # Process buffer if it reaches the desired duration
            if self.buffer_duration >= self.chunk_duration:
                # Extract chunk and clear buffer
                chunk_size = int(self.chunk_duration * self.sample_rate)
                audio_chunk = np.array(self.audio_buffer[:chunk_size])
                self.audio_buffer = self.audio_buffer[chunk_size:]
                self.buffer_duration = len(self.audio_buffer) / self.sample_rate

                # Add to processing queue if not full
                if not self.audio_queue.full():
                    self.audio_queue.put(audio_chunk)

        except Exception as e:
            self.get_logger().error(f'Error in audio callback: {e}')

    def process_audio_queue(self):
        """
        Process audio chunks from the queue in a separate thread
        """
        while rclpy.ok():
            try:
                # Get audio chunk from queue
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=0.1)

                    # Process with Whisper
                    transcript = self.transcribe_audio(audio_chunk)

                    if transcript and transcript.strip():
                        # Publish transcript
                        transcript_msg = String()
                        transcript_msg.data = transcript
                        self.transcript_pub.publish(transcript_msg)

                        # Publish confidence (simulated - Whisper doesn't provide confidence directly)
                        confidence_msg = String()
                        confidence_msg.data = f"high"  # Could be calculated based on various factors
                        self.confidence_pub.publish(confidence_msg)

                        self.get_logger().info(f'Transcribed: {transcript}')

                else:
                    # Small sleep to prevent busy waiting
                    time.sleep(0.01)

            except Exception as e:
                self.get_logger().error(f'Error in processing thread: {e}')
                time.sleep(0.1)

    def transcribe_audio(self, audio_chunk):
        """
        Transcribe audio chunk using Whisper
        """
        if self.model is None:
            return None

        try:
            with self.processing_lock:
                # Transcribe the audio
                result = self.model.transcribe(
                    audio_chunk,
                    language='en',
                    task='transcribe',
                    # Add options to improve real-time performance
                    fp16=torch.cuda.is_available()
                )

                return result['text'].strip()
        except Exception as e:
            self.get_logger().error(f'Error transcribing audio: {e}')
            return None

    def destroy_node(self):
        """
        Clean shutdown
        """
        if self.processing_thread.is_alive():
            # Let the thread finish naturally by rclpy.ok() returning False
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    whisper_node = WhisperSpeechRecognizer()

    try:
        rclpy.spin(whisper_node)
    except KeyboardInterrupt:
        whisper_node.get_logger().info('Shutting down Whisper speech recognizer')
    finally:
        whisper_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Custom Voice Command Recognition

Creating a custom voice command recognition system optimized for robotics:

```python
#!/usr/bin/env python3
# custom_voice_commands.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration
import json
import re
from typing import Dict, List, Optional

class CustomVoiceCommandRecognizer(Node):
    """
    Custom voice command recognition for humanoid robots
    """
    def __init__(self):
        super().__init__('custom_voice_command_recognizer')

        # Subscribe to Whisper transcripts
        self.transcript_sub = self.create_subscription(
            String,
            '/speech_to_text/transcript',
            self.transcript_callback,
            10
        )

        # Publishers for recognized commands
        self.command_pub = self.create_publisher(
            String,
            '/voice_command/parsed',
            10
        )

        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        # Voice command patterns and handlers
        self.command_patterns = {
            # Navigation commands
            'move_to': [
                r'go to (?:the )?(?P<location>\w+)',
                r'move to (?:the )?(?P<location>\w+)',
                r'go (?:to|toward) (?:the )?(?P<location>\w+)',
                r'walk to (?:the )?(?P<location>\w+)',
            ],
            'come_here': [
                r'come here',
                r'come to me',
                r'come over here',
            ],
            'stop': [
                r'stop',
                r'hold on',
                r'wait',
                r'freeze',
            ],
            'turn': [
                r'turn (?:left|right|around)',
                r'rotate (?:left|right)',
                r'face (?:left|right)',
            ],

            # Manipulation commands
            'pick_up': [
                r'pick up (?:the )?(?P<object>\w+)',
                r'grab (?:the )?(?P<object>\w+)',
                r'take (?:the )?(?P<object>\w+)',
                r'lift (?:the )?(?P<object>\w+)',
            ],
            'put_down': [
                r'put down (?:the )?(?P<object>\w+)',
                r'drop (?:the )?(?P<object>\w+)',
                r'place (?:the )?(?P<object>\w+)',
            ],

            # Interaction commands
            'follow_me': [
                r'follow me',
                r'come with me',
                r'follow',
            ],
            'wait_here': [
                r'wait here',
                r'stay here',
                r'hold position',
            ],
        }

        # Location mappings (in a real system, these would come from a map)
        self.location_map = {
            'kitchen': {'x': 3.0, 'y': 2.0, 'theta': 0.0},
            'living_room': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'bedroom': {'x': -2.0, 'y': 3.0, 'theta': 1.57},
            'bathroom': {'x': -1.0, 'y': -2.0, 'theta': 3.14},
            'office': {'x': 2.0, 'y': -1.0, 'theta': -1.57},
        }

        # Object locations (simplified - in real system would come from perception)
        self.object_locations = {
            'cup': {'x': 3.2, 'y': 2.1, 'z': 0.8},
            'book': {'x': 0.1, 'y': 0.1, 'z': 0.9},
            'ball': {'x': -0.5, 'y': 0.5, 'z': 0.1},
        }

        self.get_logger().info('Custom Voice Command Recognizer initialized')

    def transcript_callback(self, msg):
        """
        Process incoming transcript and extract commands
        """
        transcript = msg.data.lower().strip()
        if not transcript:
            return

        self.get_logger().info(f'Received transcript: {transcript}')

        # Parse the transcript for commands
        parsed_commands = self.parse_transcript(transcript)

        if parsed_commands:
            for command in parsed_commands:
                self.get_logger().info(f'Parsed command: {command}')

                # Publish command
                cmd_msg = String()
                cmd_msg.data = json.dumps(command)
                self.command_pub.publish(cmd_msg)

                # Execute command if possible
                self.execute_command(command)

    def parse_transcript(self, transcript: str) -> List[Dict]:
        """
        Parse transcript to extract commands using regex patterns
        """
        commands = []

        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, transcript, re.IGNORECASE)
                if match:
                    command = {
                        'type': command_type,
                        'original_text': transcript,
                        'matched_pattern': pattern,
                        'confidence': 0.9,  # High confidence for pattern matches
                        **match.groupdict()  # Add captured groups as parameters
                    }
                    commands.append(command)

        return commands

    def execute_command(self, command: Dict):
        """
        Execute the parsed command
        """
        cmd_type = command['type']

        if cmd_type == 'move_to':
            self.execute_move_to(command)
        elif cmd_type == 'come_here':
            self.execute_come_here(command)
        elif cmd_type == 'stop':
            self.execute_stop(command)
        elif cmd_type == 'turn':
            self.execute_turn(command)
        elif cmd_type == 'pick_up':
            self.execute_pick_up(command)
        elif cmd_type == 'put_down':
            self.execute_put_down(command)
        elif cmd_type == 'follow_me':
            self.execute_follow_me(command)
        elif cmd_type == 'wait_here':
            self.execute_wait_here(command)
        else:
            self.get_logger().warn(f'Unknown command type: {cmd_type}')

    def execute_move_to(self, command: Dict):
        """
        Execute move to location command
        """
        location = command.get('location', '').lower()

        if location in self.location_map:
            location_data = self.location_map[location]

            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            goal.pose.position.x = location_data['x']
            goal.pose.position.y = location_data['y']
            goal.pose.position.z = 0.0  # Ground level

            # Simple orientation based on theta
            import math
            goal.pose.orientation.z = math.sin(location_data['theta'] / 2)
            goal.pose.orientation.w = math.cos(location_data['theta'] / 2)

            self.navigation_goal_pub.publish(goal)
            self.get_logger().info(f'Moving to {location} at ({location_data["x"]}, {location_data["y"]})')
        else:
            self.get_logger().warn(f'Unknown location: {location}')
            # Could implement location learning or ask for clarification

    def execute_come_here(self, command: Dict):
        """
        Execute come here command (would need to locate user)
        """
        # In a real system, this would use person detection/localization
        # For now, we'll just acknowledge the command
        self.get_logger().info('Received "come here" command - would navigate to user location')

    def execute_stop(self, command: Dict):
        """
        Execute stop command
        """
        # Publish stop command to navigation system
        stop_msg = String()
        stop_msg.data = 'stop'
        self.command_pub.publish(stop_msg)
        self.get_logger().info('Stopping robot')

    def execute_turn(self, command: Dict):
        """
        Execute turn command
        """
        # Extract turn direction from original text
        text = command['original_text']
        if 'left' in text:
            direction = 'left'
        elif 'right' in text:
            direction = 'right'
        elif 'around' in text:
            direction = 'around'
        else:
            direction = 'unknown'

        self.get_logger().info(f'Turning {direction}')
        # Would publish turn command to navigation system

    def execute_pick_up(self, command: Dict):
        """
        Execute pick up object command
        """
        obj = command.get('object', '').lower()

        if obj in self.object_locations:
            obj_pos = self.object_locations[obj]
            self.get_logger().info(f'Picking up {obj} at ({obj_pos["x"]}, {obj_pos["y"]}, {obj_pos["z"]})')
            # Would publish pick-up command to manipulation system
        else:
            self.get_logger().warn(f'Object not found: {obj}')
            # Could use perception to locate the object

    def execute_put_down(self, command: Dict):
        """
        Execute put down object command
        """
        obj = command.get('object', '').lower()
        self.get_logger().info(f'Putting down {obj}')
        # Would publish put-down command to manipulation system

    def execute_follow_me(self, command: Dict):
        """
        Execute follow me command
        """
        self.get_logger().info('Following user - would activate follow mode')
        # Would activate person following behavior

    def execute_wait_here(self, command: Dict):
        """
        Execute wait here command
        """
        self.get_logger().info('Waiting at current location')
        # Would stop navigation and wait

def main(args=None):
    rclpy.init(args=args)
    command_recognizer = CustomVoiceCommandRecognizer()

    try:
        rclpy.spin(command_recognizer)
    except KeyboardInterrupt:
        command_recognizer.get_logger().info('Shutting down voice command recognizer')
    finally:
        command_recognizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Real-time Processing Optimization

### 1. Multi-threaded Audio Processing

Implementing efficient multi-threaded audio processing for real-time performance:

```python
#!/usr/bin/env python3
# multithreaded_audio_processing.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import AudioData
from std_msgs.msg import String
from threading import Thread, Lock, Event
from queue import Queue, Empty
import time
import numpy as np

class MultiThreadedAudioProcessor(Node):
    """
    Multi-threaded audio processing for real-time VLA systems
    """
    def __init__(self):
        super().__init__('multithreaded_audio_processor')

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio/raw',
            self.audio_callback,
            10
        )

        # Publishers
        self.transcript_pub = self.create_publisher(
            String,
            '/vad/transcript',
            10
        )

        # Audio processing queues
        self.raw_audio_queue = Queue(maxsize=10)
        self.processed_audio_queue = Queue(maxsize=5)

        # Processing threads
        self.preprocessing_thread = Thread(target=self.preprocessing_loop)
        self.speech_recognition_thread = Thread(target=self.speech_recognition_loop)
        self.postprocessing_thread = Thread(target=self.postprocessing_loop)

        # Control events
        self.shutdown_event = Event()

        # Processing locks
        self.audio_lock = Lock()

        # Start threads
        self.preprocessing_thread.start()
        self.speech_recognition_thread.start()
        self.postprocessing_thread.start()

        self.get_logger().info('Multi-threaded Audio Processor initialized')

    def audio_callback(self, msg):
        """
        Receive raw audio data
        """
        try:
            if not self.raw_audio_queue.full():
                self.raw_audio_queue.put(msg, timeout=0.01)
        except:
            # Drop frame if queue is full
            pass

    def preprocessing_loop(self):
        """
        Preprocessing thread: noise reduction, VAD, etc.
        """
        while not self.shutdown_event.is_set():
            try:
                # Get raw audio
                raw_msg = self.raw_audio_queue.get(timeout=0.1)

                # Convert to numpy
                audio_data = np.frombuffer(raw_msg.data, dtype=np.int16).astype(np.float32) / 32768.0

                # Apply preprocessing (noise reduction, normalization, etc.)
                processed_audio = self.apply_preprocessing(audio_data)

                # Check if audio contains speech
                if self.is_speech(processed_audio):
                    # Add to processing queue
                    processed_item = {
                        'data': processed_audio,
                        'timestamp': raw_msg.header.stamp if hasattr(raw_msg, 'header') else time.time()
                    }

                    if not self.processed_audio_queue.full():
                        self.processed_audio_queue.put(processed_item)
                    else:
                        self.get_logger().warn('Processed audio queue full, dropping frame')

            except Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Preprocessing error: {e}')

    def speech_recognition_loop(self):
        """
        Speech recognition thread: ASR processing
        """
        # Initialize ASR model (simplified - would use Whisper or similar)
        asr_model = self.initialize_asr_model()

        while not self.shutdown_event.is_set():
            try:
                # Get processed audio
                processed_item = self.processed_audio_queue.get(timeout=0.1)

                # Perform speech recognition
                if asr_model:
                    transcript = self.perform_asr(
                        asr_model,
                        processed_item['data']
                    )

                    if transcript:
                        # Publish transcript
                        transcript_msg = String()
                        transcript_msg.data = transcript
                        self.transcript_pub.publish(transcript_msg)

            except Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Speech recognition error: {e}')

    def postprocessing_loop(self):
        """
        Postprocessing thread: transcript validation, NLU, etc.
        """
        while not self.shutdown_event.is_set():
            # In a real system, this would handle transcript validation,
            # natural language understanding, and command generation
            time.sleep(0.1)  # Placeholder

    def apply_preprocessing(self, audio_data):
        """
        Apply audio preprocessing (noise reduction, normalization, etc.)
        """
        # Normalize audio
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.8  # 80% normalization to prevent clipping

        # Apply simple noise reduction
        # (In practice, use more sophisticated methods)
        noise_floor = np.std(audio_data) * 0.1
        audio_data = np.clip(audio_data, -noise_floor, noise_floor) + audio_data

        return audio_data

    def is_speech(self, audio_data):
        """
        Simple voice activity detection
        """
        # Calculate energy
        energy = np.mean(np.abs(audio_data))

        # Define threshold (would be adaptive in real system)
        threshold = 0.01

        return energy > threshold

    def initialize_asr_model(self):
        """
        Initialize ASR model (placeholder)
        """
        # In real implementation, load Whisper or similar model
        return True  # Placeholder

    def perform_asr(self, model, audio_data):
        """
        Perform ASR on audio data (placeholder)
        """
        # In real implementation, use actual ASR model
        # This is a simplified simulation
        if len(audio_data) > 1000:  # Only process substantial audio
            # Simulate some processing time
            time.sleep(0.05)
            # Return placeholder transcript
            return "simulated transcript from audio"
        return None

    def destroy_node(self):
        """
        Clean shutdown of processing threads
        """
        self.shutdown_event.set()

        if self.preprocessing_thread.is_alive():
            self.preprocessing_thread.join()
        if self.speech_recognition_thread.is_alive():
            self.speech_recognition_thread.join()
        if self.postprocessing_thread.is_alive():
            self.postprocessing_thread.join()

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    processor = MultiThreadedAudioProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Shutting down multi-threaded audio processor')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Voice Command Validation and Safety

### 1. Safe Voice Command Processing

Implementing safety checks for voice commands:

```python
#!/usr/bin/env python3
# voice_command_safety.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from typing import Dict, Any
import json
import re

class VoiceCommandSafetyValidator(Node):
    """
    Safety validation for voice commands in humanoid robots
    """
    def __init__(self):
        super().__init__('voice_command_safety_validator')

        # Subscribe to raw voice commands
        self.command_sub = self.create_subscription(
            String,
            '/voice_command/parsed',
            self.command_callback,
            10
        )

        # Publishers for validated commands
        self.validated_command_pub = self.create_publisher(
            String,
            '/voice_command/validated',
            10
        )

        self.safety_alert_pub = self.create_publisher(
            String,
            '/safety/alert',
            10
        )

        # Service for emergency stop
        self.emergency_stop_service = self.create_service(
            Trigger,
            '/voice_command/emergency_stop',
            self.emergency_stop_callback
        )

        # Safety configuration
        self.safety_config = {
            'max_navigation_distance': 10.0,  # meters
            'forbidden_locations': ['restricted_area', 'danger_zone'],
            'forbidden_actions': ['jump', 'run_fast', 'harm'],
            'safe_speed_limits': {
                'linear': 0.5,  # m/s
                'angular': 0.5  # rad/s
            },
            'timeout_duration': 30.0  # seconds
        }

        # User authorization (simplified - would use proper auth system)
        self.authorized_users = ['default_user']
        self.current_user = 'default_user'

        # Command history for context
        self.command_history = []

        self.get_logger().info('Voice Command Safety Validator initialized')

    def command_callback(self, msg):
        """
        Process incoming voice command with safety validation
        """
        try:
            command_data = json.loads(msg.data)
            self.get_logger().info(f'Received command: {command_data}')

            # Validate command
            is_safe, reason = self.validate_command(command_data)

            if is_safe:
                # Add to history
                self.command_history.append({
                    'command': command_data,
                    'timestamp': self.get_clock().now().nanoseconds / 1e9
                })

                # Publish validated command
                validated_msg = String()
                validated_msg.data = msg.data
                self.validated_command_pub.publish(validated_msg)

                self.get_logger().info(f'Command validated and forwarded: {command_data["type"]}')
            else:
                # Publish safety alert
                alert_msg = String()
                alert_msg.data = f'Safety violation: {reason}'
                self.safety_alert_pub.publish(alert_msg)

                self.get_logger().warn(f'Safety violation: {reason} for command {command_data}')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in command message')
        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')

    def validate_command(self, command: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate command for safety
        """
        cmd_type = command.get('type', '').lower()

        # Check user authorization
        if not self.is_user_authorized():
            return False, "User not authorized"

        # Validate based on command type
        if cmd_type in ['move_to', 'navigate']:
            return self.validate_navigation_command(command)
        elif cmd_type in ['pick_up', 'grasp', 'manipulate']:
            return self.validate_manipulation_command(command)
        elif cmd_type in ['turn', 'rotate']:
            return self.validate_rotation_command(command)
        elif cmd_type in ['stop', 'wait_here']:
            return True, "Command is safe"  # Stop commands are generally safe
        else:
            return self.validate_generic_command(command)

    def validate_navigation_command(self, command: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate navigation-related commands
        """
        # Check if destination is forbidden
        location = command.get('location', '').lower()
        if location in self.safety_config['forbidden_locations']:
            return False, f"Destination '{location}' is forbidden"

        # Check distance constraints (would need robot position for real validation)
        # This is a simplified check - real system would calculate actual distance
        if 'x' in command and 'y' in command:
            distance = (command['x']**2 + command['y']**2)**0.5
            if distance > self.safety_config['max_navigation_distance']:
                return False, f"Destination too far: {distance:.2f}m > {self.safety_config['max_navigation_distance']}m"

        return True, "Navigation command is safe"

    def validate_manipulation_command(self, command: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate manipulation-related commands
        """
        # Check if object is forbidden
        obj = command.get('object', '').lower()
        if obj in self.safety_config['forbidden_actions']:
            return False, f"Object '{obj}' manipulation is restricted"

        # Check if action is potentially dangerous
        action = command.get('type', '').lower()
        if any(dangerous in action for dangerous in ['harm', 'damage', 'break']):
            return False, "Manipulation command may cause harm"

        return True, "Manipulation command is safe"

    def validate_rotation_command(self, command: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate rotation-related commands
        """
        # Check for excessive rotation
        text = command.get('original_text', '').lower()

        if 'spin' in text or 'fast' in text:
            return False, "Rotation command too aggressive"

        return True, "Rotation command is safe"

    def validate_generic_command(self, command: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate generic commands
        """
        # Check for forbidden words in original text
        original_text = command.get('original_text', '').lower()

        for forbidden in self.safety_config['forbidden_actions']:
            if forbidden in original_text:
                return False, f"Command contains forbidden action: {forbidden}"

        return True, "Command is safe"

    def is_user_authorized(self) -> bool:
        """
        Check if current user is authorized
        """
        # In a real system, this would use proper authentication
        # For now, we'll assume the default user is authorized
        return self.current_user in self.authorized_users

    def emergency_stop_callback(self, request, response):
        """
        Handle emergency stop request
        """
        self.get_logger().warn('EMERGENCY STOP ACTIVATED')

        # Publish safety alert
        alert_msg = String()
        alert_msg.data = 'EMERGENCY STOP - All motion stopped'
        self.safety_alert_pub.publish(alert_msg)

        # In a real system, this would stop all robot motion
        response.success = True
        response.message = 'Emergency stop executed'
        return response

    def check_command_history_safety(self) -> tuple[bool, str]:
        """
        Check command history for safety violations
        """
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Check for command flooding (too many commands in short time)
        recent_commands = [
            cmd for cmd in self.command_history
            if current_time - cmd['timestamp'] < 2.0  # Last 2 seconds
        ]

        if len(recent_commands) > 5:  # More than 5 commands in 2 seconds
            return False, "Command flooding detected"

        return True, "Command history is safe"

def main(args=None):
    rclpy.init(args=args)
    safety_validator = VoiceCommandSafetyValidator()

    try:
        rclpy.spin(safety_validator)
    except KeyboardInterrupt:
        safety_validator.get_logger().info('Shutting down safety validator')
    finally:
        safety_validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files for Voice Recognition System

### 1. Complete Voice Recognition Launch

Creating launch files for the complete voice recognition system:

```python
# launch/voice_recognition_system.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot'
    )

    # Audio capture node
    audio_capture = Node(
        package='humanoid_vla',
        executable='audio_capture',
        name='audio_capture',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Audio noise reducer
    noise_reducer = Node(
        package='humanoid_vla',
        executable='audio_noise_reducer',
        name='audio_noise_reducer',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Whisper speech recognizer
    whisper_recognizer = Node(
        package='humanoid_vla',
        executable='whisper_speech_recognizer',
        name='whisper_speech_recognizer',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Custom voice command recognizer
    voice_command_recognizer = Node(
        package='humanoid_vla',
        executable='custom_voice_command_recognizer',
        name='voice_command_recognizer',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Voice command safety validator
    safety_validator = Node(
        package='humanoid_vla',
        executable='voice_command_safety_validator',
        name='voice_command_safety_validator',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Multi-threaded audio processor
    multithreaded_processor = Node(
        package='humanoid_vla',
        executable='multithreaded_audio_processor',
        name='multithreaded_audio_processor',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Group all audio processing nodes
    audio_processing_group = GroupAction(
        actions=[
            SetParameter('use_sim_time', use_sim_time),
            audio_capture,
            noise_reducer,
            whisper_recognizer,
            voice_command_recognizer,
            safety_validator,
            multithreaded_processor
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_name,

        audio_processing_group,
    ])
```

## Performance Optimization and Real-time Considerations

### 1. Real-time Audio Processing Constraints

Optimizing for real-time performance in voice recognition:

```python
# real_time_constraints.md

## Real-time Audio Processing Constraints for Humanoid Robots

### 1. Processing Latency Requirements

#### Audio Pipeline Latency Budget
- **Audio Capture**: <5ms
- **Preprocessing**: <10ms
- **Voice Activity Detection**: <5ms
- **Speech Recognition**: <100ms (for real-time interaction)
- **Command Processing**: <20ms
- **Total Pipeline**: <140ms (aim for <100ms)

#### Critical Path Optimization
- Use lock-free queues between processing stages
- Implement audio ring buffers for continuous processing
- Optimize FFT sizes for real-time performance (512-1024 samples)
- Use fixed-point arithmetic where possible for embedded systems

### 2. Computational Resource Management

#### CPU Utilization
- **Target**: <70% CPU usage for audio processing
- **Strategy**: Use multi-threading for parallel processing
- **Monitoring**: Implement CPU usage feedback control

#### Memory Management
- **Audio Buffers**: Pre-allocate all audio processing buffers
- **Real-time Allocation**: Avoid dynamic allocation during processing
- **Memory Pool**: Implement memory pools for audio frames

#### GPU Acceleration
- **Whisper Model**: Use GPU for speech recognition when available
- **Batch Processing**: Process multiple audio chunks in parallel
- **Precision**: Use FP16 for reduced memory usage and faster processing

### 3. Power Consumption Optimization

#### Adaptive Processing
- **Low Power Mode**: Reduce processing rate when no speech detected
- **Dynamic Frequency Scaling**: Adjust CPU/GPU frequency based on workload
- **Sleep States**: Enter low-power states during silence periods

### 4. Robustness and Error Handling

#### Failure Modes
- **Microphone Failure**: Fallback to alternative audio sources
- **Network Disconnection**: Local processing capabilities
- **Model Corruption**: Model integrity checking and reloading

#### Error Recovery
- **Watchdog Timers**: Monitor processing pipeline health
- **Graceful Degradation**: Fallback to simpler processing when resources limited
- **State Recovery**: Preserve command context during failures
```

## Next Steps

With the voice recognition and processing system fully implemented, you're ready to move on to integrating large language models (LLMs) for cognitive planning. The next section will cover connecting LLMs to your robotic system, implementing prompt engineering for robotics tasks, and creating cognitive planning pipelines that convert natural language commands into executable robotic actions.

The voice recognition system you've built provides the foundation for natural human-robot interaction that will be essential when implementing the higher-level cognitive functions in the following sections.