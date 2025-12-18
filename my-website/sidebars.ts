import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Tutorial Intro',
    },
    {
      type: 'doc',
      id: 'physical-ai-humanoid-robotics',
      label: 'Physical AI & Humanoid Robotics Book',
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-robotic-nervous-system/intro',
        {
          type: 'category',
          label: 'ROS 2 Architecture',
          items: [
            'module-1-robotic-nervous-system/ros2-architecture/nodes-topics-services',
            'module-1-robotic-nervous-system/ros2-architecture/message-types',
            'module-1-robotic-nervous-system/ros2-architecture/lifecycle-nodes',
          ],
        },
        {
          type: 'category',
          label: 'Python Agents',
          items: [
            'module-1-robotic-nervous-system/python-agents/rclpy-basics',
            'module-1-robotic-nervous-system/python-agents/creating-nodes',
            'module-1-robotic-nervous-system/python-agents/publishers-subscribers',
          ],
        },
        {
          type: 'category',
          label: 'URDF Modeling',
          items: [
            'module-1-robotic-nervous-system/urdf-modeling/urdf-syntax',
            'module-1-robotic-nervous-system/urdf-modeling/humanoid-robot-structure',
            'module-1-robotic-nervous-system/urdf-modeling/visualizing-urdf',
          ],
        },
        {
          type: 'category',
          label: 'Practical Exercises',
          items: [
            'module-1-robotic-nervous-system/practical-exercises/exercise-1-simple-publisher',
            'module-1-robotic-nervous-system/practical-exercises/exercise-2-robot-control-service',
            'module-1-robotic-nervous-system/practical-exercises/exercise-3-urdf-robot',
          ],
        },
        'module-1-robotic-nervous-system/module-1-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/intro',
        {
          type: 'category',
          label: 'Gazebo Simulation',
          items: [
            'module-2-digital-twin/gazebo-simulation/environment-design',
            'module-2-digital-twin/gazebo-simulation/physics-modeling',
            'module-2-digital-twin/gazebo-simulation/gazebo-ros-integration',
            'module-2-digital-twin/gazebo-simulation/sensor-simulation',
          ],
        },
        {
          type: 'category',
          label: 'Unity Simulation',
          items: [
            'module-2-digital-twin/unity-simulation/unity-robotics-setup',
            'module-2-digital-twin/unity-simulation/environment-creation',
            'module-2-digital-twin/unity-simulation/unity-ros-communication',
            'module-2-digital-twin/unity-simulation/sensor-simulation',
          ],
        },
        'module-2-digital-twin/conclusion',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-ai-robot-brain/intro',
        {
          type: 'category',
          label: 'Isaac Sim Integration',
          items: [
            'module-3-ai-robot-brain/isaac-sim/isaac-sim-setup',
          ],
        },
        {
          type: 'category',
          label: 'AI Integration',
          items: [
            'module-3-ai-robot-brain/ai-integration/isaac-ros-perception',
          ],
        },
        {
          type: 'category',
          label: 'Navigation Systems',
          items: [
            'module-3-ai-robot-brain/navigation/nav2-bipedal-navigation',
          ],
        },
        {
          type: 'category',
          label: 'Practical Exercises',
          items: [
            'module-3-ai-robot-brain/practical-exercises/vslam-implementation',
            'module-3-ai-robot-brain/practical-exercises/photorealistic-simulation',
          ],
        },
        'module-3-ai-robot-brain/module-3-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vision-language-action/intro',
        {
          type: 'category',
          label: 'Voice Processing',
          items: [
            'module-4-vision-language-action/voice-processing/voice-recognition',
            'module-4-vision-language-action/voice-processing/voice-to-action-whisper',
          ],
        },
        {
          type: 'category',
          label: 'LLM Integration',
          items: [
            'module-4-vision-language-action/llm-integration/llm-integration',
            'module-4-vision-language-action/llm-integration/llm-cognitive-planning',
          ],
        },
        {
          type: 'category',
          label: 'Action Execution',
          items: [
            'module-4-vision-language-action/action-execution/manipulation-interaction',
            'module-4-vision-language-action/action-execution/vision-action-integration',
          ],
        },
        {
          type: 'category',
          label: 'Practical Exercises',
          items: [
            'module-4-vision-language-action/practical-exercises/multimodal-perception',
          ],
        },
        'module-4-vision-language-action/module-4-project',
      ],
    },
    {
      type: 'category',
      label: 'Specifications',
      items: [
        'specs/physical-ai-humanoid-textbook/checklists/requirements',
        'specs/physical-ai-humanoid-textbook/data-model',
        'specs/physical-ai-humanoid-textbook/plan',
        'specs/physical-ai-humanoid-textbook/research',
        'specs/physical-ai-humanoid-textbook/tasks',
      ],
    },
  ],
};

export default sidebars;