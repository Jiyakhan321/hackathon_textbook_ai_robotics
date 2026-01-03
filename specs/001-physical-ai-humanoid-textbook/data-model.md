# Data Model: Physical AI & Humanoid Robotics Interactive 3D Textbook

## Module Entity
- **Name**: String (required) - Module name (e.g., "Robotic Nervous System", "Digital Twin")
- **Description**: String (required) - Brief description of the module content
- **Type**: Enum ["standard", "interactive-3d"] (required) - Whether the module uses standard Markdown or includes 3D components
- **LearningObjectives**: Array[String] (required) - List of learning objectives for the module
- **Topics**: Array[Topic] (required) - Topics covered in the module
- **Prerequisites**: Array[String] (optional) - Prerequisites for this module
- **Exercises**: Array[Exercise] (optional) - Exercises for the module

## Topic Entity
- **Title**: String (required) - Topic title
- **Content**: String (required) - Topic content in Markdown/MDX format
- **InteractiveComponents**: Array[InteractiveComponent] (optional) - 3D components for this topic (if applicable)

## InteractiveComponent Entity
- **Type**: Enum ["3d-model", "simulation", "animation", "visualization"] (required) - Type of interactive component
- **Title**: String (required) - Component title
- **Description**: String (required) - Component description
- **ModelPath**: String (optional) - Path to 3D model asset (for 3d-model type)
- **Configuration**: Object (optional) - Configuration for the component
- **AccessibilityText**: String (required) - Alternative text for accessibility

## Exercise Entity
- **Title**: String (required) - Exercise title
- **Description**: String (required) - Exercise description
- **Type**: Enum ["multiple-choice", "coding", "interactive", "written"] (required)
- **Difficulty**: Enum ["beginner", "intermediate", "advanced"] (required)
- **Solution**: String (optional) - Exercise solution

## StudentProgress Entity
- **StudentId**: String (required) - Unique identifier for the student
- **ModuleProgress**: Array[ModuleProgress] (required) - Progress for each module
- **LastAccessed**: Date (required) - Last time the student accessed the textbook
- **CompletedModules**: Array[String] (optional) - List of completed module names

## ModuleProgress Entity
- **ModuleId**: String (required) - Reference to the module
- **CompletionPercentage**: Number (required) - Percentage of module completed (0-100)
- **TimeSpent**: Number (required) - Time spent on the module in minutes
- **AssessmentScore**: Number (optional) - Score on module assessment (0-100)
- **LastAccessed**: Date (required) - Last time the module was accessed