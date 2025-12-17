<!--
     SYNC IMPACT REPORT
     Version change: 1.0.0 -> 1.1.0
     Added sections: Core Principles (updated for Physical AI & Humanoid Robotics), Additional Constraints, Development Workflow, Governance
     Modified principles: I-VI (updated for Physical AI & Humanoid Robotics focus)
     Removed sections: N/A
     Templates requiring updates:
       - .specify/templates/plan-template.md: ✅ updated
       - .specify/templates/spec-template.md: ✅ updated
       - .specify/templates/tasks-template.md: ✅ updated
       - .specify/templates/commands/*.md: ✅ updated
       - README.md: ⚠ pending
     Follow-up TODOs: None
     -->
# Physical AI & Humanoid Robotics Textbook Constitution

## Core Principles

### I. Interactive Learning-First
Prioritize immersive, interactive 3D experiences for complex concepts; All content must include interactive elements to enhance understanding; Clear, accessible 3D visualizations required for all embodied AI and robotics topics

### II. Embodied Intelligence Focus
Every module should emphasize the connection between digital brain and physical body; Content must bridge theoretical AI concepts with practical robotic implementations; Focus on the integration of perception, cognition, and action in embodied systems

### III. Test-First (NON-NEGOTIABLE)
TDD mandatory: Tests written → User approved → Tests fail → Then implement; Red-Green-Refactor cycle strictly enforced for all code changes

### IV. Performance Optimization for 3D
Focus on efficient 3D rendering and smooth interactive experiences; Optimize 3D assets, textures, and scene complexity; Ensure responsive interactions across different hardware capabilities

### V. Educational Accessibility Standards
All content must be accessible to students with varying technical backgrounds; Clear prerequisites and learning pathways required; Step-by-step progression from foundational to advanced concepts

### VI. Multi-Modal Learning Design
All concepts must be presented through multiple modalities (text, 3D visualization, interactive demos); Support diverse learning styles and preferences; Integration of vision, language, and action understanding

## Additional Constraints
- Technology stack: Docusaurus v3.x with React, React Three Fiber (R3F) for 3D interactivity
- Framework: Docusaurus for book structure, React Three Fiber (R3F) for 3D interactivity
- Target audience: Students with basic AI, Python, and robotics knowledge
- Content structure: 4 modules + Capstone project covering Physical AI systems
- Deployment: Static site hosting (GitHub Pages, Netlify, Vercel)
- Content format: Markdown/MDX with standardized frontmatter and 3D component integration
- Internationalization: Support for multi-language content when needed

## Development Workflow
- All 3D interactive components must be tested across browsers and devices
- Content reviews must verify technical accuracy of AI/robotics concepts
- Build process must pass performance benchmarks for 3D content before merging
- Educational effectiveness must be validated through user feedback
- Interactive elements must be accessible and include alternative text descriptions

## Governance
All PRs/reviews must verify compliance with educational standards; Changes to core curriculum require explicit approval; Use CLAUDE.md for development guidance

**Version**: 1.1.0 | **Ratified**: 2025-12-10 | **Last Amended**: 2025-12-17