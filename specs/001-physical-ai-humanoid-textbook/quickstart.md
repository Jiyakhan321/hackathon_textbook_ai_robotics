# Quickstart Guide: Physical AI & Humanoid Robotics Interactive 3D Textbook

## Prerequisites
- Node.js v18 or higher
- npm or yarn package manager
- Git
- Basic knowledge of Markdown and JavaScript/React (for interactive components)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Hackathone\ Book
```

### 2. Navigate to the Website Directory
```bash
cd my-website
```

### 3. Install Dependencies
```bash
npm install
# or
yarn install
```

### 4. Start the Development Server
```bash
npm start
# or
yarn start
```

This will start the development server and open the textbook in your default browser at `http://localhost:3000`.

## Adding New Content

### Adding a New Topic to a Module
1. Navigate to the appropriate module directory in `docs/`
2. Create a new Markdown file with descriptive name
3. Add frontmatter with title and description
4. Write your content using Markdown syntax
5. Update `sidebars.js` to include the new page in the navigation

### Adding Interactive 3D Components
For Modules 3, 4, and Capstone Project:
1. Create the 3D component in `src/components/Interactive3D/`
2. Import and use the component in your MDX file
3. Ensure the component has proper accessibility text
4. Test performance across different hardware capabilities

## Key Configuration Files

- `docusaurus.config.js` - Main configuration for the site (title, description, themes, plugins)
- `sidebars.js` - Navigation structure for the textbook
- `src/theme/MDXComponents.js` - Custom components for MDX files
- `static/models/` - Directory for 3D model assets

## Building for Production

To build the static files for deployment:

```bash
npm run build
# or
yarn build
```

The built files will be available in the `build/` directory and can be deployed to any static hosting service.

## Running Tests

To run tests:

```bash
npm test
# or
yarn test
```

## Contributing Guidelines

1. Follow the existing code style and naming conventions
2. Ensure all 3D components are accessible and performant
3. Write clear, educational content with appropriate examples
4. Test content across different browsers and devices
5. Follow the project's constitution principles in all contributions