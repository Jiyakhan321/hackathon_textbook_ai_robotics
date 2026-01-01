import fs from 'fs';
import { execSync } from 'child_process';

// Read the original config
const configPath = './docusaurus.config.ts';
let configContent = fs.readFileSync(configPath, 'utf8');

// Create a backup
fs.writeFileSync('./docusaurus.config.ts.bak', configContent);

// Temporarily modify the baseUrl for Vercel deployment
let modifiedConfig = configContent.replace(
  "baseUrl: '/hackathon_textbook_ai_robotics/',",
  "baseUrl: '/',"
);

// Replace the entire docs configuration with one that has routeBasePath set
// We need to find the complete docs configuration block and replace it
const docsStart = configContent.indexOf('docs: {');
if (docsStart !== -1) {
  // Find the matching closing brace for the docs config
  let openBraces = 0;
  let pos = docsStart;
  let docsEnd = -1;
  
  while (pos < configContent.length) {
    if (configContent[pos] === '{') {
      openBraces++;
    } else if (configContent[pos] === '}') {
      openBraces--;
      if (openBraces === 0) {
        docsEnd = pos + 1; // Include the closing brace
        break;
      }
    }
    pos++;
  }
  
  if (docsEnd !== -1) {
    const originalDocsConfig = configContent.substring(docsStart, docsEnd);
    
    // Create the new docs config with routeBasePath
    const newDocsConfig = `docs: {
          sidebarPath: require.resolve('./sidebars.ts'),
          editUrl:
            'https://github.com/Jiyakhan321/hackathon_textbook_ai_robotics/tree/main/',
          routeBasePath: '/docs', // Change route base path to avoid conflict with homepage
        }`;
    
    // Replace the original docs config with the new one
    modifiedConfig = configContent.substring(0, docsStart) + newDocsConfig + configContent.substring(docsEnd);
  }
}

// Write the modified config
fs.writeFileSync(configPath, modifiedConfig);

try {
  // Run the build command
  console.log('Building for Vercel deployment...');
  execSync('npm run build', { stdio: 'inherit' });
  console.log('Build completed successfully!');
} catch (error) {
  console.error('Build failed:', error.message);
  // Restore the original config even if build fails
  fs.writeFileSync(configPath, configContent);
  fs.unlinkSync('./docusaurus.config.ts.bak');
  process.exit(1);
}

// Restore the original config after successful build
fs.writeFileSync(configPath, configContent);
fs.unlinkSync('./docusaurus.config.ts.bak');
console.log('Original configuration restored.');