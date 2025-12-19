import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import { BookOpen, Cpu, Cog, Eye, Brain, Zap, Code, Layers, Home as HomeIcon } from 'lucide-react';

// Import the Chatbot component
import Chatbot from '@site/src/components/Chatbot';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', 'heroBanner')}>
      <div className="container">
        <h1 className="hero__title">Physical AI & Humanoid Robotics</h1>
        <p className="hero__subtitle"></p>
        <div className="margin-top--lg">
          <Link
            className="button button--secondary button--lg"
            to={useBaseUrl('/docs/intro')}
          >
            Start Reading
          </Link>
        </div>
      </div>
    </header>
  );
}

function Module({ module, index }) {
  // Simple icon mapping based on module name
  const getIcon = (title, index) => {
    if (title.toLowerCase().includes('intro')) return <BookOpen size={24} />;
    if (title.toLowerCase().includes('robot')) return <Cpu size={24} />;
    if (title.toLowerCase().includes('digital') || title.toLowerCase().includes('twin')) return <Layers size={24} />;
    if (title.toLowerCase().includes('ai') || title.toLowerCase().includes('brain')) return <Brain size={24} />;
    if (title.toLowerCase().includes('vision') || title.toLowerCase().includes('language') || title.toLowerCase().includes('action')) return <Eye size={24} />;
    if (title.toLowerCase().includes('nervous') || title.toLowerCase().includes('system')) return <Zap size={24} />;
    if (title.toLowerCase().includes('simulation')) return <Cog size={24} />;
    if (title.toLowerCase().includes('module')) return <Code size={24} />;
    // Default to book icon
    return <BookOpen size={24} />;
  };

  return (
    <div className="module-item">
      <div className="module-content">
        <div className="icon-circle">
          {getIcon(module.title, index)}
        </div>
        <h3 className="module-title">{module.title}</h3>
        <p className="module-description">{module.description || 'Learn more about this topic'}</p>
      </div>
      <div className="module-footer">
        <Link
          className="button button--primary"
          to={useBaseUrl(module.slug || module.id)}
        >
          Start Module
        </Link>
      </div>
    </div>
  );
}

function CompleteJourneyCard({ modules }) {
  return (
    <div className="col col--12 card-item">
      <div className="card">
        <div className="card__header">
          <h3 className="card__title">Complete Journey</h3>
        </div>
        <div className="card__body">
          <div className="modules-container">
            {modules.map((module, index) => (
              <Module key={module.id} module={module} index={index} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// Function to get docs data
function useAllDocsData() {
  try {
    // Try to get docs data from the generated config
    const { docs } = require('@generated/docusaurus.config').default || {};
    if (docs && docs.current) {
      return docs.current;
    }
    return [];
  } catch (error) {
    console.warn('Could not load docs data, returning empty array:', error);
    return [];
  }
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();

  // Get all docs using our custom data fetching function
  const allDocs = useAllDocsData();

  // Filter to get only the main module docs that are in the root of the docs directory
  const allModules = allDocs.filter(doc => {
    // Only include docs that are directly under docs/ (not in subdirectories)
    // Based on the sidebar structure, these are the main module introductions
    const pathParts = doc.permalink ? doc.permalink.split('/') : [];
    return pathParts.length === 3 &&
           pathParts[1] === 'docs' &&
           (doc.id.includes('/intro') ||
            doc.id === 'intro' ||
            doc.id === 'physical-ai-humanoid-robotics' ||
            doc.id === 'module-1-robotic-nervous-system/intro' ||
            doc.id === 'module-2-digital-twin/intro' ||
            doc.id === 'module-3-ai-robot-brain/intro' ||
            doc.id === 'module-4-vision-language-action/intro');
  });

  // Create a fallback array if docs are not available or filtered incorrectly
  const fallbackModules = [
    {
      id: 'intro',
      title: 'Introduction',
      description: 'Getting started with Physical AI & Humanoid Robotics',
      slug: '/docs/intro'
    },
    {
      id: 'module-1-robotic-nervous-system',
      title: 'Module 1: The Robotic Nervous System (ROS 2)',
      description: 'Master ROS 2 fundamentals, nodes, topics, services, and URDF for humanoid robots',
      slug: '/docs/module-1-robotic-nervous-system/intro'
    },
    {
      id: 'module-2-digital-twin',
      title: 'Module 2: The Digital Twin (Gazebo & Unity)',
      description: 'Create physics-accurate simulations with Gazebo and Unity, including sensor modeling',
      slug: '/docs/module-2-digital-twin/intro'
    },
    {
      id: 'module-3-ai-robot-brain',
      title: 'Module 3: The AI-Robot Brain',
      description: 'Implement NVIDIA Isaac for synthetic data, SLAM, and navigation systems',
      slug: '/docs/module-3-ai-robot-brain/intro'
    },
    {
      id: 'module-4-vision-language-action',
      title: 'Module 4: Vision-Language-Action',
      description: 'Build multimodal AI systems with voice commands, LLM planning, and action execution',
      slug: '/docs/module-4-vision-language-action/intro'
    }
  ];

  // Use filtered docs or fallback if needed
  const modulesToUse = allModules.length > 0 ? allModules : fallbackModules;

  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics textbook for hackathon preparation and success">
      <HomepageHeader />
      <main>
        <section className="modules-section">
          <div className="container">
            <Heading as="h2" className="section-title">Learning All Modules</Heading>
            <div className="row">
              <CompleteJourneyCard modules={modulesToUse} />
            </div>
          </div>
        </section>
      </main>
      <Chatbot />
    </Layout>
  );
}