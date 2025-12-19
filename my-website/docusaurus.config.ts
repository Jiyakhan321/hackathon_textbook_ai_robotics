module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Comprehensive Guide to Embodied Intelligence',
  url: 'https://Jiyakhan321.github.io',  // Base domain, NOT the git URL
  baseUrl: '/hackathon_textbook_ai_robotics/',  // Repo name for GitHub Pages

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',

  // Even with a dark theme, we want to allow both color modes
  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.ts'),
          editUrl:
            'https://github.com/Jiyakhan321/hackathon_textbook_ai_robotics/tree/main/',
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/Jiyakhan321/hackathon_textbook_ai_robotics/tree/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'Home',
      hideOnScroll: false,
      logo: {
        alt: 'Home Icon',
        src: 'img/home-icon.svg',
        width: 32,
        height: 32
      },
      items: [
        { type: 'docSidebar', sidebarId: 'tutorialSidebar', position: 'left', label: 'Book' },
        { to: '/blog', label: 'Blog', position: 'left' },
        {
          href: 'https://github.com/Jiyakhan321/hackathon_textbook_ai_robotics',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [{ label: 'Book', to: '/docs/intro' }],
        },
        {
          title: 'Community',
          items: [
            { label: 'Stack Overflow', href: 'https://stackoverflow.com/questions/tagged/docusaurus' },
            { label: 'Discord', href: 'https://discordapp.com/invite/docusaurus' },
            { label: 'Twitter', href: 'https://twitter.com/docusaurus' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'Blog', to: '/blog' },
            {
              label: 'GitHub',
              href: 'https://github.com/Jiyakhan321/hackathon_textbook_ai_robotics',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} My Project, Inc. Built with Docusaurus.`,
    },
    prism: {
      theme: require('prism-react-renderer').themes.github,
      darkTheme: require('prism-react-renderer').themes.dracula,
    },
  },
};
