#!/bin/bash

# Create documentation directory structure
mkdir -p docs/{technical,user,api,maintenance}

# Move documentation files to appropriate directories
mv docs/system_documentation.md docs/technical/
mv docs/api_documentation.md docs/api/
mv docs/user_manual.md docs/user/
mv docs/installation_maintenance.md docs/maintenance/

# Create index files
echo "# Medical Chatbot Documentation" > docs/README.md
echo "See subdirectories for specific documentation:" >> docs/README.md
echo "- [Technical Documentation](technical/)" >> docs/README.md
echo "- [API Documentation](api/)" >> docs/README.md
echo "- [User Manual](user/)" >> docs/README.md
echo "- [Installation & Maintenance](maintenance/)" >> docs/README.md

# Create documentation website
mkdir -p docs/website
cd docs/website

# Create package.json for documentation website
cat > package.json << EOL
{
  "name": "medical-chatbot-docs",
  "version": "1.0.0",
  "description": "Medical Chatbot Documentation",
  "scripts": {
    "start": "docsify serve",
    "build": "docsify-cli generate ."
  },
  "dependencies": {
    "docsify-cli": "^4.4.4"
  }
}
EOL

# Create index.html for documentation website
cat > index.html << EOL
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Medical Chatbot Documentation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify@4/lib/themes/vue.css">
</head>
<body>
  <div id="app"></div>
  <script>
    window.$docsify = {
      name: 'Medical Chatbot',
      repo: 'https://github.com/your-username/medical-chatbot',
      loadSidebar: true,
      subMaxLevel: 3,
      search: 'auto',
      auto2top: true
    }
  </script>
  <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
  <script src="//cdn.jsdelivr.net/npm/docsify/lib/plugins/search.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-bash.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-python.min.js"></script>
</body>
</html>
EOL

# Create _sidebar.md for documentation navigation
cat > _sidebar.md << EOL
* [Home](/)
* Technical Documentation
  * [System Overview](/technical/system_documentation.md)
* API Documentation
  * [API Reference](/api/api_documentation.md)
* User Guide
  * [User Manual](/user/user_manual.md)
* Maintenance
  * [Installation & Maintenance](/maintenance/installation_maintenance.md)
EOL

echo "Documentation setup completed."