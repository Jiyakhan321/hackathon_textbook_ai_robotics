@echo off
echo Setting up git configuration...
git config --global user.name "Jiya Mughal"
git config --global user.email "jiyakhan321@example.com"

echo Checking git status...
git status

echo Adding all files...
git add .

echo Committing changes...
git commit -m "Update full Hackathon Book with latest changes"

echo Setting upstream and pushing to GitHub...
git push -u origin main

echo All changes have been pushed to GitHub successfully!
pause