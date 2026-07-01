# Script to push project to GitHub
# Make sure Git is installed before running this script

Write-Host "Initializing Git repository..." -ForegroundColor Green

# Initialize git if not already initialized
if (-not (Test-Path ".git")) {
    git init
}

# Add remote (will update if already exists)
git remote remove origin 2>$null
git remote add origin https://github.com/rohith20069/Academic-Abstract-Classifier.git

Write-Host "Adding files..." -ForegroundColor Green
git add .

Write-Host "Creating commit..." -ForegroundColor Green
git commit -m "Initial commit: Academic Abstract Classifier project"

Write-Host "Pushing to GitHub..." -ForegroundColor Green
git branch -M main
git push -u origin main

Write-Host "Done! Your project has been pushed to GitHub." -ForegroundColor Green

