# Script to push files to GitHub using REST API
# Requires GitHub Personal Access Token with 'repo' permissions

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubToken
)

$repoOwner = "rohith20069"
$repoName = "Academic-Abstract-Classifier"
$baseUrl = "https://api.github.com/repos/$repoOwner/$repoName"

$headers = @{
    "Authorization" = "token $GitHubToken"
    "Accept" = "application/vnd.github.v3+json"
}

# Function to encode file content to base64
function Encode-File {
    param([string]$FilePath)
    $bytes = [System.IO.File]::ReadAllBytes($FilePath)
    [Convert]::ToBase64String($bytes)
}

# Function to push a file to GitHub
function Push-File {
    param(
        [string]$FilePath,
        [string]$Content,
        [string]$Message = "Add $FilePath"
    )
    
    $relativePath = $FilePath.Replace((Get-Location).Path + "\", "").Replace("\", "/")
    
    $body = @{
        message = $Message
        content = $Content
    } | ConvertTo-Json
    
    $uri = "$baseUrl/contents/$relativePath"
    
    try {
        $response = Invoke-RestMethod -Uri $uri -Method Put -Headers $headers -Body $body -ContentType "application/json"
        Write-Host "✓ Pushed: $relativePath" -ForegroundColor Green
        return $true
    } catch {
        if ($_.Exception.Response.StatusCode -eq 422) {
            Write-Host "⚠ File exists or error: $relativePath" -ForegroundColor Yellow
        } else {
            Write-Host "✗ Failed: $relativePath - $($_.Exception.Message)" -ForegroundColor Red
        }
        return $false
    }
}

Write-Host "Starting to push files to GitHub..." -ForegroundColor Cyan

# Files to push (excluding those in .gitignore)
$filesToPush = @(
    "app.py",
    "train_model.py",
    "generate_visualizations.py",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    "PROJECT_EXPLANATION.md",
    "project_report.tex",
    ".gitignore"
)

# Push each file
foreach ($file in $filesToPush) {
    if (Test-Path $file) {
        $content = Encode-File -FilePath $file
        Push-File -FilePath $file -Content $content
    }
}

Write-Host "`nDone! Files pushed to GitHub." -ForegroundColor Green

