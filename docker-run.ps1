# PowerShell helper script for building and running the website in Docker

param(
    [switch]$Dev,
    [switch]$Rebuild,
    [switch]$WithTools,
    [switch]$WithProxy
)

function Test-Command {
    param([string]$Name)
    return (Get-Command $Name -ErrorAction SilentlyContinue) -ne $null
}

Write-Host "Starting Docker helper for The Big Everything Prompt Library" -ForegroundColor Cyan

if (-not (Test-Command 'docker')) {
    # Attempt to locate Docker in common install locations and add to PATH
    $commonDockerDirs = @(
        "C:\Program Files\Docker\Docker\resources\bin",
        "C:\Program Files\Docker\Docker\frontend",
        "C:\Program Files\Docker\Docker",
        "C:\Program Files\Docker"
    )
    foreach ($d in $commonDockerDirs) {
        if (Test-Path (Join-Path $d 'docker.exe')) {
            Write-Host "Docker CLI found in '$d'. Adding to PATH for this session..." -ForegroundColor Yellow
            $env:PATH = "$d;$env:PATH"
            break
        }
    }

    if (-not (Test-Command 'docker')) {
        Write-Error "Docker CLI was not found. Please install Docker Desktop or add it to your PATH."
        exit 1
    }
}

# Determine compose command/executable
$ComposeExe = $null
$BaseComposeArgs = @()

if (Test-Command 'docker-compose') {
    # Legacy standalone docker-compose
    $ComposeExe = 'docker-compose'
} else {
    # Use 'docker compose' sub-command
    $ComposeExe = 'docker'
    $BaseComposeArgs += 'compose'
}

Push-Location $PSScriptRoot

# Choose compose file / profiles
$ComposeOptions = @()
if ($Dev) {
    $ComposeOptions += '-f'
    $ComposeOptions += 'docker-compose.dev.yml'
}
if ($WithTools) {
    $ComposeOptions += '--profile'
    $ComposeOptions += 'tools'
}
if ($WithProxy) {
    $ComposeOptions += '--profile'
    $ComposeOptions += 'production'
}

# Build option
if ($Rebuild) {
    & $ComposeExe @BaseComposeArgs @ComposeOptions build --pull --no-cache
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Launching containers..." -ForegroundColor Green
& $ComposeExe @BaseComposeArgs @ComposeOptions up -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""; Write-Host "Containers started successfully!" -ForegroundColor Green
Write-Host "Web Interface: http://localhost:8000"
if ($WithProxy) { Write-Host "Nginx proxy:   http://localhost" }
if ($WithTools) { Write-Host "File browser: http://localhost:8080" }
Write-Host "API Docs:     http://localhost:8000/docs"

Pop-Location 