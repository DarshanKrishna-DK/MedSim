<#
.SYNOPSIS
    Deploys MedSim-Gemini to Google Cloud Run with strict configuration.
.DESCRIPTION
    Verifies that the current git branch is MedSim-Gemini, validates gcloud and project settings,
    enables required GCP APIs, builds/deploys the container image, and tests the health endpoint.
#>

param(
    [string]$Region = "us-central1",
    [string]$ServiceName = "medsim-gemini",
    [switch]$UseCloudBuild
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "MedSim-Gemini Cloud Run Deployment Script" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Verify Branch is MedSim-Gemini
try {
    $currentBranch = (git branch --show-current).Trim()
    Write-Host "[Branch Guard] Current git branch: $currentBranch" -ForegroundColor Yellow
    if ($currentBranch -ne "MedSim-Gemini") {
        Write-Host "ERROR: Deployment must be performed from the 'MedSim-Gemini' branch! Current branch is '$currentBranch'." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "[Branch Guard] Git branch check skipped (not a git repo or git unavailable)." -ForegroundColor DarkYellow
}

# 2. Check gcloud CLI and active project
Write-Host "[GCloud Check] Verifying gcloud CLI..." -ForegroundColor Yellow
try {
    $gcloudVersion = gcloud --version 2>&1
} catch {
    Write-Host "ERROR: 'gcloud' CLI is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

$project = (gcloud config get-value project 2>&1).Trim()
if (-not $project -or $project -eq "(unset)") {
    Write-Host "ERROR: No active gcloud project configured. Run 'gcloud config set project <PROJECT_ID>'." -ForegroundColor Red
    exit 1
}
Write-Host "[GCloud Check] Active Project: $project | Region: $Region" -ForegroundColor Green

# 3. Enable Required APIs
Write-Host "[GCloud Setup] Enabling required Cloud APIs..." -ForegroundColor Yellow
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com containerregistry.googleapis.com --quiet

# 4. Extract allowlisted Gemini/Cloud Run environment variables ONLY (never upload legacy vendor keys)
$allowedKeys = @(
    "GEMINI_API_KEY",
    "PATIENT_LLM_PROVIDER",
    "GEMINI_FAST_MODEL",
    "GEMINI_REASONING_MODEL",
    "GEMINI_LIVE_MODEL",
    "GEMINI_TTS_MODEL",
    "GEMINI_EFFORT_CAP",
    "BACKEND_SHARED_SECRET",
    "MEDSIM_SHARED_SECRET",
    "FRONTEND_ORIGINS",
    "MEDSIM_ALLOWED_ORIGINS"
)
$envMap = @{}
$envLocalPath = Join-Path $PSScriptRoot "..\backend\.env.local"
if (Test-Path $envLocalPath) {
    Write-Host "[Env] Loading allowlisted Gemini env vars from backend/.env.local (secrets masked)..." -ForegroundColor Yellow
    foreach ($line in Get-Content $envLocalPath) {
        $trimmed = $line.Trim()
        if ($trimmed -and -not $trimmed.StartsWith("#") -and $trimmed.Contains("=")) {
            $parts = $trimmed.Split("=", 2)
            $k = $parts[0].Trim()
            $v = $parts[1].Trim()
            if ($allowedKeys -contains $k -and $v) {
                $envMap[$k] = $v
            }
        }
    }
}
foreach ($k in $allowedKeys) {
    $envVal = [Environment]::GetEnvironmentVariable($k)
    if ($envVal) {
        $envMap[$k] = $envVal
    }
}
$envArgs = @()
foreach ($k in $envMap.Keys) {
    $envArgs += "$k=$($envMap[$k])"
}

# 5. Build and Deploy
if ($UseCloudBuild) {
    Write-Host "[Cloud Build] Submitting build via cloudbuild.yaml..." -ForegroundColor Cyan
    gcloud builds submit --config=cloudbuild.yaml .
} else {
    Write-Host "[Cloud Run] Deploying directly from source with gcloud run deploy..." -ForegroundColor Cyan
    $deployArgs = @(
        "run", "deploy", $ServiceName,
        "--source", ".",
        "--region", $Region,
        "--platform", "managed",
        "--timeout=3600",
        "--session-affinity",
        "--no-use-http2",
        "--port=8080",
        "--min-instances=0",
        "--max-instances=3",
        "--memory=1Gi",
        "--cpu=1",
        "--allow-unauthenticated"
    )
    if ($envArgs.Count -gt 0) {
        $deployArgs += "--set-env-vars"
        $deployArgs += ($envArgs -join ",")
    }
    gcloud @deployArgs
}

# 6. Retrieve Service URL and verify /health
Write-Host "[Verification] Retrieving service URL..." -ForegroundColor Yellow
$serviceUrl = (gcloud run services describe $ServiceName --region $Region --format="value(status.url)" 2>&1).Trim()

if ($serviceUrl -and $serviceUrl.StartsWith("https://")) {
    Write-Host "SUCCESS: Service deployed at: $serviceUrl" -ForegroundColor Green
    $healthUrl = "$serviceUrl/health"
    Write-Host "[Health Check] Testing $healthUrl..." -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 15
        Write-Host "Health Check Status: $($response.StatusCode)" -ForegroundColor Green
        Write-Host $response.Content
    } catch {
        Write-Host "WARNING: Health check request returned an error or timed out: $_" -ForegroundColor DarkYellow
    }
} else {
    Write-Host "Deployment completed, but could not resolve service URL automatically." -ForegroundColor Yellow
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Deployment workflow finished successfully." -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
