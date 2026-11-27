# MedSim-Gemini Deployment Guide (Cloud Run)

This guide outlines the deployment of MedSim to Google Cloud Run using the 100% Google Gemini Stack.

## Prerequisites

1. Google Cloud SDK (`gcloud`) installed and authenticated.
2. Active GCP project configured:
   ```powershell
   gcloud config set project YOUR_GCP_PROJECT_ID
   ```
3. Environment variables configured in `backend/.env.local` (never committed to git).

## Deployment Methods

### Method 1: Using the Automated PowerShell Script (Recommended for Windows)

Run the deployment script from the repository root:
```powershell
.\scripts\deploy-cloudrun.ps1
```

This script:
- Verifies you are on branch `MedSim-Gemini`.
- Enables required Cloud Run and Cloud Build APIs.
- Builds and deploys the container image with strict flags (`--timeout=3600 --session-affinity --no-use-http2 --port=8080`).
- Tests the deployed `/health` endpoint.

### Method 2: Using Cloud Build

Submit a cloud build job:
```powershell
gcloud builds submit --config=cloudbuild.yaml .
```

## Cloud Run Service Parameters

- **Service Name:** `medsim-gemini`
- **Region:** `us-central1`
- **Timeout:** `3600` seconds
- **Session Affinity:** Enabled (`--session-affinity`)
- **HTTP/2:** Disabled (`--no-use-http2`)
- **Port:** `8080`
- **CPU / Memory:** 1 vCPU, 1 GiB Memory
- **Instances:** 0 to 3 auto-scaling (`--min-instances=0 --max-instances=3`)
