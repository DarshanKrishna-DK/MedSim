@echo off
setlocal
echo ==================================================
echo Starting MedSim-Gemini (Google Gemini Stack)
echo ==================================================

cd /d "%~dp0"

if not exist backend\.env.local (
    echo [Warning] backend\.env.local not found. Copy backend\.env.example to backend\.env.local and add your keys.
)

echo [1/2] Starting FastAPI backend (port 8080)...
start "MedSim Backend" cmd /k "cd /d %~dp0backend && ..\backend\.venv\Scripts\python.exe server.py"

echo [2/2] Starting Vite Frontend dev server...
node node_modules/vite/bin/vite.js

endlocal
