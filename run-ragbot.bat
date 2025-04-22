@echo off
:: RAGbot Starter Script - Bypasses PowerShell execution policy
:: This batch file automatically handles execution policy and starts RAGbot

echo Starting RAGbot System...

:: Start the containers
echo Starting containers...
docker compose up -d

:: Wait a moment for containers to initialize
echo Waiting for services to start...
timeout /t 5 /nobreak > nul

:: Configure OpenWebUI with execution policy bypass
powershell -ExecutionPolicy Bypass -File setup-openwebui.ps1

echo RAGbot system is now running!
echo Web interface available at: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul