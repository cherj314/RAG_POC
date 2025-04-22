@echo off
:: RAGbot Reset Script - Bypasses PowerShell execution policy
:: This batch file automatically handles execution policy and resets RAGbot

echo RAGbot System Reset
echo ===================

echo Step 1: Stopping all containers...
docker compose down

echo Step 2: Removing volumes...
docker volume rm rag_poc_pgdata rag_poc_openwebui-data 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo No volumes to remove or already removed
) else (
    echo Volumes removed successfully
)

echo Step 3: Pruning unused containers and networks...
docker container prune -f
docker network prune -f
echo Docker resources cleaned up

echo Step 4: Starting PostgreSQL container...
docker compose up -d postgres
echo PostgreSQL container started

echo Step 5: Waiting for PostgreSQL to initialize...
echo   Waiting 15 seconds for database to be ready...
timeout /t 15 /nobreak > nul

echo Step 6: Starting remaining services...
docker compose up -d
echo All services started

echo Step 7: Checking service status...
timeout /t 5 /nobreak > nul
docker compose ps

echo Step 8: Configuring OpenWebUI...
timeout /t 10 /nobreak > nul
powershell -ExecutionPolicy Bypass -File setup-openwebui.ps1

echo.
echo RAGbot system has been reset and restarted!
echo Web interface will be available at: http://localhost:3000
echo API endpoint available at: http://localhost:8000
echo.
echo Press any key to exit...
pause > nul