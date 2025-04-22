# PowerShell script to automate OpenWebUI configuration for Windows
# Fixed version to address encoding issues and file handling

# Environment variables
$API_URL = "http://ragbot-api:8000"
$API_ENDPOINT = "/api"
$WEBUI_CONTAINER = "rag_poc-openwebui-1"

# Get the OpenAI API key from .env file
$OPENAI_API_KEY = (Get-Content .env | Where-Object { $_ -match 'OPENAI_API_KEY=' } | ForEach-Object { $_ -replace 'OPENAI_API_KEY=', '' })

if (-not $OPENAI_API_KEY) {
    Write-Host "Error: OPENAI_API_KEY not found in .env file" -ForegroundColor Red
    exit 1
}

Write-Host "Setting up OpenWebUI with RAGbot API connection..." -ForegroundColor Cyan

# Wait for OpenWebUI to be ready
Write-Host "Waiting for OpenWebUI container to be ready..." -ForegroundColor Yellow
do {
    $containerStatus = docker inspect $WEBUI_CONTAINER --format='{{.State.Running}}' 2>$null
    if ($containerStatus -ne "true") {
        Write-Host "Waiting for OpenWebUI container..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
} while ($containerStatus -ne "true")

Write-Host "OpenWebUI container is running, configuring settings..." -ForegroundColor Green
Start-Sleep -Seconds 45  # Additional wait to ensure SQLite database is initialized

# Create the SQL script content
$setupSql = @"
-- Add the API configuration to OpenWebUI
INSERT OR REPLACE INTO settings (key, value, created_at, updated_at)
VALUES 
('CUSTOM_API_BASE_URL', '$API_URL', datetime('now'), datetime('now')),
('CUSTOM_API_KEY', '$OPENAI_API_KEY', datetime('now'), datetime('now')),
('CUSTOM_API_CONTEXT_SIZE', '4096', datetime('now'), datetime('now')),
('API_BACKEND', 'custom', datetime('now'), datetime('now')),
('DEFAULT_MODEL', 'ragbot', datetime('now'), datetime('now'));

-- Add the model configuration
INSERT OR REPLACE INTO models (
    id, name, model_id, description, context_length, default_prompt_template_id,
    hidden, display_name, chat_template_id, 
    vendor_id, created_at, updated_at
)
VALUES (
    'ragbot', 'RAGbot', 'ragbot', 
    'Retrieval-augmented generation assistant for proposal generation', 
    4096, null, 0, 'RAGbot', null, 
    'custom', datetime('now'), datetime('now')
);
"@

# Execute SQL commands to pre-configure OpenWebUI directly using echo and stdin
Write-Host "Applying configuration to OpenWebUI database..." -ForegroundColor Yellow

# Method 1: Use echo directly with docker exec
try {
    $sqlCommand = "cd /app/backend && echo `"$setupSql`" | sqlite3 data/settings.db"
    docker exec $WEBUI_CONTAINER bash -c $sqlCommand
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully configured OpenWebUI database" -ForegroundColor Green
    } else {
        Write-Host "Warning: Configuration may not have been applied properly" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error applying configuration: $_" -ForegroundColor Red
}

# Restart OpenWebUI to apply changes
Write-Host "Restarting OpenWebUI container to apply configuration..." -ForegroundColor Yellow
docker restart $WEBUI_CONTAINER

Write-Host "✅ OpenWebUI has been configured to use the RAGbot API with your OpenAI API key" -ForegroundColor Green
Write-Host "You can now access the web interface at http://localhost:3000" -ForegroundColor Green
Write-Host "The connection to the API should be pre-configured" -ForegroundColor Green