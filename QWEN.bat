@echo off
SETLOCAL EnableDelayedExpansion

REM SIMPLIFIED huihui_ai/qwen3-coder-abliterated:30b-a3b-instruct-q3_K_M  LAUNCHER - USING CONFIG.PY SETTINGS
REM Hardware: RTX 5090 32GB, 64GB DDR5, Intel i9 24-core
REM Models: Qwen3 abliterated:30b  (chat) + qwen3-embedding:4b  (embeddings)
REM qwen3-vl:30b (Image and Video processing)
REM Configuration: All settings now controlled by config.py
REM Note: Requires manually started Ollama server
REM ============================================================================

ECHO.
ECHO ================================
ECHO qwen3  SIMPLIFIED LAUNCHER
ECHO ================================
ECHO Hardware: RTX 5090 32GB VRAM
ECHO Models: qwen3-coder-abliterated:30b-a3b-instruct-q3_K_M  + qwen3-embedding:4b + qwen3-vl:30b
ECHO Configuration: Using config.py settings
ECHO Note: Requires manually started Ollama
ECHO ================================
ECHO.

REM Color coding for status messages
COLOR 0A

REM ============================================================================
REM MINIMAL OLLAMA ENVIRONMENT - LET CONFIG.PY HANDLE THE REST
REM ============================================================================

ECHO [CONFIG] Setting minimal environment variables...

REM Only set essential base configuration
SET "OLLAMA_BASE_URL=http://localhost:11434"
SET "OLLAMA_MAX_LOADED_MODELS=2"
SET OLLAMA_KEEP_ALIVE=-1
SET OLLAMA_GPU_MEMORY_FRACTION=0.95


REM Display configuration
ECHO.
ECHO [CONFIG] Minimal Settings Applied:
ECHO   Base URL: http://localhost:11434
ECHO   Max Models: 2 (QWEN 3 + QWEN3 Embeddings)
ECHO   All other settings: Controlled by config.py
ECHO.

REM ============================================================================
REM PROCESS CLEANUP
REM ============================================================================

ECHO [CLEANUP] Stopping existing processes (preserving Ollama)...
taskkill /f /im streamlit.exe 2>nul
taskkill /f /im python.exe /fi "WINDOWTITLE eq Streamlit*" 2>nul
REM Note: Ollama will remain running as manually started

ECHO [CLEANUP] Waiting for cleanup...
timeout /t 3 /nobreak

REM ============================================================================
REM OLLAMA SERVER VERIFICATION (MANUAL STARTUP REQUIRED)
REM ============================================================================

ECHO [OLLAMA] Checking for manually started Ollama server...

REM Check if Ollama is accessible via API
curl -s http://localhost:11434/api/tags >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO [ERROR] Ollama server is not running!
    ECHO [ERROR] Please start Ollama manually before running this script
    ECHO [INFO] Start Ollama GUI and configure your context window as needed
    ECHO [INFO] Then run this script again
    PAUSE
    GOTO CLEANUP_EXIT
) ELSE (
    ECHO [OLLAMA] Server is running and accessible
    ECHO [OLLAMA] You can adjust context window via Ollama GUI as needed
)

REM Check current model status
ECHO [OLLAMA] Checking current model status...
curl -s http://localhost:11434/api/ps 2>nul | find "name" >nul 2>&1
IF NOT ERRORLEVEL 1 (
    ECHO [OLLAMA] Models are currently loaded
    curl -s http://localhost:11434/api/ps
) ELSE (
    ECHO [OLLAMA] No models currently loaded - they will load on first use
)

REM ============================================================================
REM MODEL AVAILABILITY CHECK
REM ============================================================================

ECHO [MODELS] Verifying required models are available...
curl -s http://localhost:11434/api/tags | findstr "qwen" >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO [WARNING] QWEN model may not be installed
    ECHO [INFO] Check Ollama GUI to pull required models if needed
) ELSE (
    ECHO [MODELS] QWEN model appears to be available
)

curl -s http://localhost:11434/api/tags | findstr "qwen3-embedding:4b" >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO [WARNING] qwen3-embedding:4b embedding model may not be installed
    ECHO [INFO] Consider pulling qwen3-embedding:4b 
) ELSE (
    ECHO [MODELS] qwen3-embedding:4b g model appears to be available
)

REM ============================================================================
REM VALIDATE CONFIG.PY SETTINGS
REM ============================================================================

ECHO [CONFIG] Validating config.py settings...
cd /d C:\Users\kenba\source\repos\Ollama3

REM ============================================================================
REM STREAMLIT LAUNCH
REM ============================================================================

ECHO [STREAMLIT] Launching Streamlit interface...

REM Launch Streamlit in new window
START "Streamlit - Ollama Interface" cmd /c "streamlit run Main.py"

ECHO [STREAMLIT] Waiting for interface startup...
timeout /t 10 /nobreak

ECHO [STREAMLIT] Interface launched in separate window

REM ============================================================================
REM MONITORING
REM ============================================================================

ECHO.
ECHO ================================
ECHO SYSTEM STATUS MONITORING
ECHO ================================
ECHO Ollama Server: Running on localhost:11434 (manually started)
ECHO Streamlit: Running in separate window
ECHO Models: QWEN3:30b + qwen3-embedding:4b 
ECHO Configuration: Using config.py settings
ECHO Context Window: Configurable via Ollama GUI
ECHO.
ECHO To monitor GPU usage manually:
ECHO   nvidia-smi -l 30
ECHO.
ECHO To check model GPU utilization:
ECHO   ollama ps
ECHO.
ECHO To adjust context window:
ECHO   Use Ollama GUI interface
ECHO.
ECHO Press Ctrl+C to stop project processes
ECHO ================================
ECHO.

:MONITOR_LOOP
REM Check if Ollama is still accessible
curl -s http://localhost:11434/api/tags >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO [%TIME%] ERROR: Ollama server is no longer accessible!
    ECHO [ERROR] Please check Ollama GUI or restart Ollama manually
    ECHO [INFO] Script will continue monitoring for server return
)

REM Check if Streamlit is running
tasklist /FI "IMAGENAME eq streamlit.exe" 2>NUL | find /I "streamlit.exe" >NUL
IF ERRORLEVEL 1 (
    tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I "python.exe" >NUL
    IF ERRORLEVEL 1 (
        ECHO [%TIME%] WARNING: Streamlit may have stopped
        ECHO [INFO] Check browser or restart manually if needed
    )
)

ECHO [%TIME%] System running - Ollama managed manually - Check 'ollama ps' for GPU utilization
timeout /t 60 /nobreak
GOTO MONITOR_LOOP

REM ============================================================================
REM CLEANUP AND EXIT
REM ============================================================================

:CLEANUP_EXIT
ECHO.
ECHO [CLEANUP] Shutting down project processes (preserving Ollama)...
taskkill /f /im streamlit.exe 2>nul
taskkill /f /im python.exe /fi "WINDOWTITLE eq Streamlit*" 2>nul
REM Note: Ollama left running - stop manually if needed via GUI

ECHO [CLEANUP] Project processes stopped
ECHO [INFO] Ollama server left running for manual management
ECHO [INFO] GPU monitoring: nvidia-smi -l 30
ECHO [INFO] Context window: Adjust via Ollama GUI as needed
ECHO [INFO] Next time: Start Ollama manually first, then run this script
PAUSE
EXIT /B 0

REM Handle Ctrl+C gracefully
:CTRL_C_HANDLER
GOTO CLEANUP_EXIT