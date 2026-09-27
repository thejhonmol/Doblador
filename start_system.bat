@echo off
chcp 65001 >nul
title Doblador - System Starter
cd /d "%~dp0"

set PY_PORT=8000
set API_PORT=3000
set WEB_PORT=5173
set PY_WAIT_TRIES=90
set PY_WAIT_SECONDS=2

echo ========================================================
echo        STARTING AUTONOMOUS AI DUBBING SYSTEM
echo ========================================================
echo.

:: ── 0. Check and free ports, cleanly stopping prior instances ──
echo [0/4] Checking ports (8000, 3000, 5173)...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\prepare_ports.ps1"
if %errorlevel% neq 0 (
    echo.
    pause
    exit /b 1
)

:: 1. Start Docker (PostgreSQL and Redis)
echo.
echo [1/4] Starting Docker containers (PostgreSQL and Redis)...
docker compose up -d
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start Docker. Please make sure Docker Desktop is running.
    pause
    exit /b %errorlevel%
)
echo      -> Docker containers started successfully.
ping 127.0.0.1 -n 3 >nul

:: 2. Start Python Microservices (FastAPI / MOSS / Demucs / Diarization)
echo.
echo [2/4] Starting Python Services (loading Whisper, MOSS, Demucs on GPU)...
start "Doblador - Python Services" /D "%~dp0python-services" cmd /k "title Doblador - Python Services && color 0B && venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port %PY_PORT%"

set PY_TRIES=0
:wait_python
set /a PY_TRIES+=1
ping 127.0.0.1 -n 3 >nul
curl -s -o NUL -w "%%{http_code}" "http://127.0.0.1:%PY_PORT%/" 2>NUL | findstr /b "200" >NUL
if not errorlevel 1 goto python_ready
if %PY_TRIES% GEQ %PY_WAIT_TRIES% goto python_timeout
echo          ... loading models onto GPU (%PY_TRIES%/%PY_WAIT_TRIES%) ...
goto wait_python

:python_timeout
echo.
echo [ERROR] Python services did not respond on port %PY_PORT% after %PY_WAIT_TRIES% attempts.
echo         Please inspect the "Doblador - Python Services" terminal window for details.
pause
exit /b 1

:python_ready
echo      -^> [OK] AI models loaded and ready on port %PY_PORT%.

:: 3. Start Orchestrator (Node.js / BullMQ)
echo.
echo [3/4] Compiling TypeScript orchestrator...
pushd "%~dp0orchestrator"
call npm run build
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] TypeScript compilation failed. See errors above.
    popd
    pause
    exit /b 1
)
popd
echo [3/4] Starting Orchestrator (Node.js on port %API_PORT%)...
start "Doblador - Orchestrator" /D "%~dp0orchestrator" cmd /k "title Doblador - Orchestrator && color 0A && node src/index.js"
echo      -> Orchestrator started in background.
ping 127.0.0.1 -n 3 >nul

:: 4. Start Frontend (Vite)
echo.
echo [4/4] Starting Web Frontend (Vite on port %WEB_PORT%)...
start "Doblador - Frontend" /D "%~dp0frontend" cmd /k "title Doblador - Frontend && color 0E && npm.cmd run dev -- --port %WEB_PORT%"
echo      -> Frontend started in background.
ping 127.0.0.1 -n 3 >nul

echo.
echo ========================================================
echo        SYSTEM STARTED SUCCESSFULLY
echo ========================================================
echo  * Web Frontend:    http://localhost:%WEB_PORT%/
echo  * Orchestrator:    http://127.0.0.1:%API_PORT%/
echo  * Python API:      http://127.0.0.1:%PY_PORT%/
echo  * Database:        localhost:5432 (PostgreSQL)
echo  * Queues:          localhost:6379 (Redis)
echo ========================================================
echo.
echo To shut down all services cleanly, run 'stop_system.bat'.
echo.
if "%1"=="--no-pause" goto done
pause
:done
