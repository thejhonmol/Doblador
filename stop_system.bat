@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title Doblador - System Stopper
cd /d "%~dp0"

echo ========================================================
echo        STOPPING AUTONOMOUS AI DUBBING SYSTEM
echo ========================================================
echo.

:: -- 1. Stop Docker containers first --
echo [1/3] Stopping Docker containers (PostgreSQL and Redis)...
docker compose down
echo      -^> Docker stopped and database ports released.

:: -- 2. Free service ports by PID --
echo [2/3] Releasing ports (8000, 3000, 5173)...
for %%P in (8000 3000 5173) do (
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%%P " ^| findstr "LISTENING" 2^>nul') do (
        echo      -^> Terminating PID %%a on port %%P...
        taskkill /f /t /pid %%a >NUL 2>&1
    )
)

:: -- 3. Terminate remaining system windows --
echo [3/3] Closing remaining system service windows...
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq cmd.exe" /FO CSV /NH 2^>nul ^| findstr "Doblador"') do (
    echo      -^> Terminating cmd.exe PID %%a...
    taskkill /f /t /pid %%a >NUL 2>&1
)

echo.
echo ========================================================
echo       ALL SERVICES TERMINATED SUCCESSFULLY
echo ========================================================
echo.
echo Intermediate pipeline files remain stored in orchestrator\uploads\.
echo They can be purged safely whenever no active jobs are running.
echo.
pause
