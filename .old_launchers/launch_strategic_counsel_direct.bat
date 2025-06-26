@echo off
title Strategic Counsel - Direct Launcher
cls
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║           Strategic Counsel - AI Legal Platform         ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo [INFO] Initializing application...
echo.

:: Kill any existing streamlit processes to avoid port conflicts
wsl -d Ubuntu-22.04 bash -c "pkill -f streamlit 2>/dev/null || true"

echo [INFO] Starting Strategic Counsel application...
echo [INFO] Server URL: http://localhost:8501
echo [INFO] Browser should open automatically
echo.
echo [TIP] To stop the application, press Ctrl+C or close this window
echo.
echo ══════════════════════════════════════════════════════════
echo.

:: Launch with the most direct approach possible
wsl -d Ubuntu-22.04 /home/jcockburn/.local/bin/streamlit run /home/jcockburn/SC-Gen-3/app.py --server.headless false --server.port 8501

echo.
echo [INFO] Application has stopped.
pause 