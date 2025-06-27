@echo off
title Strategic Counsel - Multi-Agent AI Legal Analysis Platform
color 0B
echo.
echo  ████████╗██████╗  █████╗ ████████╗███████╗ ██████╗ ██╗ ██████╗
echo  ██╔═════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝██╔════╝ ██║██╔════╝
echo  ███████╗   ██║   ██████╔╝   ██║   █████╗  ██║  ███╗██║██║     
echo  ╚════██║   ██║   ██╔══██╗   ██║   ██╔══╝  ██║   ██║██║██║     
echo  ███████║   ██║   ██║  ██║   ██║   ███████╗╚██████╔╝██║╚██████╗
echo  ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝ ╚═╝ ╚═════╝
echo.
echo                    COUNSEL - Multi-Agent AI Legal Platform
echo.
echo Starting Strategic Counsel...
echo.

REM Kill any existing streamlit processes
echo Stopping any existing instances...
wsl pkill -f streamlit 2>nul
timeout /t 2 /nobreak >nul

REM Start Strategic Counsel in WSL
echo Launching Strategic Counsel...
echo Please wait while the application starts...
echo.

REM Start in background and open browser
start /min wsl bash -c "cd /home/jcockburn/SC-Gen-4 && python3 -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"

echo Waiting for application to start...
timeout /t 8 /nobreak >nul

echo Opening browser...
start http://localhost:8501

echo.
echo ✅ Strategic Counsel is now running!
echo 🌐 Access URL: http://localhost:8501
echo.
echo Press any key to stop Strategic Counsel...
pause >nul

echo.
echo Stopping Strategic Counsel...
wsl pkill -f streamlit 2>nul
echo Strategic Counsel stopped.
echo.
echo Press any key to close...
pause >nul 