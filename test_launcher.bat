@echo off
title Strategic Counsel - Launcher Test
echo Strategic Counsel Launcher Diagnostic
echo =====================================
echo.

echo Testing WSL connection...
wsl echo "WSL is working" || (echo ❌ WSL not working & pause & exit)
echo ✅ WSL connection OK
echo.

echo Testing Python in WSL...
wsl python3 --version || (echo ❌ Python3 not found in WSL & pause & exit)
echo ✅ Python3 available
echo.

echo Testing Streamlit in WSL...
wsl python3 -c "import streamlit; print('Streamlit version:', streamlit.__version__)" || (echo ❌ Streamlit not available & pause & exit)
echo ✅ Streamlit available
echo.

echo Testing project directory...
wsl ls /home/jcockburn/SC-Gen-4/app.py || (echo ❌ Project directory not found & pause & exit)
echo ✅ Project directory found
echo.

echo All tests passed! The main launcher should work.
echo.
echo Would you like to:
echo 1) Try the main launcher now
echo 2) Start Strategic Counsel directly (manual)
echo 3) Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo Starting main launcher...
    start Strategic_Counsel_Launcher.bat
) else if "%choice%"=="2" (
    echo Starting Strategic Counsel manually...
    echo You can close this window after the browser opens.
    wsl bash -c "cd /home/jcockburn/SC-Gen-4 && python3 -m streamlit run app.py --server.port 8501"
) else (
    echo Goodbye!
    pause
    exit
)

pause 