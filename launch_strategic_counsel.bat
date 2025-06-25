@echo off
echo ====================================================
echo    Strategic Counsel - AI Legal Analysis Platform
echo ====================================================
echo Starting application...

echo.
echo Checking if Streamlit is already running...

REM Check if port 8501 is in use
netstat -an | find "8501" >nul
if %errorlevel% == 0 (
    echo.
    echo Streamlit is already running on port 8501!
    echo You can access it at: http://localhost:8501
    echo.
    echo Options:
    echo 1. Open browser to existing app
    echo 2. Kill existing and restart
    echo 3. Start on different port
    echo.
    choice /c 123 /m "Choose option (1/2/3)"
    
    if errorlevel 3 goto :different_port
    if errorlevel 2 goto :kill_and_restart
    if errorlevel 1 goto :open_browser
) else (
    goto :start_app
)

:open_browser
echo Opening browser...
start http://localhost:8501
goto :end

:kill_and_restart
echo Killing existing Streamlit processes...
wsl pkill -f streamlit
timeout /t 3 /nobreak >nul
goto :start_app

:different_port
set /a port=8501+%random% %% 100
echo Starting on port %port%...
cd /d "%~dp0"
wsl ~/.local/bin/streamlit run app.py --server.port %port%
goto :end

:start_app
echo Starting Strategic Counsel...
cd /d "%~dp0"
wsl ~/.local/bin/streamlit run app.py --server.port 8501

:end
echo.
echo Application closed.
pause 