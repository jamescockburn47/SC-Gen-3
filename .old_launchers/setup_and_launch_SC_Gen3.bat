@echo off
setlocal enabledelayedexpansion

echo === Strategic Counsel Gen 3: Environment Setup and Launch ===

REM Change directory to the script's location
cd /d %~dp0
echo Current directory: %cd%

REM --- Environment Check ---
if not exist ".env" (
    echo WARNING: .env file not found in current directory.
    echo Please ensure your .env file is properly configured.
    echo Press any key to continue anyway or Ctrl+C to abort...
    pause > nul
)

REM --- Python Version Check ---
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM --- Virtual Environment Setup ---
echo Ensuring virtual environment 'venv' exists...
if exist "venv" (
    echo Found existing virtual environment.
) else (
    echo Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Please ensure Python is installed and configured correctly in your PATH.
        pause
        exit /b 1
    )
)

REM --- Activate Virtual Environment ---
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment appears to be corrupted.
    echo Please delete the 'venv' folder and run this script again.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM --- Requirements Installation ---
echo Checking and installing requirements...
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

echo Installing requirements from requirements.txt...
python -m pip install -r requirements.txt --no-cache-dir
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    echo Please check your network connection and requirements.txt file.
    pause
    exit /b 1
)

REM --- Verify Critical Dependencies ---
echo Verifying critical dependencies...
python -c "import PyPDF2; print('PyPDF2 version:', PyPDF2.__version__)"
if errorlevel 1 (
    echo ERROR: PyPDF2 installation failed or is not accessible.
    echo Please try running: pip install PyPDF2==3.0.1
    pause
    exit /b 1
)

REM --- Create Required Directories ---
if not exist "scratch" mkdir scratch
if not exist "logs" mkdir logs
if not exist "exports" mkdir exports
if not exist "summaries" mkdir summaries
if not exist "memory" mkdir memory

REM --- Launch Application ---
echo.
echo === Launching Strategic Counsel ===
echo Press Ctrl+C to stop the application.
echo.
streamlit run app.py
if errorlevel 1 (
    echo.
    echo ERROR: Application terminated with an error.
    echo Please check the logs for more information.
    pause
    exit /b 1
)

echo.
echo Strategic Counsel has been closed.
pause
