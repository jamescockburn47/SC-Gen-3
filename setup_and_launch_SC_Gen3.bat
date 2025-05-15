@echo off
echo === Strategic Counsel Gen 3: Environment Setup and Launch ===

REM Change directory to the script's location
cd /d %~dp0
echo Current directory: %cd%

REM --- Virtual Environment Setup ---
echo Ensuring virtual environment 'venv' exists...
REM Always try to run the venv command. If venv exists and is healthy, it's a quick no-op.
REM If it's missing or corrupted, this will attempt to create/repair it.
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create or ensure virtual environment using 'python -m venv venv'.
    echo Please ensure Python is installed and configured correctly in your PATH.
    echo You might also try deleting the 'venv' folder manually and running this script again.
    pause
    exit /b 1
)
echo Virtual environment setup/check complete.

REM Check if activate.bat exists *after* the python command
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: 'venv\Scripts\activate.bat' was NOT found even after 'python -m venv venv' command.
    echo This indicates a problem with Python's venv creation on your system when called from a batch file.
    echo Please check your Python installation.
    pause
    exit /b 1
)
echo 'venv\Scripts\activate.bat' found.

echo Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate the virtual environment using 'call venv\Scripts\activate.bat'.
    echo The activation script might have encountered an error.
    pause
    exit /b 1
)
echo Virtual environment activated.

REM --- Requirements Installation ---
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements using 'pip install -r requirements.txt'.
    echo Check your requirements.txt file, network connection, and that the venv is active.
    pause
    exit /b 1
)
echo Requirements installed.

REM --- Launch Application ---
echo Launching Strategic Counsel...
streamlit run app.py
if errorlevel 1 (
    echo ERROR: Failed to launch Streamlit application 'app.py'.
    pause
    exit /b 1
)

echo Strategic Counsel finished or closed by user.
pause
