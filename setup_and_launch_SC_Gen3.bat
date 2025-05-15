@echo off
echo === Strategic Counsel Gen 3: Environment Setup and Launch ===
cd /d %~dp0
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Installing requirements...
pip install -r requirements.txt
echo Launching Strategic Counsel...
streamlit run app.py
pause
