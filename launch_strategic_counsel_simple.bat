@echo off
title Strategic Counsel Launcher
echo ====================================================
echo    Strategic Counsel - AI Legal Analysis Platform  
echo ====================================================
echo.
echo Starting application...
echo.
echo 📡 Server will be available at: http://localhost:8501
echo 🌐 Your browser should open automatically
echo.
echo To stop the application, press Ctrl+C in this window
echo ====================================================
echo.

wsl -d Ubuntu-22.04 bash -c "cd /home/jcockburn/SC-Gen-3 && /home/jcockburn/.local/bin/streamlit run app.py --server.headless false --server.port 8501" 