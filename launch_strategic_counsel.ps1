Write-Host "====================================================" -ForegroundColor Green
Write-Host "   Strategic Counsel - AI Legal Analysis Platform" -ForegroundColor Green  
Write-Host "====================================================" -ForegroundColor Green
Write-Host "Starting application..." -ForegroundColor Yellow
Write-Host ""

try {
    # Launch the app in WSL with direct path
    Write-Host "Launching Strategic Counsel in WSL..." -ForegroundColor Cyan
    wsl -d Ubuntu-22.04 bash -c "cd /home/jcockburn/SC-Gen-3 && /home/jcockburn/.local/bin/streamlit run app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false"
}
catch {
    Write-Host "Error launching application: $_" -ForegroundColor Red
    Write-Host "Make sure WSL and Ubuntu-22.04 are properly installed." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Application closed." -ForegroundColor Yellow
Read-Host "Press Enter to exit" 