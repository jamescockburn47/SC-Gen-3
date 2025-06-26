Set WshShell = WScript.CreateObject("WScript.Shell")
Set oShellLink = WshShell.CreateShortcut(WshShell.SpecialFolders("Desktop") & "\Strategic Counsel.lnk")

oShellLink.TargetPath = "cmd"
oShellLink.Arguments = "/c ""wsl -d Ubuntu-22.04 bash -c 'cd /home/jcockburn/SC-Gen-3 && /home/jcockburn/.local/bin/streamlit run app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false'"""
oShellLink.WorkingDirectory = "C:\Users\James"
oShellLink.Description = "Strategic Counsel - AI Legal Analysis Platform"
oShellLink.WindowStyle = 1

oShellLink.Save

WScript.Echo "Desktop shortcut created successfully!"
WScript.Echo "You can now double-click 'Strategic Counsel' on your desktop to launch the app." 