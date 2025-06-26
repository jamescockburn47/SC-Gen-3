Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get user's desktop path
DesktopPath = WshShell.SpecialFolders("Desktop")

' Create shortcut
Set oShellLink = WshShell.CreateShortcut(DesktopPath & "\Strategic Counsel.lnk")
oShellLink.TargetPath = DesktopPath & "\Strategic Counsel.bat"
oShellLink.WorkingDirectory = DesktopPath
oShellLink.Description = "Strategic Counsel - Multi-Agent AI Legal Analysis Platform"

' Set icon if the PNG file exists (Windows will convert it automatically)
IconPath = DesktopPath & "\strategic_counsel_logo.png"
If fso.FileExists(IconPath) Then
    oShellLink.IconLocation = IconPath
End If

oShellLink.Save

WScript.Echo "Strategic Counsel shortcut created successfully!"
WScript.Echo "Location: " & DesktopPath & "\Strategic Counsel.lnk" 