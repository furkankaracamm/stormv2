@echo off
color 0C
echo [STORM EMERGENCY SHUTDOWN]
echo ==========================
echo Stopping Python Processes...
taskkill /F /IM python.exe /T
echo.
echo Stopping Java Processes (Tabula-py)...
taskkill /F /IM java.exe /T
echo.
echo All STORM processes (and other Python scripts) have been terminated.
pause
