@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%
python test_system_brain.py
pause
