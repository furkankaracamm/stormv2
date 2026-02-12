@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%
python -m storm_modules.theory_builder
pause
