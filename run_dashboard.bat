@echo off
cd /d "%~dp0"
title STORM Dashboard
color 0B
echo ==================================================
echo STORM DASHBOARD LAUNCHING...
echo ==================================================
echo.
echo Dashboard URL: http://localhost:8501
echo.
python -m streamlit run storm_dashboard.py

