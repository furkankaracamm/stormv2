@echo off
cd /d "%~dp0"
set PYTHONPATH=%CD%
title STORM COMMANDER - ZERO-ERROR ARCHITECTURE
color 0A

echo ==================================================
echo STORM ZERO-ERROR STARTUP CHECK
echo ==================================================
echo.

REM Run database migration first
python db_migrate.py
if errorlevel 1 (
    echo [FATAL] Database migration failed!
    pause
    exit /b 1
)

echo.

REM Run startup dependency check
python startup_check.py
if errorlevel 1 (
    echo.
    echo [FATAL] Dependency check failed! Fix errors above.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo STORM SYSTEMS INITIALIZING...
echo [MODE] Autonomous Research
echo [TARGET] Dead Internet Theory
echo [MODULE] Librarian: ACTIVE (Semantic Sorting)
echo [MODULE] GROBID Full: ACTIVE (Authors, Refs, Keywords)
echo [MODULE] Deep Analysis: ACTIVE (Claims, Arguments)
echo [MODULE] LLM Gateway: ACTIVE (Groq Primary)
echo [MODULE] Gap Finder: ACTIVE (Every 10 cycles)
echo ==================================================

set STORM_ENABLE_GROBID_FULL=1
set STORM_ENABLE_PARS_CIT=1
set STORM_ENABLE_TABLE_EXTRACT=1
set STORM_ENABLE_FIGURE_EXTRACT=1
set STORM_ENABLE_DEEP_ANALYSIS=1
set STORM_ENABLE_LLM=1
set GROQ_API_KEY=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
set STORM_TELEGRAM_API_ID=30881934
set STORM_TELEGRAM_API_HASH=f21730701d0b1da80764c094c73effdb

echo.
python storm_commander.py
echo.
echo [SYSTEM STOPPED]
pause
