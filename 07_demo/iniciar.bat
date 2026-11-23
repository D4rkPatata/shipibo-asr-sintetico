@echo off
rem Abre la demo en http://127.0.0.1:8000 (la primera vez descarga MMS-1b-all, ~4 GB, y tarda unos minutos)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
start "" cmd /c "timeout /t 90 >nul & start http://127.0.0.1:8000"
.venv\Scripts\python demo\app.py
