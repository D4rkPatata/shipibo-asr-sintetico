@echo off
rem Crea el entorno de la demo (una sola vez). Requiere uv: https://docs.astral.sh/uv/
cd /d "%~dp0"
uv venv .venv --python 3.12 || goto :error
uv pip install --python .venv torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu || goto :error
uv pip install --python .venv -r requirements.txt || goto :error
echo.
echo Entorno listo. Para abrir la demo: iniciar.bat
goto :eof
:error
echo Fallo la instalacion.
exit /b 1
