@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo Creating the PetroAnalyst Python environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        python -m venv .venv
    )
)

if not exist "%VENV_PYTHON%" (
    echo Python 3 was not found. Install Python 3.11 or newer, then run this file again.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -c "import fastapi, uvicorn, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing PetroAnalyst dependencies. This is required only on the first run...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    "%VENV_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check the network connection and requirements.txt.
        pause
        exit /b 1
    )
)

if not exist ".env" (
    if exist ".env.example" copy /y ".env.example" ".env" >nul
)

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8000/'"
"%VENV_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

endlocal
