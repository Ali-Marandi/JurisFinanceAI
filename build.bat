@echo off
echo =========================================
echo   JurisFinanceAI - Build Script (Windows)
echo =========================================
echo.

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.12+
    pause
    exit /b 1
)

echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Building application with PyInstaller...
pyinstaller build.spec --noconfirm
if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo [3/3] Build complete!
echo Output: dist\JurisFinanceAI\JurisFinanceAI.exe
echo.

REM Open the output folder
explorer dist\JurisFinanceAI

pause