@echo off
setlocal
cd /d "%~dp0"
title Cai dat thu vien cho App Camera (Model 1 - ASL)

echo ====================================================================
echo   CAI DAT THU VIEN CHO APP CAMERA (MODEL 1 - ASL)
echo ====================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [LOI] May tinh nay chua cai dat Python!
    echo Vui long truy cap https://www.python.org/downloads/ tai Python.
    echo *QUAN TRONG*: Khi cai, nho tich chon "Add Python to PATH"!
    echo.
    pause
    exit /b 1
)

echo [1/2] Dang kiem tra va nang cap pip...
python -m pip install --upgrade pip

echo.
echo [2/2] Dang cai dat cac thu vien (torch, opencv, mediapipe, pyyaml)...
python -m pip install torch torchvision opencv-python mediapipe pyyaml numpy

if %errorlevel% neq 0 (
    echo.
    echo ====================================================================
    echo [LOI] Cai dat that bai. Vui long kiem tra ket noi Internet va thu lai!
    echo ====================================================================
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo   THANH CONG! May tinh nay da san sang chay App Camera.
echo   Ban chi can nhap dup vao 'run_camera.bat' de mo camera test!
echo ====================================================================
echo.
pause
