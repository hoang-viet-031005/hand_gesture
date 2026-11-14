@echo off
setlocal
cd /d "%~dp0"
title App Camera Nhan Dien Cu Chi Tay (Model 1 - ASL)

echo ====================================================================
echo   KHOI DONG CAMERA NHAN DIEN CU CHI TAY (ASL)
echo ====================================================================
echo.
echo [1/2] Dang kiem tra moi truong Python...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python trong he thong!
    echo Vui long cai dat Python hoac them Python vao bien moi truong PATH.
    pause
    exit /b 1
)

echo [2/2] Dang nap mo hinh va mo Camera...
python app_camera.py

if %errorlevel% neq 0 (
    echo.
    echo ====================================================================
    echo [LOI] Co loi xay ra trong khi chay Camera!
    echo ====================================================================
    pause
)
