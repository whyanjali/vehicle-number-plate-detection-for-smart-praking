@echo off
title Smart Parking Assistant - Web Dashboard
cd /d "%~dp0"
echo ===============================================================
echo   Starting Smart Parking Assistant Web Dashboard...
echo   Open http://127.0.0.1:5000 in your web browser
echo ===============================================================
yolov8_env\Scripts\python.exe app.py
pause
