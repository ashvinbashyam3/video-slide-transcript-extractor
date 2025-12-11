@echo off
REM Video Slide & Transcript Extractor
REM Double-click this file to run the program
REM File dialog opens first (fast), then WSL processes the video

setlocal enabledelayedexpansion

REM Use PowerShell to show file dialog (native Windows, fast)
echo Select a video file...
for /f "delims=" %%i in ('powershell -Command "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.InitialDirectory = [Environment]::GetFolderPath('UserProfile') + '\Downloads'; $f.Filter = 'Video files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|All files (*.*)|*.*'; $f.Title = 'Select Video File'; if ($f.ShowDialog() -eq 'OK') { $f.FileName }"') do set "SELECTED_FILE=%%i"

REM Check if user cancelled
if "%SELECTED_FILE%"=="" (
    echo No file selected. Exiting.
    pause
    exit /b
)

echo Selected: %SELECTED_FILE%
echo.
echo Starting WSL processing...
echo.

REM Convert Windows path to WSL path
REM C:\Users\... becomes /mnt/c/Users/...
set "WSL_PATH=%SELECTED_FILE%"
set "WSL_PATH=%WSL_PATH:\=/%"
set "WSL_PATH=/mnt/%WSL_PATH:~0,1%%WSL_PATH:~2%"

REM Make drive letter lowercase
set "DRIVE_LETTER=%WSL_PATH:~5,1%"
for %%a in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    if /i "%DRIVE_LETTER%"=="%%a" set "WSL_PATH=/mnt/%%a%WSL_PATH:~6%"
)

REM Get the directory where this batch file is located
set "BATCH_DIR=%~dp0"
set "BATCH_DIR=%BATCH_DIR:\=/%"
set "WSL_BATCH_DIR=/mnt/%BATCH_DIR:~0,1%%BATCH_DIR:~2%"
set "DRIVE_LETTER2=%WSL_BATCH_DIR:~5,1%"
for %%a in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    if /i "%DRIVE_LETTER2%"=="%%a" set "WSL_BATCH_DIR=/mnt/%%a%WSL_BATCH_DIR:~6%"
)

REM Run the Python script in WSL
wsl bash -c "cd '%WSL_BATCH_DIR%' && python3 main.py '%WSL_PATH%'"

echo.
echo Processing complete!
pause
