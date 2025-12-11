@echo off
REM Video Slide & Transcript Extractor
REM Double-click this file to run the program
REM A file dialog will open to select a video file from your Downloads folder

cd /d "%~dp0"
python main.py
pause
