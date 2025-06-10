@echo off
echo 🚀 Starting The Big Everything Prompt Library Web Interface...
echo 📍 Server will be available at: http://localhost:8000
echo 🔍 API documentation at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0backend"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause