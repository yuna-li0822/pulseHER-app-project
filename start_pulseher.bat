@echo off
echo 🌸 Starting PulseHER - Unified Server Edition 🌸
echo.

REM Change to the correct directory
cd /d "C:\Users\yueyu\OneDrive\CardiIQ"

echo 📁 Starting from: %cd%
echo.

REM Start the unified Flask backend and frontend server
echo 🚀 Starting PulseHER Unified Server on http://localhost:5000
start "PulseHER Server" cmd /c "cd backend && ..\.venv\Scripts\python.exe app.py"
timeout /t 5 /nobreak >nul

REM --- DEPRECATED ---
REM REM Start HTTP file server
REM echo 🌐 Starting HTTP file server...
REM start "PulseHER Web Server" cmd /c "python -m http.server 3000"
REM timeout /t 3 /nobreak >nul

REM Open the web app in the default browser's private/incognito mode to bypass cache
echo 💖 Opening PulseHER web app in a clean browser session...
echo    (This helps avoid caching issues)
timeout /t 2 /nobreak >nul
start "" "msedge" --inprivate "http://localhost:5000/"

echo.
echo ✨ PulseHER is now running! ✨
echo 🌸 Web App: http://localhost:5000
echo 🔧 Backend API is also at: http://localhost:5000
echo.
echo This window will close automatically in 30 seconds...
timeout /t 30 /nobreak >nul
exit