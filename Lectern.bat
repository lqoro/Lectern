@echo off
rem Windows launcher (experimental). First run creates a private Python
rem environment and installs the dependencies, then starts Lectern.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv || goto :nopython
  .venv\Scripts\python -m pip install -r requirements.txt
)
start "" http://127.0.0.1:8477/
.venv\Scripts\python -m lectern
goto :eof
:nopython
echo Python 3.10 or newer is required: https://www.python.org/downloads/
pause
