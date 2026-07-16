@echo off
REM ============================================================
REM  @file        run.bat
REM  @author      ParisNeo
REM  @description OpenImageGen - Windows launcher.
REM               Activates the virtual environment (if found) and starts the server.
REM               Usage: run.bat [--host HOST] [--port PORT] [--config PATH]
REM ============================================================
setlocal

REM --- Activate virtualenv if one exists in the project ---
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM --- Prefer the installed console script, fall back to module ---
where openimagegen >nul 2>nul
if %errorlevel%==0 (
    openimagegen %*
) else (
    python -m openimagegen.cli %*
)

endlocal