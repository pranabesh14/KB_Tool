@echo off
setlocal EnableDelayedExpansion

:: =============================================================================
:: deploy.bat — KB Tool setup and launch script for Windows
:: Usage: Double-click or run from Command Prompt / PowerShell
:: =============================================================================

title KB Tool — Deploying...
echo.
echo ============================================================
echo   KB Tool — Setup ^& Launch
echo ============================================================
echo.

:: ── 0. Check Python is available ─────────────────────────────────────────────
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo        Found: %%v
echo.

:: ── 1. Check / create virtual environment ────────────────────────────────────
echo [2/5] Setting up virtual environment...
if not exist ".venv\" (
    echo        Creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo        Virtual environment created.
) else (
    echo        .venv already exists, skipping creation.
)
echo.

:: ── 2. Activate venv and install requirements ────────────────────────────────
echo [3/5] Installing requirements...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install requirements. Check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo        All packages installed successfully.
echo.

:: ── 3. Create required directories ───────────────────────────────────────────
echo [4/5] Creating required directories...
set DIRS=data faiss_pdf faiss_video qa_indexes modules

for %%d in (%DIRS%) do (
    if not exist "%%d\" (
        mkdir "%%d"
        echo        Created: %%d\
    ) else (
        echo        Already exists: %%d\
    )
)
echo.

:: ── 4. Check .env file ───────────────────────────────────────────────────────
echo [5/5] Checking environment configuration...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [WARN] .env not found — copied from .env.example.
        echo        Please edit .env and fill in your Azure OpenAI credentials before using the app.
    ) else (
        echo [WARN] Neither .env nor .env.example found.
        echo        Create a .env file with your Azure OpenAI credentials.
    )
) else (
    echo        .env found.
)
echo.

:: ── 5. Launch Streamlit app ───────────────────────────────────────────────────
echo ============================================================
echo   Launching KB Tool...
echo   Open your browser at: http://localhost:8501
echo   Press Ctrl+C in this window to stop the app.
echo ============================================================
echo.

streamlit run app.py --server.port 8501 --server.headless false

:: If streamlit exits, pause so the user can read any error messages
if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit exited with an error. See output above.
    pause
)

endlocal