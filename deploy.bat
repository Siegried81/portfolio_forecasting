@echo off
REM Build and run the Portfolio Forecasting Docker deployment (Windows)

setlocal enabledelayedexpansion

set PROJECT_NAME=portfolio-forecasting
set IMAGE_NAME=%PROJECT_NAME%
set CONTAINER_NAME=%PROJECT_NAME%
set PORT=8501

echo.
echo === Building Docker image for %PROJECT_NAME% ===
echo This may take 3-5 minutes due to large dependencies (scipy, cvxpy, statsmodels)...
echo.

docker build -t %IMAGE_NAME% .

if errorlevel 1 (
    echo.
    echo ERROR: Build failed. Check Docker Desktop is running and has enough disk space.
    pause
    exit /b 1
)

echo.
echo === Build complete ===
docker images | findstr %IMAGE_NAME%

echo.
echo === Running container ===
echo Container will be accessible at: http://localhost:%PORT%
echo.

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found!
    echo   Copy .env.example to .env and add your API keys:
    echo   - GROQ_API_KEY (for AI Analyst tab)
    echo   - NEWSAPI_KEY (for news digest)
    echo   - FRED_API_KEY (for live Treasury yields)
    echo   - TWELVEDATA_API_KEY (for market data fallback)
    echo.
    echo   The app will run without these keys, but features will be limited.
    echo.
)

echo Starting with docker compose...
docker compose up -d --pull always

echo.
echo === Container is starting ===
echo Check status:
echo   docker ps ^| findstr %CONTAINER_NAME%
echo.
echo View logs:
echo   docker compose logs -f
echo.
echo Stop:
echo   docker compose down
echo.

pause
