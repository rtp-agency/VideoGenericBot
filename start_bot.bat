@echo off
echo ========================================
echo  Telegram Video Generation Bot
echo ========================================
echo.

REM Check if environment variables are set
if "%BOT_TOKEN%"=="" (
    echo ERROR: BOT_TOKEN environment variable is not set!
    echo Please set it first:
    echo   set BOT_TOKEN=your_telegram_bot_token
    echo.
    pause
    exit /b 1
)

if "%FAL_KEY%"=="" (
    echo ERROR: FAL_KEY environment variable is not set!
    echo Please set it first:
    echo   set FAL_KEY=your_fal_api_key
    echo.
    pause
    exit /b 1
)

echo Starting bot...
echo.
python bot.py

pause
