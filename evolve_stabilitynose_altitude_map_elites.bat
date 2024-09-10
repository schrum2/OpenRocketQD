@echo off
setlocal enabledelayedexpansion

REM Check if N is provided as a command line parameter
if "%~1"=="" (
    echo Please provide the value of N as an experiment number.
    exit /b 1
)

set N=%~1

for /L %%i in (0,1,%N%-1) do (
    python evolve_rockets.py map_elites %%i stabilitynose_altitude
)

endlocal
