@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Please provide the value of N as starting experiment number.
    exit /b 1
)

if "%~2"=="" (
    echo Please provide the value of M as ending experiment number.
    exit /b 1
)

set N=%~1
set M=%~2

for /L %%i in (%N%,1,%M%) do (
    python evolve_rockets.py map_elites %%i stabilitynose_altitude
)

endlocal
