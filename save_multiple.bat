@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: %0 csv_file prefix number1 number2 ...
    exit /b 1
)

set "csv_file=%~1"
set "prefix=%~2"
shift
shift

set "numbers=%*"
for %%i in (%numbers%) do (
    set "num=%%i"
    python rocket_evaluate.py "!csv_file!" !num! "!prefix!!num!.ork" skip
)
endlocal
