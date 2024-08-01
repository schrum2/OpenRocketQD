@echo off
setlocal enabledelayedexpansion

set numbers=5210 5175 5534 5402 5570 5571 5628 6077 5892 6598 5908 5322
for %%i in (%numbers%) do (
    set num=%%i
    python rocket_evaluate.py .\evolve_rockets_output\cma_me_imp_stabilitynose_altitude_2_archive.csv !num! cma_mae0_!num!.ork skip
)
endlocal
