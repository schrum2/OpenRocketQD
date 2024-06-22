
REM need to find way to generalize this

$numbers = @(7874, 7844, 7865, 7832, 6297, 7853, 7839, 7867, 7830); 
foreach ($num in $numbers) { python rocket_evaluate.py .\evolve_rockets_output\cma_mae0_archive.csv $num cma_mae0_$num.ork skip; }