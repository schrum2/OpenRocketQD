REM assumes 30 runs of each algorithm have completed
python process_saved_archives.py -r 0 29 -p evolve_rockets_output/map_elites_stabilitynose_altitude -o map_elites_stabilitynose_altitude_0to29_megaarchive.pdf
python process_saved_archives.py -r 0 29 -p evolve_rockets_output/cma_me_imp_stabilitynose_altitude -o cma_me_imp_stabilitynose_altitude_0to29_megaarchive.pdf
python process_saved_archives.py -r 0 29 -p evolve_rockets_output/cma_mae_stabilitynose_altitude -o cma_mae_stabilitynose_altitude_0to29_megaarchive.pdf