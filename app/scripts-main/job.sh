#!/bin/sh
startall=$(date +%s)

cd scripts-main

# run processing workflow:
echo ----
echo 1. STALTA triggering
python3 stalta_trigger.py
echo ----
echo 2.merging events
python3 merge_events.py
echo ----
echo 3.plotting events
# python3 plot_events_from_catalogue.py
echo plotting turned off
echo ----
echo 4.extracting velo features
python3 extract_features_velo.py
echo ----
echo 5.extracting covmat features
python3 extract_features_covmat.py
echo ---

endall=$(date +%s)
echo "Elapsed Time: $(($endall-$startall)) seconds"
