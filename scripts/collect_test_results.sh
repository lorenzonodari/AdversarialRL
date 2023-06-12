#!/bin/bash

# Read script arguments for environment to collect results for (ie: cw, kw, sw)
TESTED_ENV=$1

# Create temporary folder for storing results
# This is useful when the results/test folder already contains subfolders with the same name as the
# ones that would be imported when collecting the results
mkdir tmp_results

# Collect baseline results
docker cp TestPerfect"${TESTED_ENV^^}":/code/results/test/test_"${TESTED_ENV}"_baseline tmp_results/

# Collect Random Noise results
docker cp TestPerfect"${TESTED_ENV^^}":/code/results/test/test_"${TESTED_ENV}"_randomlf tmp_results/

# Collect Event Blinding results
docker cp TestPerfect"${TESTED_ENV^^}":/code/results/test/test_"${TESTED_ENV}"_evt-blind tmp_results/

# Collect Edge Blinding results
docker cp TestPerfect"${TESTED_ENV^^}":/code/results/test/test_"${TESTED_ENV}"_edg-blind tmp_results/

# Move everything from temporary folder to actual results folder
mv tmp_results/* results/test/

# Delete temporary folder
rmdir tmp_results
