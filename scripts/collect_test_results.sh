#!/bin/bash

# Read script arguments for environment to collect results for (ie: cw, kw, sw)
TESTED_ENV=$1
TEST_TYPE=$2

# Create temporary folder for storing results
# This is useful when the results/test folder already contains subfolders with the same name as the
# ones that would be imported when collecting the results
mkdir tmp_results

# Collect results
docker cp TestPerfect"${TESTED_ENV^^}"_"${TEST_TYPE}":/code/results/test/test_"${TESTED_ENV}"_"${TEST_TYPE}" tmp_results/

# Move everything from temporary folder to actual results folder
mv tmp_results/* results/test/

# Delete temporary folder
rmdir tmp_results
