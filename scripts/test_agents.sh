#!/bin/bash

echo "Testing environment: ${TESTING_ENV}"

# Baseline tests
python src/testing.py -t baseline \
                      -n 5000 \
                      -m 500 \
                      -a perfect_${TESTING_ENV} \
                      -s test_${TESTING_ENV}_baseline

# Random LF noise tests
python src/testing.py -t randomlf \
                      -n 5000 \
                      -m 500 \
                      -a perfect_${TESTING_ENV} \
                      -s test_${TESTING_ENV}_randomlf

# Event Blinding tests
python src/testing.py -t evt-blind \
                      -n 5000 \
                      -m 500 \
                      -a perfect_${TESTING_ENV} \
                      -s test_${TESTING_ENV}_evt-blind \
                      --n_strategies 5 \
                      --traces_from test_${TESTING_ENV}_baseline

# Edge Blinding tests
python src/testing.py -t edg-blind \
                      -n 5000 \
                      -m 500 \
                      -a perfect_${TESTING_ENV} \
                      -s test_${TESTING_ENV}_edg-blind \
                      --n_strategies 5 \
                      --traces_from test_${TESTING_ENV}_baseline
