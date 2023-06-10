#!/bin/bash

# Cleanup previous images
docker image rm lrm:train_perfect_cw
docker image rm lrm:train_perfect_kw
docker image rm lrm:train_perfect_sw

# Build image for Perfect RM training in CookieWorld
docker build -f train.Dockerfile \
	     --build-arg scenario=CW \
	     --build-arg n_runs=10 \
	     --build-arg session_name=perfect_cw \
	     --build-arg config=cw_perfect.conf \
	     -t lrm:train_perfect_cw \
	     --no-cache \
	     .

# Build image for Perfect RM training in KeysWorld
docker build -f train.Dockerfile \
	     --build-arg scenario=KW \
	     --build-arg n_runs=10 \
	     --build-arg session_name=perfect_kw \
	     --build-arg config=kw_perfect.conf \
	     -t lrm:train_perfect_kw \
	     --no-cache \
	     .

# Build image for Perfect RM training in SymbolWorld
docker build -f train.Dockerfile \
	     --build-arg scenario=SW \
	     --build-arg n_runs=10 \
	     --build-arg session_name=perfect_sw \
	     --build-arg config=sw_perfect.conf \
	     -t lrm:train_perfect_sw \
	     --no-cache \
	     .
