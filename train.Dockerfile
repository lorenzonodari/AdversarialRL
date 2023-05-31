# syntax=docker/dockerfile:1
FROM rayproject/ray-ml:2.3.0-py310-gpu

# Build-time arguments
ARG n_runs=10
ARG session_name
ARG env
ARG config

# Make sure protobuf <= 3.20
RUN pip install protobuf==3.20

# Copy code and cd into directory
COPY --chown=ray:users . /code
WORKDIR /code

# Execute interactive shell
CMD python src/training.py -e ${env} -n ${n_runs} -c config/${config} -s ${session_name}