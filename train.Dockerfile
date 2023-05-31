# syntax=docker/dockerfile:1
FROM rayproject/ray-ml:2.3.0-py310-gpu

# Build-time arguments
ARG n_runs=10
ARG session_name
ARG scenario
ARG config

# Propagate build-time arguments to environment variables to make sure that the container
# can access their value at runtime.
ENV TRAINING_N_RUNS=$n_runs
ENV TRAINING_SESSION_NAME=$session_name
ENV TRAINING_SCENARIO=$scenario
ENV TRAINING_CONFIG=$config

# Make sure protobuf <= 3.20
RUN pip install protobuf==3.20

# Copy code and cd into directory
COPY --chown=ray:users . /code
WORKDIR /code

# Execute interactive shell
CMD ["python", "src/training.py", "-e", "$TRAINING_SCENARIO", "-n", "$TRAINING_N_RUNS", "-c", "config/$TRAINING_CONFIG", "-s", "$TRAINING_SESSION_NAME" ]