# syntax=docker/dockerfile:1
FROM rayproject/ray-ml:2.3.0-py310-gpu

# Build-time arguments
ARG env

# Propagate build-time arguments to environment variables to make sure that the container
# can access their value at runtime.
ENV TESTING_ENV=${env}

# Make sure protobuf <= 3.20
RUN pip install protobuf==3.20

# Copy code and cd into directory
COPY --chown=ray:users ./config /code/config
COPY --chown=ray:users ./maps /code/maps
COPY --chown=ray:users ./src /code/src
COPY --chown=ray:users ./agents/ /code/agents

WORKDIR /code

# Execute testing script
CMD python src/testing.py -t rand-blind -n 1000 -m 500 -a perfect_${TESTING_ENV} -s test_${TESTING_ENV}_rand-blind