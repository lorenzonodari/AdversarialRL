# syntax=docker/dockerfile:1
FROM rayproject/ray-ml:2.3.0-py310-gpu

# Build-time arguments
ARG prefix

# Propagate build-time arguments to environment variables to make sure that the container
# can access their value at runtime.
ENV AGENT_PREFIX=${prefix}

# Make sure protobuf <= 3.20
RUN pip install protobuf==3.20

# Copy code and cd into directory
COPY --chown=ray:users ./config /code/config
COPY --chown=ray:users ./maps /code/maps
COPY --chown=ray:users ./src /code/src
COPY --chown=ray:users ./agents/ /code/agents
COPY --chown=ray:users ./results/test /code/results/test

WORKDIR /code

# Execute testing script
CMD python src/testing.py -t evt-blind -n 1000 -m 500 -a ${AGENT_PREFIX} -s test_${AGENT_PREFIX}_evt-blind --traces_from test_${AGENT_PREFIX}_baseline