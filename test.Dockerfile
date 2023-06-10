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
COPY --chown=ray:users . /code
WORKDIR /code

COPY --chown=ray:users ./agents /code/agents

# Execute testing script
CMD bash scripts/test_agents.sh