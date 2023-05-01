# syntax=docker/dockerfile:1
FROM rayproject/ray-ml:2.3.0-py310-gpu

# Install POPGym
RUN pip install "popgym[baselines]"

# Copy code and cd into directory
COPY --chown=ray:users . /code
WORKDIR /code

# Execute interactive shell
CMD python src/training.py