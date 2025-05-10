# The tag is either cpu or cuda
ARG TAG=cpu

FROM ghcr.io/katherlab/toolarena:${TAG}

# Use bash (so we can use the "." command)
SHELL ["/bin/bash", "-c"]

COPY install.sh .env ./

# Run the install script (using environment variables)
# We prefix the command with ">>>START INSTALL<<<" and ">>>END INSTALL<<<" to make it easier to extract the output of the install script from the logs
RUN echo ">>>START INSTALL<<<" && \
    chmod +x install.sh && \
    set -o allexport && \
    . .env && \
    set +o allexport && \
    ./install.sh
RUN echo ">>>END INSTALL<<<"

COPY task.yaml implementation.py ./
