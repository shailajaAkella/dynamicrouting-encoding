# hash:sha256:3c65d25e2ce7992bc86b53bed201077952f2df635f204ac506584aaa3d873639
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    code-ocean-aux-tools \
    matplotlib==3.9.2 \
    numba==0.60.0 \
    pandas[all]==2.2.3 \
    polars==1.17.1 \
    pynwb==2.8.2 \
    s3fs==2024.10.0 \
    scikit-learn==1.5.2 \
    universal-pathlib==0.1.4 \
    zarr==2.18.0
