# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update \
    && apt-get install -y \
        software-properties-common \
        ffmpeg \
        libsm6 \
        libxext6 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y  \
        python3.8  \
        python3.8-dev  \
        python3.8-distutils \
        python3-pip

RUN ln -s /usr/bin/python3.8 /usr/bin/python

COPY requirements.txt requirements.txt
COPY --from=nn . /nn

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pip==23.2.1 setuptools==68.2.0 \
    && python -m pip install -r requirements.txt

COPY ./inference_rayserve /app/inference_rayserve

CMD [ "serve", "run", "--port", "8000", "--host", "0.0.0.0", "inference_rayserve.service:app" ]
