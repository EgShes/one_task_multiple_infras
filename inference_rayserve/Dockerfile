# syntax=docker/dockerfile:1
FROM python:3.8

ENV PYTHONBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6

COPY requirements.txt requirements.txt
COPY --from=nn . /nn

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pip==23.2.1 setuptools==68.2.0 \
    && pip install -r requirements.txt

COPY ./inference_rayserve /app/inference_rayserve

CMD [ "serve", "run", "--port", "8000", "--host", "0.0.0.0", "inference_rayserve.service:app" ]
