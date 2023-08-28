# BentoML

Recognition of russian car license plates using neural networks served by [BentoML](https://www.bentoml.com/).

# Prod

...


# Dev

Install virtual environment

```bash
python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip pip-tools
pip install -r requirements.txt
```

## Run bentoml

```bash
export BENTOML_HOME=./bento_store
bentoml build
```

Locally

```bash
bentoml serve plate_recognition
```

With docker

```bash
bentoml containerize --opt build-context=nn=$PWD/.. plate_recognition
```

## Tests

```bash
pytest -vv --disable-warnings tests
```

## Prepare models

Models have already been added to this repo. There is no need to run this code to start the project.
The code below is just to explain how the artifacts were created.

#### STN

```bash
python -m inference_bentoml.scripts.prepare_stn
```

#### LPRNET

```bash
python -m inference_bentoml.scripts.prepare_lprnet
```
