# BentoML

Recognition of russian car license plates using neural networks served by [BentoML](https://www.bentoml.com/).

## Run bentoml

Create virtual environment

```bash
python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip pip-tools
pip install -r requirements.txt
```

```bash
export BENTOML_HOME=./bento_store
bentoml build
```

### Locally

```bash
bentoml serve plate_recognition
```

### With docker

```bash
bentoml containerize --opt build-context=nn=$PWD/.. -t plate_recognition:latest plate_recognition
docker run --rm -p 3000:3000 --gpus all plate_recognition:latest
```

## Tests

### Unit

```bash
pytest -vv --disable-warnings tests/test_models.py
```

### Functional

Start the server. See Run bentoml.

```bash
pytest -vv --disable-warnings tests/test_service.py
```

## Prepare models

Models have already been added to this repo. There is no need to run this code to start the project.
The code below is just to explain how the artifacts were created.

Activate venv.

#### STN

```bash
python -m inference_bentoml.scripts.prepare_stn
```

#### LPRNET

```bash
python -m inference_bentoml.scripts.prepare_lprnet
```
