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

## Tests

```bash
pytest -vv --disable-warnings tests
```

## Prepare models

Models have already been added to this repo. There is no need to run this code to start the project.
The code below is just to explain how the artifacts were created.

```bash
export BENTOML_HOME=./bento_store
```

#### STN

```bash
python -m inference_bentoml.scripts.prepare_stn
```
