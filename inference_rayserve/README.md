# Ray Serve

Recognition of russian car license plates using neural networks served by [Ray Serve](https://www.ray.io/ray-serve).

## Run

Create virtual environment

```bash
python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip pip-tools
pip-sync requirements.txt
```

## Tests

### Unit

```bash
pytest -vv --disable-warnings tests/test_models.py
```
