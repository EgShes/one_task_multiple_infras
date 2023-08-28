import os

import pytest

from inference_bentoml.settings import settings


@pytest.fixture(autouse=True)
def set_envs():
    os.environ["BENTOML_HOME"] = str(settings.BENTO_HOME)
