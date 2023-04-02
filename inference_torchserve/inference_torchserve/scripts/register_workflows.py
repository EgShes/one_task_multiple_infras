import logging
import sys
from time import sleep
from typing import List

import requests
from tap import Tap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


class ArgumentParser(Tap):
    workflow_names: List[str]


def wait_server_loaded():
    for _ in range(60):
        try:
            response = requests.get("http://localhost:8080/ping")
            if response.status_code == 200:
                logger.info("Server is ok.")
                return
        except Exception:
            pass

        logger.info("Server not healthy. Retrying in 1 sec.")
        sleep(1)

    logger.info("Server not healthy.")
    sys.exit(1)


def register_workflows(workflows: List[str]):
    url = "http://localhost:8081/workflows?url={workflow}"

    for workflow in workflows:
        requests.post(url.format(workflow=workflow))
        logger.info(f"Registered workflow {workflow}")


if __name__ == "__main__":
    args = ArgumentParser().parse_args()

    wait_server_loaded()
    register_workflows(args.workflow_names)
