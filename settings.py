import os
import logging
import sys
import pathlib
import numpy as np

np.random.seed(1)


BASE_DIR = pathlib.Path(os.path.dirname(__file__))

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
