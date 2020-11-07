import os
import logging
import sys
import pathlib
import numpy as np

np.random.seed(1)


BASE_DIR = pathlib.Path(os.path.dirname(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s:%(lineno)d %(levelname)s - %(message)s')
logging.getLogger('neptune.internal.channels.channels_values_sender').setLevel(logging.ERROR)

NEPTUNE_PROJECT = 'radion/TCN'
NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
