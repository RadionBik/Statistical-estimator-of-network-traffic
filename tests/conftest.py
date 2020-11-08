import json
import pickle
from collections import defaultdict

import numpy as np
import pytest

import settings
from features.gaussian_quantizer import GaussianQuantizer
from pcap_parsing.device_level import extract_host_stats

STATIC_DIR = settings.BASE_DIR / 'tests' / 'static'


def load_json_states(name):
    with open(STATIC_DIR / name, 'r') as jf:
        return np.array(json.load(jf))


@pytest.fixture()
def gmm_model():
    with open(STATIC_DIR/'skype_gmm.pkl', 'rb') as f:
        model_dict = pickle.load(f)
        item = next(iter(model_dict.values()))
        return item['from']


@pytest.fixture
def gmm_state():
    return load_json_states('gmm_skype_from.json')


@pytest.fixture
def gmm_states():
    states = defaultdict(dict)
    states['UDP 192.168.0.102:18826 192.168.0.105:26454'] = {'from': load_json_states('gmm_skype_from.json'),
                                                             'to': load_json_states('gmm_skype_to.json')}
    return states


@pytest.fixture
def pcap_file():
    return STATIC_DIR / 'rtp_711.pcap'


@pytest.fixture
def raw_host_stats(pcap_file):
    return extract_host_stats(pcap_file, '00:04:76:22:20:17')


@pytest.fixture
def gaussian_quantizer():
    return GaussianQuantizer.from_pretrained(STATIC_DIR / 'amazon')
