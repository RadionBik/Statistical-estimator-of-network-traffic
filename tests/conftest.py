from collections import defaultdict
import json

import numpy as np
import pytest

import pcap_parser as estimator
import settings
import utils

STATIC_DIR = settings.BASE_DIR / 'tests' / 'static'


def load_json_states(name):
    with open(STATIC_DIR / name, 'r') as jf:
        return np.array(json.load(jf))


@pytest.fixture
def gmm_state():
    return load_json_states('gmm_skype_from.json')


@pytest.fixture
def traffic_dict():
    pcapfile = f'{settings.BASE_DIR}/traffic_dumps/skypeLANhome.pcap'
    return estimator.get_traffic_features(pcapfile,
                                          type_of_identifier=utils.TrafficObjects.FLOW,
                                          percentiles=(1, 99),
                                          min_samples_to_estimate=100)[0]


@pytest.fixture
def gmm_states():
    states = defaultdict(dict)
    states['UDP 192.168.0.102:18826 192.168.0.105:26454'] = {'from': load_json_states('gmm_skype_from.json'),
                                                             'to': load_json_states('gmm_skype_to.json')}
    return states

