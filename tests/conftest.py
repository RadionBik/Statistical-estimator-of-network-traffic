import pytest
import stat_estimator as estimator
import settings
from collections import defaultdict
import numpy as np


@pytest.fixture
def traffic_dict():
    pcapfile = f'{settings.BASE_DIR}/traffic_dumps/skypeLANhome.pcap'
    return estimator.getTrafficFeatures(pcapfile,
                                        typeIdent='flow',
                                        fileIdent='all',
                                        percentiles=(1, 99),
                                        min_samples_to_estimate=100)[0]


@pytest.fixture
def gmm_states():
    states = defaultdict(dict)
    states['UDP 192.168.0.102:18826 192.168.0.105:26454'] = {'from': np.random.randint(0, 16, 3547),
                                                             'to': np.random.randint(0, 16, 3625)}
    return states

