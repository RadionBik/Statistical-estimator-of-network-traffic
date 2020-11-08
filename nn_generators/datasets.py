import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ManyToManyDataset(Dataset):
    def __init__(self, states: np.ndarray, window=200):
        self.window_size = window
        states_tensor = torch.as_tensor(
            states,
            dtype=torch.float,
        )
        # chunk vector into windows shifted by one position, autoregressive approach
        ar_states = states_tensor.unfold(0, self.window_size, 1)
        ar_states = ar_states

        self.x = ar_states[:-1]
        self.y = ar_states[1:].to(dtype=torch.long)
        self.len_init_states = len(states)
        logger.info(f'dataset: got {len(states)} states with {len(set(states))} unique')

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class ManyToOneDataset(Dataset):
    def __init__(self, states: np.ndarray, window=200):
        self.window_size = window
        states_tensor = torch.as_tensor(
            states,
            dtype=torch.float,
        )
        # chunk vector into windows shifted by one position, autoregressive approach
        ar_states = states_tensor.unfold(0, self.window_size + 1, 1)
        ar_states = ar_states

        self.x = ar_states[:, :-1]
        self.y = ar_states[:, -1].to(dtype=torch.long).squeeze(0)
        self.len_init_states = len(states)
        logger.info(f'dataset: got {len(states)} states with {len(set(states))} unique')

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
