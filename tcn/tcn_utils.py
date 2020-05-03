import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class StatesDataset(Dataset):
    def __init__(self, states: np.ndarray, window=200, device='cpu'):
        self.window_size = window
        states_tensor = torch.as_tensor(states,
                                        dtype=torch.float,
                                        device=device,
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


def generate_states(model, dataset, sample_number, window_size, shuffle=False, prepend_init_states=True, device='cpu'):
    def _get_next_prediction():
        with torch.no_grad():
            out = model(input_seq.unsqueeze(1))
            max_out = out.squeeze(0).max(1, keepdim=True)[1]
            return max_out[-1].unsqueeze(1)

    model.eval()

    input_seq, _ = next(iter(DataLoader(dataset, batch_size=1, drop_last=True, shuffle=shuffle)))

    init_index = 0
    generated_samples = torch.zeros(sample_number,
                                    device=device,
                                    dtype=torch.long)
    logger.debug(f'starting generating with sample_number={sample_number}, '
                 f'prepend_init_states={prepend_init_states}')

    if prepend_init_states:
        sample_number -= window_size
        init_index += window_size
        generated_samples[:window_size] = input_seq[0, :]

    for iteration in range(sample_number):
        next_sample = _get_next_prediction()
        input_seq = input_seq.roll(-1)
        input_seq[0, -1] = next_sample
        generated_samples[init_index + iteration] = next_sample
    return generated_samples.cpu()


def get_model_size(model):
    return sum(p.numel() for p in model.parameters())


def get_eff_memory(filter_size, n_layers):
    final_dilation_factor = 2**(n_layers - 1)
    last_layer_memory = (filter_size - 1) * final_dilation_factor
    return last_layer_memory