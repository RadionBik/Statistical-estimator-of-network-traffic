import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class StatesDataset(Dataset):
    def __init__(self, states: np.ndarray, window=200, device='cpu'):
        self.window_size = window
        states_tensor = torch.as_tensor(
            states,
            dtype=torch.float,
            device=device
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


def generate_states(model, sample_number, window_size, device='cpu'):
    model.eval()

    start_seq = get_init_seq(window_size)
    input_seq = torch.tensor(start_seq, dtype=torch.float).unsqueeze(0)

    init_index = 0
    generated_samples = torch.zeros(sample_number,
                                    device=device,
                                    dtype=torch.long)
    logger.debug(f'starting generating with sample_number={sample_number}')

    for iteration in range(sample_number):
        next_sample = model.get_next_prediction(input_seq)
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


def estimate_cheapest_parameters(estimated_data_window, n_classes, tcn_model_class, upper_memory_size=512):
    lower_memory_bound = estimated_data_window
    upper_memory_bound = int(2 ** np.ceil(np.log2(estimated_data_window)))
    print(f'window bounds are: {lower_memory_bound} - {upper_memory_bound}')
    min_params_pair = {}
    for n_levels in range(2, int(np.ceil(np.log2(upper_memory_size))) + 1 + 1):
        for kernel_size in range(4, upper_memory_size + 1):
            model_memory = get_eff_memory(kernel_size, n_levels)
            if lower_memory_bound <= model_memory <= upper_memory_bound:
                model = tcn_model_class(1, n_classes, [n_classes] * n_levels, kernel_size)
                size = get_model_size(model)
                min_params_pair[(n_levels, kernel_size)] = size
            else:
                continue
    print(f'found {len(min_params_pair)} candidate combinations')
    (n_levels, kernel_size), param_number = sorted(min_params_pair.items(), key=lambda x: x[1])[0]
    print(f'selected pair with {param_number} params: layers={n_levels}, kernel={kernel_size}')
    return n_levels, kernel_size


def get_init_seq(window_size, pad_token=0, start_token=1):
    # [0, 0,...,1] -- pad tokens with start of seq.
    kickstart_seq = np.empty(window_size, dtype=np.int)
    kickstart_seq[:-1] = pad_token
    kickstart_seq[-1] = start_token
    return kickstart_seq
