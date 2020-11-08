import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def generate_states(model, sample_number, window_size, device='cpu'):
    model.eval()

    start_seq = get_init_seq(window_size)
    input_seq = torch.tensor(start_seq, dtype=torch.float).unsqueeze(0)

    init_index = 0
    generated_samples = torch.zeros(sample_number,
                                    device=device,
                                    dtype=torch.long)
    logger.info(f'starting generating with sample_number={sample_number}')

    for iteration in range(sample_number):
        next_sample = model.get_next_prediction(input_seq)
        input_seq = input_seq.roll(-1)
        input_seq[0, -1] = next_sample
        generated_samples[init_index + iteration] = next_sample
    return generated_samples.cpu()


def get_model_size(model):
    return sum(p.numel() for p in model.parameters())


def get_init_seq(window_size, pad_token=0, start_token=1):
    # [0, 0,...,1] -- pad tokens with start of seq.
    kickstart_seq = np.empty(window_size, dtype=np.int)
    kickstart_seq[:-1] = pad_token
    kickstart_seq[-1] = start_token
    return kickstart_seq
