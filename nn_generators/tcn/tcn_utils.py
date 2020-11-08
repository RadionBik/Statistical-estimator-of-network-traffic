import logging

import numpy as np

from nn_generators.nn_utils import get_model_size

logger = logging.getLogger(__name__)


def get_eff_memory(filter_size, n_layers):
    final_dilation_factor = 2 ** (n_layers - 1)
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
