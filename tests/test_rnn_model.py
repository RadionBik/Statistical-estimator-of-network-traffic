import torch

from nn_generators.rnn.model import RNNGenerator
from nn_generators.rnn.config import RNNConfig


def test_model(gmm_state):
    n_class = max(gmm_state) + 1
    window = 32

    config = RNNConfig('test', n_classes=n_class, window_size=window)
    model = RNNGenerator(config)
    states = torch.tensor(gmm_state)
    input_ = states[:window * 2].view(2, -1).to(torch.float)
    out = model(input_)
    assert out.shape == torch.Size([2, n_class])
    pred = model.get_next_prediction(input_[:1])
    assert pred.shape == torch.Size([])
