import torch
from torch.nn import functional as F

from nn_generators.base_model import NNBaseGenerator


class RNNGenerator(NNBaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.rnn = torch.nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout
        )
        self.linear = torch.nn.Linear(config.hidden_size, config.n_classes)

    def forward(self, x):
        out, h_t = self.rnn(x.unsqueeze(2))
        # pull hidden state at last t
        out = out[:, -1, :]
        out = self.linear(out)
        return out

    def _calc_loss(self, output, target):
        return F.cross_entropy(output, target)

    def get_next_prediction(self, input_seq):
        with torch.no_grad():
            out = self(input_seq).squeeze(0)
            last_prob = torch.softmax(out, 0)
            pred = torch.multinomial(last_prob, 1)
            return pred[0]
