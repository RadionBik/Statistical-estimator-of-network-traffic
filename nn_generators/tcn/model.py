import torch
from torch.nn import functional as F

from .reference_model import TemporalConvNet
from ..base_model import NNBaseGenerator


class TCNGenerator(NNBaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.tcn = TemporalConvNet(config.input_size,
                                   config.num_channels,
                                   kernel_size=config.kernel_size,
                                   dropout=config.dropout)
        self.linear = torch.nn.Linear(config.num_channels[-1], config.n_classes)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.tcn(x.unsqueeze(1))
        out = self.linear(out.transpose(1, 2))
        return out

    def _calc_loss(self, output, target):
        loss = F.cross_entropy(output.view(-1, self.config.n_classes), target.view(-1))
        return loss

    def get_next_prediction(self, input_seq):
        with torch.no_grad():
            out = self(input_seq).squeeze(0)
            last_prob = torch.softmax(out[-1], 0)
            pred = torch.multinomial(last_prob, 1)
            return pred[0]
