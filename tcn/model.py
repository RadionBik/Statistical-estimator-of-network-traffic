import logging

import torch
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from base_model import TemporalConvNet


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s:%(lineno)d %(levelname)s - %(message)s')
logging.getLogger('neptune.internal.channels.channels_values_sender').setLevel(logging.ERROR)


class TCNGenerator(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.tcn = TemporalConvNet(config.input_size,
                                   config.num_channels,
                                   kernel_size=config.kernel_size,
                                   dropout=config.dropout)
        self.linear = torch.nn.Linear(config.num_channels[-1], config.n_classes)
        self.init_weights()
        self.config = config

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.tcn(x.unsqueeze(1))
        out = self.linear(out.transpose(1, 2))
        return out

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimizer)(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def _calc_loss(self, output, target, n_classes):
        loss = F.cross_entropy(output.view(-1, n_classes), target.view(-1))
        return loss

    def _calc_accuracy(self, output, target, n_classes):
        pred = output.view(-1, n_classes).data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum()
        counter = output.view(-1, n_classes).size(0)
        return 100. * correct / counter

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._calc_loss(y_hat, y, self.config.n_classes)
        accuracy = self._calc_accuracy(y_hat, y, self.config.n_classes)
        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.contiguous())
        loss = self._calc_loss(y_hat, y, self.config.n_classes)
        accuracy = self._calc_accuracy(y_hat, y, self.config.n_classes)
        log = {'val_loss': loss, 'val_accuracy': accuracy}
        return log

    def get_next_prediction(self, input_seq):
        with torch.no_grad():
            out = self(input_seq).squeeze(0)
            last_prob = torch.softmax(out[-1], 0)
            pred = torch.multinomial(last_prob, 1)
            return pred[0]
