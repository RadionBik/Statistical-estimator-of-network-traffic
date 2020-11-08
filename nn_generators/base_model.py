import torch
from pytorch_lightning import LightningModule


class NNBaseGenerator(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimizer)(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def _calc_loss(self, output, target):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._calc_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.contiguous())
        loss = self._calc_loss(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return {'val_loss': loss}

    def get_next_prediction(self, input_seq):
        raise NotImplementedError
