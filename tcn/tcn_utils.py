import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class StatesDataset(Dataset):
    def __init__(self, states: np.ndarray, window=200):
        states_tensor = torch.Tensor(states)
        # chunk vector into windows shifted by one position, autoregressive approach
        ar_states = states_tensor.unfold(0, window, 1)
        self.x = ar_states[:-1]
        self.y = ar_states[1:]
        self.n_states = len(set(states))
        logger.info(f'dataset: got {len(states)} states with {self.n_states} unique')

    def __getitem__(self, index):
        return self.x[index], self.y[index].type(torch.LongTensor)

    def __len__(self):
        return self.x.shape[0]


def evaluate(model, test_dataset: StatesDataset, n_classes):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    for one_batch in DataLoader(test_dataset, batch_size=len(test_dataset), drop_last=True, shuffle=True):
        test_x, test_y = one_batch

        with torch.no_grad():
            out = model(test_x.unsqueeze(1).contiguous())
            loss = criterion(out.view(-1, n_classes), test_y.view(-1))
            pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
            correct = pred.eq(test_y.data.view_as(pred)).cpu().sum()
            counter = out.view(-1, n_classes).size(0)
            logger.info('Test set: Average loss: {:.8f}  |  Accuracy: {:.4f}\n'.format(
                loss.item(), 100. * correct / counter))
            return loss.item()


def train(model, optimizer, train_dataset: StatesDataset, current_epoch, parameters):
    model.train()
    total_loss = 0
    correct = 0
    counter = 0
    criterion = torch.nn.CrossEntropyLoss()

    dataloader = DataLoader(train_dataset, batch_size=parameters['batch_size'], drop_last=True, shuffle=True)

    for batch_idx, data_batch in enumerate(dataloader):
        x, y = data_batch

        optimizer.zero_grad()

        input = x.unsqueeze(1).contiguous()
        out = model(input)
        out = out.view(-1, parameters['n_classes'])
        loss = criterion(out, y.view(-1))

        if parameters['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['grad_clip'])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        counter += out.size(0)

        if batch_idx > 0 and batch_idx % parameters['log_interval'] == 0:
            avg_loss = total_loss / parameters['log_interval']
            logger.info('| {:5d}/{:5d} batches | loss {:5.8f} | '
                        'accuracy {:5.4f}'.format(batch_idx,
                                                  len(train_dataset) // parameters['batch_size'] + 1,
                                                  avg_loss, 100. * correct / counter))
            total_loss = 0
            correct = 0
            counter = 0
