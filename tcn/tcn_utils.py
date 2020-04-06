import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class StatesDataset(Dataset):
    def __init__(self, states: np.ndarray, window=200):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        states_tensor = torch.Tensor(states)
        # chunk vector into windows shifted by one position, autoregressive approach
        ar_states = states_tensor.unfold(0, window, 1)
        ar_states = ar_states.to(device)

        self.x = ar_states[:-1]
        self.y = ar_states[1:].type(torch.LongTensor).to(device)
        self.n_states = len(set(states))
        logger.info(f'dataset: got {len(states)} states with {self.n_states} unique')

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def calc_loss(criterion, output, target, n_classes):
    return criterion(output.view(-1, n_classes), target.view(-1))


def calc_accuracy(output, target, n_classes):
    pred = output.view(-1, n_classes).data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    counter = output.view(-1, n_classes).size(0)
    return 100. * correct / counter


def evaluate(model, test_dataset: StatesDataset, n_classes):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    for one_batch in DataLoader(test_dataset, batch_size=len(test_dataset), drop_last=True, shuffle=True):
        test_x, test_y = one_batch

        with torch.no_grad():
            output = model(test_x.unsqueeze(1).contiguous())
            loss = calc_loss(criterion, output, test_y, n_classes)
            accuracy = calc_accuracy(output, test_y, n_classes)
            logger.info('Test set: Average loss: {:.8f}  |  Accuracy: {:.4f}\n'.format(
                loss.item(), accuracy))
            return loss.item()


def train(model, optimizer, train_dataset: StatesDataset, parameters):
    model.train()
    losses = []
    accuracies = []
    criterion = torch.nn.CrossEntropyLoss()

    dataloader = DataLoader(train_dataset, batch_size=parameters['batch_size'], drop_last=True, shuffle=True)

    for batch_idx, data_batch in enumerate(dataloader):
        x, y = data_batch

        optimizer.zero_grad()

        input = x.unsqueeze(1).contiguous()
        output = model(input)
        loss = calc_loss(criterion, output, y, parameters['n_classes'])

        if parameters['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), parameters['grad_clip'])
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(calc_accuracy(output, y, parameters['n_classes']))

        if batch_idx > 0 and batch_idx % parameters['log_interval'] == 0:
            avg_loss = np.mean(losses[batch_idx - parameters['log_interval']:batch_idx])
            accuracy = np.mean(accuracies[batch_idx - parameters['log_interval']:batch_idx])

            logger.info('| {:5d}/{:5d} batches | '
                        'loss {:5.8f} | accuracy {:5.4f}'.format(batch_idx,
                                                                 len(train_dataset) // parameters['batch_size'] + 1,
                                                                 avg_loss, accuracy))
