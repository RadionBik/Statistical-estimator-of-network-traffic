import logging
import sys

import numpy as np
import neptune
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

sys.path.append('..')
import stat_metrics


logger = logging.getLogger(__name__)


class StatesDataset(Dataset):
    def __init__(self, states: np.ndarray, window=200, device='cpu'):
        self.window_size = window
        states_tensor = torch.as_tensor(states, dtype=torch.float, device=device)
        # chunk vector into windows shifted by one position, autoregressive approach
        ar_states = states_tensor.unfold(0, self.window_size, 1)
        ar_states = ar_states

        self.x = ar_states[:-1]
        self.y = ar_states[1:].to(dtype=torch.long)
        self.n_states = len(set(states))
        self.len_init_states = len(states)
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


def validate(model, test_dataset: StatesDataset, n_classes, log=False):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    for one_batch in DataLoader(test_dataset, batch_size=len(test_dataset),
                                drop_last=True,
                                shuffle=True):
        test_x, test_y = one_batch

        with torch.no_grad():
            output = model(test_x.unsqueeze(1).contiguous())
            loss = calc_loss(criterion, output, test_y, n_classes)
            accuracy = calc_accuracy(output, test_y, n_classes)
            logger.info('Test set: Average loss: {:.8f}  |  Accuracy: {:.4f}'.format(
                loss.item(), accuracy))
            if log:
                neptune.log_metric('evaluation/loss', loss)
                neptune.log_metric('evaluation/accuracy', accuracy)
    return loss.item()


def train(model, optimizer, train_dataset: StatesDataset, parameters, log=False):
    model.train()
    losses = []
    accuracies = []
    criterion = torch.nn.CrossEntropyLoss()

    dataloader = DataLoader(train_dataset,
                            batch_size=parameters.batch_size,
                            drop_last=True,
                            shuffle=True)

    for batch_idx, data_batch in enumerate(dataloader):
        x, y = data_batch

        optimizer.zero_grad()

        input = x.unsqueeze(1).contiguous()
        output = model(input)
        loss = calc_loss(criterion, output, y, parameters.n_classes)

        if parameters.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.grad_clip)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(calc_accuracy(output, y, parameters.n_classes))

        if batch_idx > 0 and batch_idx % parameters.log_interval == 0:
            avg_loss = np.mean(losses[batch_idx - parameters.log_interval:batch_idx])
            accuracy = np.mean(accuracies[batch_idx - parameters.log_interval:batch_idx])

            logger.info('| {:5d}/{:5d} batches | '
                        'loss {:5.8f} | accuracy {:5.4f}'.format(batch_idx,
                                                                 len(train_dataset) // parameters.batch_size + 1,
                                                                 avg_loss, accuracy))
    if log:
        neptune.log_metric('training/loss', np.mean(losses))
        neptune.log_metric('training/accuracy', np.mean(accuracies))


def generate_states(model, dataset, sample_number=None, shuffle=False, device='cpu', prepend_init_states=True):
    def _get_next_prediction():
        with torch.no_grad():
            out = model(input_seq.unsqueeze(1))
            max_out = out.squeeze(0).max(1, keepdim=True)[1]
            return max_out[-1].unsqueeze(1)

    model.eval()

    input_seq, _ = next(iter(DataLoader(dataset, batch_size=1, drop_last=True, shuffle=shuffle)))
    window_size = dataset.window_size
    if not sample_number:
        # match len with original states
        sample_number = dataset.len_init_states

    init_index = 0
    generated_samples = torch.zeros(sample_number, device=device, dtype=torch.long)

    logger.debug(f'starting generating with sample_number={sample_number}, '
                 f'prepend_init_states={prepend_init_states}')

    if prepend_init_states:
        sample_number -= window_size
        init_index += window_size
        generated_samples[:window_size] = input_seq[0, :]

    for iteration in range(sample_number):
        next_sample = _get_next_prediction()
        input_seq = input_seq.roll(-1)
        input_seq[0, -1] = next_sample
        generated_samples[init_index + iteration] = next_sample
    return generated_samples.cpu()


def evaluate_KL_distance(generated_states, orig_states, log):
    logger.debug('calculating KL distance...')
    distance = stat_metrics.get_KL_divergence_pdf(orig_states, generated_states)
    if log:
        neptune.log_metric('training/KL_divergence', distance)
    logger.info(f'KL distance is: {distance}')
    return distance


def get_model_size(model):
    return sum(p.numel() for p in model.parameters())
