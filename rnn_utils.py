import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

logger = logging.getLogger(__name__)


class StopAtLossValue(keras.callbacks.Callback):

    def __init__(self, threshold):
        self.threshold = threshold

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('loss') <= self.threshold:
            self.model.stop_training = True


def change_pmf_temperature(preds, temperature=1.0):
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    try:
        probas = np.random.multinomial(1, preds, 1)
    except ValueError:
        probas = np.random.multinomial(1, np.around(preds), 1)
    return np.argmax(probas)


def rnn_gener_state(model: keras.Model,
                    full_init_states,
                    window_size: int,
                    temperature=1.0,
                    sample_number_to_gener=None):
    # start_index = random.randint(0, len(init_states) - window_size - 1)
    start_index = 0
    seed_states = full_init_states[start_index: start_index + window_size]
    if not sample_number_to_gener:
        sample_number_to_gener = len(full_init_states)
    generated_states = np.zeros(sample_number_to_gener)
    generated_states[:window_size] = seed_states
    # logger.info('Testing temperature:', temperature)
    state_numb = max(set(full_init_states)) + 1
    for i in range(sample_number_to_gener - window_size):
        # one-hot encode seed_states
        sampled = np.zeros((1, window_size, state_numb))
        for t, state in enumerate(seed_states):
            sampled[0, t, state] = 1
        one_prediction = model.predict_on_batch(sampled)[0]

        next_state = change_pmf_temperature(one_prediction, temperature)
        generated_states[window_size + i - 1] = next_state
        seed_states[0] = next_state
        seed_states = np.roll(seed_states, -1)
        if i%1000 == 0:
            logger.info(f'generated {i} states')

    return generated_states


def get_windowed_training_set_m2m(df, window_size, shift=1):
    feature_number = df.shape[1]
    sample_number = df.shape[0]
    X = np.zeros((sample_number - window_size + 1, window_size, feature_number))
    y = np.zeros((sample_number - window_size + 1, window_size, feature_number))

    for batch in range(sample_number - window_size + 1):
        X[batch, :, :] = df.iloc[batch:batch + window_size, :]
        y[batch, :, :] = df.shift(shift).fillna(0).iloc[batch:batch + window_size, :]
    return X, y


def get_batched_training_set_m2m(df, window_size, shift=1):
    batch_num = int(np.ceil(df.shape[0] / window_size))
    zeros_to_append = np.zeros((batch_num * window_size - df.shape[0], df.shape[1]))
    df = df.append(pd.DataFrame(zeros_to_append, columns=df.columns), ignore_index=True)

    X = np.reshape(df.values, (batch_num, window_size, 2))
    y = np.reshape(df.shift(shift).fillna(0).values, (batch_num, window_size, 2))

    # return only non-zero batches
    return X[1:, :, :], y[1:, :, :]


def get_windowed_training_set_m2o(df, window_size, shift=1):
    if len(df.shape) == 1:
        feature_number = 1
    else:
        feature_number = df.shape[1]

    sample_number = df.shape[0]
    X = np.zeros((sample_number - window_size + 1, window_size, feature_number))
    y = np.zeros((sample_number - window_size + 1, feature_number))
    logger.info(X.shape)
    if len(df.shape) == 1:
        for batch in range(sample_number - window_size):
            X[batch, :, 0] = df.iloc[batch:batch + window_size]
            y[batch, 0] = df.iloc[batch + window_size]
    else:
        for batch in range(sample_number - window_size):
            X[batch, :, :] = df.iloc[batch:batch + window_size, :]
            y[batch, :] = df.iloc[batch + window_size, :]

    return X, y


def get_one_hot_training_states(states, window_size, step=3):
    unique_states = list(set(states))
    state_numb = max(unique_states) + 1
    state_mapping = {}
    for idx, state in enumerate(unique_states):
        state_mapping[state] = idx

    sentences = []
    next_states = []
    for i in range(0, len(states) - window_size, step):
        sentences.append(states[i: i + window_size])
        next_states.append(states[i + window_size])

    logger.info(f'Number of sequences: {len(sentences)}')
    logger.info(f'Unique states: {len(unique_states)}')

    # one-hot encoding
    x = np.zeros((len(sentences), window_size, state_numb), dtype=np.bool)
    y = np.zeros((len(sentences), state_numb), dtype=np.bool)
    # pdb.set_trace()

    for i, sentence in enumerate(sentences):
        for t, state in enumerate(sentence):
            x[i, t, state] = 1
        y[i, state] = 1

    return x, y


def plot_training_val(history, metric=None):
    print(history.history.keys())

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    if metric:
        acc = history.history[metric]
        val_acc = history.history['val_' + metric]
        plt.plot(epochs, acc, 'bo', label='Training ' + metric)
        plt.plot(epochs, val_acc, 'b', label='Validation ' + metric)
        plt.title('Training and validation ' + metric)
        plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
