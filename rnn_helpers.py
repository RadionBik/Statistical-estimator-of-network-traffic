from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import stat_tests
from traffic_helpers import unpack_2layer_traffic_dict, construct_dict_2_layers, iterate_traffic_dict
import ts_helpers as timeseries
import matplotlib.pyplot as plt


class StopAtLossValue(keras.callbacks.Callback):

    def __init__(self, threshold):
        self.threshold = threshold

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('loss') <= self.threshold:
            self.model.stop_training = True


@unpack_2layer_traffic_dict
def get_rnn_models(train_states, window_size, loss_threshold, layers=1):
    # stop_callback = StopAtLossValue(loss_threshold)
    stop_callback = EarlyStopping(patience=5,
                                  restore_best_weights=True,
                                  monitor='val_loss')
    state_numb = int(max(set(train_states)) + 1)
    X, y = get_one_hot_training_states(train_states, window_size, step=5)
    model = build_gru_predictor(window_size, state_numb, layers=layers)

    history = model.fit(X,
                        y,
                        epochs=100,
                        validation_split=0.2,
                        batch_size=20,
                        callbacks=[stop_callback])

    pd.DataFrame(history.history).plot()

    return model


def build_gru_predictor(window_size, state_numb, layers=1):
    model = Sequential()
    for _ in range(layers - 1):
        model.add(GRU(state_numb,
                      activation='relu',
                      input_shape=(window_size, state_numb),
                      return_sequences=True))

    model.add(GRU(state_numb, input_shape=(window_size, state_numb)))
    model.add(Dropout(0.2))
    model.add(Dense(state_numb, activation='softmax'))

    optimizer = keras.optimizers.RMSprop(lr=0.005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  )
    return model


@unpack_2layer_traffic_dict
def get_mixture_state_predictions(mixture_model, traffic_df):
    return mixture_model.predict(traffic_df)


def change_pmf_temperature(preds, temperature=1.0):
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    try:
        probas = np.random.multinomial(1, preds, 1)
    except ValueError:
        probas = np.random.multinomial(1, np.around(preds), 1)
    return np.argmax(probas)


def rnn_gener_state(model, sample_number_to_gener, init_states, window_size, state_numb, temperature=1.0):
    # start_index = random.randint(0, len(init_states) - window_size - 1)
    start_index = 0
    seed_states = init_states[start_index: start_index + window_size]
    generated_states = np.zeros(sample_number_to_gener)
    generated_states[:window_size] = seed_states
    # print('Testing temperature:', temperature)
    for i in range(sample_number_to_gener - window_size):
        # one-hot encode seed_states
        sampled = np.zeros((1, window_size, state_numb))
        for t, state in enumerate(seed_states):
            sampled[0, t, state] = 1
        one_prediction = model.predict(sampled, verbose=0)[0]

        next_state = change_pmf_temperature(one_prediction, temperature)
        generated_states[window_size + i - 1] = next_state
        seed_states[0] = next_state
        seed_states = np.roll(seed_states, -1)

    return generated_states


def gener_rnn_states_with_temperature(model,
                                      orig_states,
                                      window_size,
                                      sample_number_to_gener=None,
                                      temperatures=None):
    if not sample_number_to_gener:
        sample_number_to_gener = len(orig_states)

    state_numb = max(set(orig_states)) + 1
    if not temperatures:
        init_entropy = np.mean(timeseries.calc_windowed_entropy_discrete(orig_states))
        temperatures = [init_entropy]
        # if init_entropy < 1.7:
        #     temperatures = [init_entropy - 0.2,
        #                     init_entropy,
        #                     init_entropy + 0.2]
        # else:
        #     temperatures = [init_entropy - 0.4,
        #                     init_entropy - 0.2,
        #                     init_entropy]
        # print('\nSelected base temperature from init_states: {:.3f}'.format(
        #     init_entropy))
    temp_dist = {}
    for temperature in temperatures:
        print('Trying temperature={:.3f}...'.format(temperature))
        rnn_states = rnn_gener_state(model,
                                     sample_number_to_gener,
                                     orig_states,
                                     window_size,
                                     state_numb,
                                     temperature)

        # min_len = min(len(rnn_states), len(init_states))
        # distance = get_KL_divergence_pmf(init_states[:min_len],
        #                                   rnn_states[:min_len],
        #                                   state_numb)
        temp_dist[temperature] = stat_tests.get_KL_divergence_pmf(orig_states,
                                                                  rnn_states)

        print('Got KL={:.3f}'.format(temp_dist[temperature]))

    # print('\nBest KL divergence={:.3f} for states with temperature={:.3f}'.format(least_distance, best_temper))

    return rnn_states


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
    print(X.shape)
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

    print('Number of sequences:', len(sentences))
    print('Unique states:', len(unique_states))

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
