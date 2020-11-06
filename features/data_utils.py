import pandas as pd

from pcap_parsing.parsed_fields import select_features


def load_train_test_dataset(path, nrows=None):
    extr_stats = pd.read_csv(path, nrows=nrows)
    train_size = len(extr_stats) - len(extr_stats) // 3
    train_df, test_df = extr_stats.iloc[:train_size], extr_stats.iloc[train_size:]
    return train_df, test_df


def quantize_datatset(quantizer, train_df, test_df, prepend_with_init_tokens=0):
    train_states = quantizer.transform(*select_features(train_df), prepend_with_init_tokens=prepend_with_init_tokens)
    test_states = quantizer.transform(*select_features(test_df))
    return train_states, test_states