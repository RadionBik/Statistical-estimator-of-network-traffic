import pandas as pd

from pcap_parsing.parsed_fields import select_features, ParsedFields


def _filter_iat_outliers(df):
    q_low = df[ParsedFields.iat].quantile(0.01)
    q_hi = df[ParsedFields.iat].quantile(0.99)

    return df[(df[ParsedFields.iat] < q_hi) & (df[ParsedFields.iat] > q_low)]


def load_train_test_dataset(path, max_rows=10_000, filter_iat_outliers=True):
    extr_stats = pd.read_csv(path, nrows=max_rows)
    if filter_iat_outliers:
        extr_stats = _filter_iat_outliers(extr_stats)

    train_size = len(extr_stats) - len(extr_stats) // 3
    train_df, test_df = extr_stats.iloc[:train_size], extr_stats.iloc[train_size:]
    return train_df, test_df


def quantize_datatset(quantizer, train_df, test_df, prepend_with_init_tokens=0):
    train_states = quantizer.transform(*select_features(train_df), prepend_with_init_tokens=prepend_with_init_tokens)
    test_states = quantizer.transform(*select_features(test_df))
    return train_states, test_states


def restore_features(quantizer, gen_states):
    restored_packets, from_idx = quantizer.inverse_transform(gen_states)
    packets = pd.DataFrame(restored_packets, columns=[ParsedFields.ps, ParsedFields.iat])
    packets[ParsedFields.is_source] = from_idx
    return packets
