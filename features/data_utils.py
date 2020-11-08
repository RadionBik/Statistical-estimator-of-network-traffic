import logging
import pickle

import pandas as pd

import settings
from pcap_parsing.parsed_fields import select_features, ParsedFields

logger = logging.getLogger(__name__)


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
    restored_packets, from_idx = quantizer.inverse_transform(gen_states, prob_sampling=True)
    packets = pd.DataFrame(restored_packets, columns=[ParsedFields.ps, ParsedFields.iat])
    packets[ParsedFields.is_source] = from_idx
    return packets


def save_obj(obj, name, by_stem=True):
    """
    save_obj() saves python object to the file inside 'obj' directory
    """
    if by_stem:
        dest_path = settings.BASE_DIR / f'obj/{name}.pkl'
    else:
        dest_path = name
    with open(dest_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    logger.info(f'pickled obj to {dest_path}')
    return dest_path.as_posix()


def load_obj(name, by_stem=True):
    """
    load_obj() loads python object from the file inside 'obj' directory
    """
    if by_stem:
        load_path = settings.BASE_DIR / f'obj/{name}.pkl'
    else:
        load_path = name
    with open(load_path, 'rb') as f:
        obj = pickle.load(f)
        logger.info(f'loaded obj from {load_path}')
        return obj, load_path.as_posix()
