import argparse

import stat_metrics
from features.data_utils import load_train_test_dataset, quantize_datatset, restore_features
from features.evaluation import evaluate_traffic
from features.gaussian_quantizer import GaussianQuantizer
from markov_baseline.model import MarkovSequenceGenerator


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='path to preprocessed .csv dataset',
        required=True
    )
    parser.add_argument(
        '--quantizer_path',
        help='path to the quantizer checkpoint',
        required=True
    )
    return parser.parse_args()


def main():

    args = _parse_args()

    train_df, test_df = load_train_test_dataset(args.dataset)
    quantizer = GaussianQuantizer.from_pretrained(args.quantizer_path)
    train_states, test_states = quantize_datatset(quantizer, train_df, test_df)
    model = MarkovSequenceGenerator()
    model.fit(train_states)
    sampled = model.sample(len(test_states))
    seq_metrics = stat_metrics.calc_stats(test_states, sampled)
    gen_df = restore_features(quantizer, sampled)
    eval_metrics = evaluate_traffic(gen_df, test_df)


if __name__ == '__main__':
    main()
