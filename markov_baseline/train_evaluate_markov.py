import argparse

import neptune

import features.evaluation
import settings
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
    parser.add_argument(
        '--log_neptune',
        dest='log_neptune',
        action='store_true',
        default=False
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
    seq_metrics = features.evaluation.calc_stats(test_states, sampled)
    gen_df = restore_features(quantizer, sampled)
    eval_metrics = evaluate_traffic(gen_df, test_df)
    if args.log_neptune:
        neptune.init(
            settings.NEPTUNE_PROJECT,
            settings.NEPTUNE_API_TOKEN,
        )
        neptune.create_experiment(name='markov_model', params=vars(args))
        for name, value in dict(**seq_metrics, **eval_metrics).items():
            neptune.log_metric(name, value)
        neptune.stop()


if __name__ == '__main__':
    main()
