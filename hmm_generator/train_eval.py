import argparse
import pathlib

import matplotlib.pyplot as plt
import neptune

from features.auto_select_model_by_bic import plot_bics
from features.data_utils import load_train_test_dataset
from features.evaluation import evaluate_traffic
from hmm_generator.model import HMMGenerator
from pcap_parsing.parsed_fields import select_features
from settings import BASE_DIR, NEPTUNE_PROJECT, NEPTUNE_API_TOKEN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='path to preprocessed .csv dataset',
        required=True
    )
    parser.add_argument(
        '--log_neptune',
        dest='log_neptune',
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    ds_path = pathlib.Path(args.dataset)
    save_dir = BASE_DIR / 'obj' / ('hmm_' + ds_path.stem)

    train_df, test_df = load_train_test_dataset(ds_path)

    generator, bic_dict = HMMGenerator().fit(
        *select_features(train_df),
        min_comp=10, max_comp=120, step_comp=4,
        return_bic_dict=True
    )
    generator.save_pretrained(save_dir)

    plot_bics(bic_dict['from'], 'От источника')
    plot_bics(bic_dict['to'], 'К источнику')
    plt.tight_layout()
    plt.savefig(save_dir / 'BICs.png', dpi=300)

    gen_df = generator.sample_like(test_df)
    eval_metrics = evaluate_traffic(gen_df, test_df)
    if args.log_neptune:
        neptune.init(
            NEPTUNE_PROJECT,
            NEPTUNE_API_TOKEN,
        )
        neptune.create_experiment(name='hmm_model', params=vars(args))
        for name, value in eval_metrics.items():
            neptune.log_metric(name, value)
        neptune.stop()


if __name__ == '__main__':
    main()
