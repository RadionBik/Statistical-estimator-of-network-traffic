import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from features.data_utils import load_train_test_dataset
from features.evaluation import plot_packets_dist
from features.gaussian_quantizer import GaussianQuantizer
from features.packet_scaler import PacketScaler
from pcap_parsing.parsed_fields import select_features, ParsedFields
from settings import BASE_DIR


def plot_bics(bics: dict, direction):
    bics_df = pd.DataFrame(bics.items())
    bics_df.columns = ['Число компонент', 'BIC']
    ax = sns.regplot(data=bics_df, x='Число компонент', y='BIC',
                     order=2, label=direction, scatter=True)
    ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='path to preprocessed .csv dataset',
        required=True
    )
    args = parser.parse_args()
    ds_path = pathlib.Path(args.dataset)
    save_dir = BASE_DIR / 'obj' / ds_path.stem

    train_df, test_df = load_train_test_dataset(ds_path)

    quantizer, bic_dict = GaussianQuantizer().fit(
        *select_features(train_df),
        min_comp=10, max_comp=120, step_comp=4,
        return_bic_dict=True
    )
    quantizer.save_pretrained(save_dir)

    plot_bics(bic_dict['from'], 'От источника')
    plot_bics(bic_dict['to'], 'К источнику')
    plt.tight_layout()
    plt.savefig(save_dir / 'BICs.png', dpi=300)
    plot_packets_dist(train_df)
    plt.savefig(save_dir / 'packets.png', dpi=300)

    scaled = PacketScaler().transform(select_features(train_df)[0])
    scaled = pd.DataFrame(scaled, columns=['PS, байт / 1500', 'log(IAT, мс)'])
    scaled[ParsedFields.is_source] = train_df[ParsedFields.is_source].reset_index(drop=True)

    plot_packets_dist(scaled, x='log(IAT, мс)', y='PS, байт / 1500')
    plt.savefig(save_dir / 'scaled_packets.png', dpi=300)


if __name__ == '__main__':
    main()
