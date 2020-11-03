import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from features.gaussian_quantizer import GaussianQuantizer
from pcap_parsing.parsed_fields import select_features
from settings import BASE_DIR


def plot_bics(bics, direction):
    bics_df = pd.DataFrame(bics.items())
    bics_df.columns = ['Число компонент', 'BIC']
    ax = sns.regplot(data=bics_df, x='Число компонент', y='BIC',
                     order=2, label=direction, scatter=True)
    ax.legend()


if __name__ == '__main__':
    extr_stats = pd.read_csv(BASE_DIR / 'traffic_dumps/iot_amazon_echo.pcap.csv',
                             nrows=10_000)

    scenario = 'amazon_10k'

    train_size = len(extr_stats) - len(extr_stats) // 3
    train_df, test_df = extr_stats.iloc[:train_size], extr_stats.iloc[train_size:]

    quantizer, bic_dict = GaussianQuantizer().fit(
        *select_features(train_df),
        min_comp=10, max_comp=100, step_comp=4,
        return_bic_dict=True
    )

    quantizer.save_pretrained(BASE_DIR / 'obj' / scenario)

    plot_bics(bic_dict['from'], 'От источника')
    plot_bics(bic_dict['to'], 'К источнику')
    plt.tight_layout()
    plt.savefig(f'{scenario}.png', dpi=300)
