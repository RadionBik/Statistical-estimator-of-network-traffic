import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from features.gaussian_quantizer import GaussianQuantizer
from features.data_utils import load_train_test_dataset
from pcap_parsing.parsed_fields import select_features
from settings import BASE_DIR


def plot_bics(bics, direction):
    bics_df = pd.DataFrame(bics.items())
    bics_df.columns = ['Число компонент', 'BIC']
    ax = sns.regplot(data=bics_df, x='Число компонент', y='BIC',
                     order=2, label=direction, scatter=True)
    ax.legend()


if __name__ == '__main__':

    scenario = 'amazon_10k'

    train_df, test_df = load_train_test_dataset(BASE_DIR / 'traffic_dumps/iot_amazon_echo.pcap.csv', 10_000)

    quantizer, bic_dict = GaussianQuantizer().fit(
        *select_features(train_df),
        min_comp=10, max_comp=120, step_comp=4,
        return_bic_dict=True
    )

    quantizer.save_pretrained(BASE_DIR / 'obj' / scenario)

    plot_bics(bic_dict['from'], 'От источника')
    plot_bics(bic_dict['to'], 'К источнику')
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'obj' / scenario / 'BICs.png', dpi=300)
