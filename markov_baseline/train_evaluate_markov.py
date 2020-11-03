from pprint import pprint

import settings
import stat_metrics
from features.gaussian_quantizer import GaussianQuantizer
from features.utils import load_train_test_dataset, quantize_datatset
from markov_baseline.model import MarkovSequenceGenerator

if __name__ == '__main__':
    train_df, test_df = load_train_test_dataset(settings.BASE_DIR / 'traffic_dumps/iot_amazon_echo.pcap.csv', 10_000)
    q_path = settings.BASE_DIR / 'obj' / 'amazon_10k'
    quantizer = GaussianQuantizer.from_pretrained(q_path)
    train_states, test_states = quantize_datatset(quantizer, train_df, test_df)
    model = MarkovSequenceGenerator()
    model.fit(train_states)
    sampled = model.sample(len(test_states))
    res_metrics = stat_metrics.calc_stats(test_states, sampled)
