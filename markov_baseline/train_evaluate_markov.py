import settings
import stat_metrics
from features.evaluation import evaluate_traffic
from features.gaussian_quantizer import GaussianQuantizer
from features.data_utils import load_train_test_dataset, quantize_datatset, restore_features
from markov_baseline.model import MarkovSequenceGenerator


def main():
    train_df, test_df = load_train_test_dataset(settings.BASE_DIR / 'traffic_dumps/iot_amazon_echo.pcap.csv',
                                                10_000)
    q_path = settings.BASE_DIR / 'obj' / 'amazon_10k'
    quantizer = GaussianQuantizer.from_pretrained(q_path)
    train_states, test_states = quantize_datatset(quantizer, train_df, test_df)
    model = MarkovSequenceGenerator()
    model.fit(train_states)
    sampled = model.sample(len(test_states))
    seq_metrics = stat_metrics.calc_stats(test_states, sampled)
    gen_df = restore_features(quantizer, sampled)
    eval_metrics = evaluate_traffic(gen_df, test_df)


if __name__ == '__main__':
    main()
