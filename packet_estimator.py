"""
usage example:

python packet_estimator.py -p traffic_dumps/skypeLANhome.pcap -i flow --plot

python packet_estimator.py -p traffic_dumps/skypeLANhome.pcap -i flow -f all -n auto

"""

import argparse
import mixture_models
import plotting
import utils
import logging
import settings
from pcap_parser import get_traffic_features


logger = logging.getLogger(__name__)


def _get_pcap_filename(args):
    return args.p.split('/')[len(args.p.split('/')) - 1].split('.')[0]


def _save_estimations(args, identifiers, estimations):
    # save estimations to disk

    id_s = ['_'.join(identifier.split(' ')[:2]) for identifier in identifiers]
    file_to_save_objects = _get_pcap_filename(args) + '_' + '_'.join(id_s). \
        replace(':', '_').replace(' ', '__').replace('.', '_') + '_' + str(args.n) + '_components'
    utils.save_obj(estimations, file_to_save_objects)
    logger.info(f'Saved results to: obj/{file_to_save_objects}.pkl')


def _parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", help="pcap file to appoximate flow/device from", required=True)
    arg_parser.add_argument("-i",
                            help="specify identifier type to be read from the file, either 'IP', or 'MAC', or 'flow' ",
                            default='flow')
    # parser.add_argument("-l", help="approximation level, either 'flow' or 'device'",default="device")
    arg_parser.add_argument("-f",
                            help="file with devices or flow identifiers to process: MAC (e.g. xx:xx:xx:xx:xx:xx), "
                                 "IP (e.g. 172.16.0.1), 5-tuple (e.g. TCP 172.16.0.1:4444 172.16.0.2:8888). "
                                 "if not specified, then every flow within the pcap will be estimated",
                            default=None)
    arg_parser.add_argument("-n",
                            help="estimate with N components. Opt for 'auto' if not sure, although takes more time to "
                                 "estimate",
                            default='auto')
    arg_parser.add_argument("-percentiles",
                            help="specify the lower and upper percentiles to remove possible outliers. Default is 3,97",
                            default='3,97')
    arg_parser.add_argument('--plot', dest='plot', action='store_true')
    arg_parser.add_argument('--hist', dest='hist', action='store_true')
    arg_parser.add_argument('--no-hist', dest='hist', action='store_false')
    arg_parser.add_argument('--no-plot', dest='plot', action='store_false')
    arg_parser.add_argument('--save-plot', dest='save_plot', action='store_true')
    arg_parser.set_defaults(plot=False, hist=False, save_plot=False)

    return arg_parser.parse_args()


def main():
    args = _parse_args()
    min_samples_to_estimate = 100
    percentiles = (int(args.percentiles.split(',')[0]), int(args.percentiles.split(',')[1]))
    extracted_traffic, identifiers = get_traffic_features(args.p,
                                                          file_with_identifiers=args.f,
                                                          type_of_identifier=utils.TrafficObjects(
                                                              args.i),
                                                          percentiles=percentiles,
                                                          min_samples_to_estimate=min_samples_to_estimate)

    estimated_em, kde_estimators = mixture_models.get_traffic_estimations(extracted_traffic,
                                                                          component_numb=args.n,
                                                                          min_samples_to_estimate=min_samples_to_estimate)

    # kde_estimations = get_KDE(extracted_traffic)

    # plot_KDE_traffic(extracted_traffic)

    traffic_extreme_values = utils.get_traffic_extreme_values(extracted_traffic)
    _save_estimations(args, identifiers, (estimated_em, traffic_extreme_values))

    mixture_models.printEstimatedParametersEM(estimated_em)

    if args.plot:
        plotting.plot_hist_kde_em(extracted_traffic, kde_estimators, estimated_em, log_scale=False,
                                  save_to_file=args.save_plot)

    if args.hist:
        plotting.hist_dfs(traffic=extracted_traffic, log_scale=False)

    # input("Press any key to continue.")


if __name__ == "__main__":
    main()
