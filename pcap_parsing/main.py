import argparse
import logging
import pathlib

from pcap_parsing.device_level import extract_host_stats
from pcap_parsing.flow_level import extract_flow_stats
import settings

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pcapfile',
        help='path to .pcap',
        required=True
    )

    parser.add_argument(
        '--flow_level',
        help='specifies extraction mode: either device-level (default) or flow-level (when this flag is set). '
             'Target identifier must be adjusted accordingly',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--identifier',
        dest='identifier',
        help='must be either MAC|IP address or a flow with format "TCP|UDP SRC_IP:SRC_PORT DST_IP:DST_PORT"',
        required=True
    )
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    logger.info(args)
    if args.flow_level:
        extractor = extract_flow_stats
    else:
        extractor = extract_host_stats

    stats = extractor(args.pcapfile, args.identifier)
    source_pcap = pathlib.Path(args.pcapfile)
    stats.to_csv(source_pcap.parent / f'{source_pcap.stem}_{args.identifier}.csv')


if __name__ == '__main__':
    main()
