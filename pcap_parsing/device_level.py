import logging

import dpkt
import pandas as pd
from utils import TrafficObjects, mac_addr, ip_to_string, is_mac_addr, is_ip_addr

logger = logging.getLogger(__name__)


def extract_id_from_packet(packet: dpkt.ethernet.Ethernet, identifier_type: TrafficObjects):
    if identifier_type == TrafficObjects.MAC:
        return mac_addr(packet.src), mac_addr(packet.dst)
    elif identifier_type == TrafficObjects.IP:
        return ip_to_string(packet.data.src), ip_to_string(packet.data.dst)
    else:
        raise NotImplementedError


def extract_host_stats(
        pcapfile,
        host: str,
        payload_only=False,
) -> pd.DataFrame:
    """
    extract_host_stats() uses the dpkt lib to extract packet features of a host.
    'host' specifies the host by MAC addresses or IP

    """

    # create the layered dict

    if is_mac_addr(host):
        identifier_type = TrafficObjects.MAC
    elif is_ip_addr(host):
        identifier_type = TrafficObjects.IP
    else:
        raise ValueError

    stats = {
        'is_client': [],
        'PS': [],
        'TS': []
    }

    packet_counter = 0
    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        if packet_counter % 100000 == 0 and packet_counter > 0:
            logger.info('Processed {} packets...'.format(packet_counter))
        eth = dpkt.ethernet.Ethernet(raw)
        ip = eth.data
        packet_counter += 1

        # skip ARP, ICMP, etc.
        if not isinstance(ip, dpkt.ip.IP):
            continue
        # filter out segments and datagrams without payload (e.g. SYN, SYN/ACK, etc.)
        if payload_only and len(ip.data.data) == 0:
            continue

        src_id, dst_id = extract_id_from_packet(eth, identifier_type)

        if src_id == host:
            stats['PS'].append(len(ip))
            stats['TS'].append(ts)
            stats['is_client'].append(True)
        elif dst_id == host:
            stats['PS'].append(len(ip))
            stats['TS'].append(ts)
            stats['is_client'].append(False)

    stats = pd.DataFrame(stats)
    iat_from = stats[stats['is_client']]['TS'].diff().fillna(0)
    iat_to = stats[~stats['is_client']]['TS'].diff().fillna(0)

    stats['IAT, sec'] = pd.concat([iat_to, iat_from]).sort_index()
    return stats


if __name__ == '__main__':
    import settings
    source_pcap = settings.BASE_DIR / 'traffic_dumps/iot_amazon_echo.pcap'
    extr_stats = extract_host_stats(
        source_pcap,
        '44:65:0d:56:cc:d3',
    )
    print(extr_stats)
    extr_stats.to_csv(source_pcap.as_posix() + '.csv')
