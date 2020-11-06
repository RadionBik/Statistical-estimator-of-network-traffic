import logging

import dpkt
import pandas as pd

from pcap_parsing.parsed_fields import ParsedFields as PF
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
        PF.is_source: [],
        PF.ps: [],
        PF.ts: []
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
            stats[PF.ps].append(len(ip))
            stats[PF.ts].append(ts)
            stats[PF.is_source].append(True)
        elif dst_id == host:
            stats[PF.ps].append(len(ip))
            stats[PF.ts].append(ts)
            stats[PF.is_source].append(False)

    stats = pd.DataFrame(stats)
    # calc IAT and convert from seconds to ms
    iat_from = stats[stats[PF.is_source]][PF.ts].diff().fillna(0) * 1000
    iat_to = stats[~stats[PF.is_source]][PF.ts].diff().fillna(0) * 1000

    stats[PF.iat] = pd.concat([iat_to, iat_from]).sort_index().round(0)
    return stats.reset_index(drop=True)


if __name__ == '__main__':
    import settings
    source_pcap = settings.BASE_DIR / 'traffic_dumps/iot_amazon_echo.pcap'
    extr_stats = extract_host_stats(
        source_pcap,
        '44:65:0d:56:cc:d3',
    )
    print(extr_stats)
    extr_stats.to_csv(source_pcap.as_posix() + '.csv')
