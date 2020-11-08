import logging

import dpkt
import pandas as pd

from pcap_parsing.calc_stats import form_df
from pcap_parsing.packet_iterator import packet_iterator
from pcap_parsing.parsed_fields import ParsedFields as PF
from pcap_parsing.ip_utils import TrafficObjects, mac_addr, ip_to_string, is_mac_addr, is_ip_addr

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

    for ts, eth, ip in packet_iterator(pcapfile, payload_only):
        src_id, dst_id = extract_id_from_packet(eth, identifier_type)

        if src_id == host:
            stats[PF.ps].append(len(ip))
            stats[PF.ts].append(ts)
            stats[PF.is_source].append(True)
        elif dst_id == host:
            stats[PF.ps].append(len(ip))
            stats[PF.ts].append(ts)
            stats[PF.is_source].append(False)

    return form_df(stats)

