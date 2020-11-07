from typing import NamedTuple, Optional

import dpkt
import pandas as pd

from pcap_parsing.calc_stats import form_df
from pcap_parsing.packet_iterator import packet_iterator
from pcap_parsing.parsed_fields import ParsedFields as PF
from settings import logger
from utils import ip_to_string


class FlowTuple(NamedTuple):
    proto: str
    src_tuple: str
    dst_tuple: str


def _extract_5tuple_from_packet(packet: dpkt.ip.IP) -> Optional[FlowTuple]:

    if packet.p == 6:
        proto = 'TCP'
    elif packet.p == 17:
        proto = 'UDP'
    else:
        logger.warning(f'unsupported packet: {packet}')
        return None
    return FlowTuple(
        proto,
        f'{ip_to_string(packet.src)}:{packet.data.sport}',
        f'{ip_to_string(packet.dst)}:{packet.data.dport}',
    )


def _extract_5tuple_from_str(flow: str):
    return FlowTuple(*flow.split())


def extract_flow_stats(
        pcapfile,
        flow: str,
        payload_only=False,
) -> pd.DataFrame:
    """
    extract_flow_stats() uses the dpkt lib to extract packet features of the flow.
    'flow' is a string of format 'TCP|UDP SRC_IP:SRC_PORT DST_IP:DST_PORT'

    """

    target_flow = _extract_5tuple_from_str(flow)

    stats = {
        PF.is_source: [],
        PF.ps: [],
        PF.ts: []
    }
    for ts, eth, ip in packet_iterator(pcapfile, payload_only):
        curr_flow = _extract_5tuple_from_packet(ip)
        if not curr_flow:
            continue
        if curr_flow.proto != target_flow.proto:
            continue

        if curr_flow.src_tuple == target_flow.src_tuple and curr_flow.dst_tuple == target_flow.dst_tuple:
            stats[PF.ps].append(len(ip))
            stats[PF.ts].append(ts)
            stats[PF.is_source].append(True)
        elif curr_flow.src_tuple == target_flow.dst_tuple and curr_flow.dst_tuple == target_flow.src_tuple:
            stats[PF.ps].append(len(ip))
            stats[PF.ts].append(ts)
            stats[PF.is_source].append(False)

    return form_df(stats)

