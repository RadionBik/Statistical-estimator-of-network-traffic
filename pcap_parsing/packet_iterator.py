from typing import Tuple

import dpkt

from settings import logger


def packet_iterator(pcapfile, payload_only) -> Tuple[float, dpkt.ethernet.Ethernet, dpkt.ip.IP]:

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

        yield ts, eth, ip
