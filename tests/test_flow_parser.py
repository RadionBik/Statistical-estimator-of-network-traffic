
from pcap_parsing.flow_level import extract_flow_stats
from pcap_parsing.parsed_fields import select_features


def test_parser(pcap_file):
    flow = 'UDP 10.1.3.143:5000 10.1.6.18:2006'
    parsed = extract_flow_stats(pcap_file, flow)
    packets, from_idx = select_features(parsed)

    assert from_idx.sum() == 236
    assert packets.shape[0] == 465
    assert int(packets[from_idx, 0].sum()) == 66080
