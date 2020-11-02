import numpy as np


class PacketScaler:
    def __init__(self, max_packet_len=1500):
        self.max_packet_len = max_packet_len

    def transform(self, packet_pairs):
        """
        :param packet_pairs: (N, 2), 0 -- packet_len, 1 -- IAT
        :return: transformed_packets (N, 2)
        """
        packet_pairs[:, 0] = packet_pairs[:, 0] / self.max_packet_len
        # avoids warning and -inf values. the scale here is in microseconds (?)
        zero_iats = np.isclose(packet_pairs[:, 1], 0.)
        packet_pairs[:, 1][zero_iats] = 0
        packet_pairs[:, 1][~zero_iats] = np.log10(packet_pairs[:, 1][~zero_iats])
        return packet_pairs

    def inverse_transform(self, packet_pairs):
        packet_pairs[:, 0] = packet_pairs[:, 0] * self.max_packet_len
        # to correctly rescale, we need to know which were initially zeros
        zero_iats = np.isclose(packet_pairs[:, 1], 0., atol=1e-3)
        packet_pairs[:, 1][zero_iats] = 0
        packet_pairs[:, 1][~zero_iats] = 10 ** packet_pairs[:, 1][~zero_iats]
        return packet_pairs
