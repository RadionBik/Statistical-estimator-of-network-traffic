import logging
import re
import socket
from enum import Enum

from dpkt.compat import compat_ord

logger = logging.getLogger(__name__)


def mod_addr(mac):
    """
    replaces : and . with _ for addresses to enable saving to a disk
    """
    return mac.replace(':', '_').replace('.', '_').replace(' ', '_')


def ip_to_string(inet):
    """Convert inet object to a string
        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def is_mac_addr(string):
    if re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", string.lower()):
        return True
    else:
        return False


def is_ip_addr(string):
    try:
        socket.inet_aton(string)
        return True
    except socket.error:
        return False


def is_ip_port(string):
    if re.match(
            "6553[0-6]|655[0-2][0-9]|65[0-4][0-9]{2}|6[0-4][0-9]{3}|[1-5][0-9]{4}|0?[1-9][0-9]{3}|0?0?[1-9][0-9]{"
            "2}|0?0?0?[1-9][0-9]|0{0,4}[1-9]",
            string):
        return True
    else:
        return False


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


class TrafficObjects(Enum):
    MAC = 'MAC'
    IP = 'IP'
    FLOW = 'flow'
