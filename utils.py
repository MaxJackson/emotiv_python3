# -*- coding: utf-8 -*-
# vim:set et ts=4 sw=4:
#
# Based on work done by Ozan Çağlayan <ocaglayan@gsu.edu.tr>, 2012. He is a good person.
# Forked and ported to Python 3 by Max Jackson <max_jackson@knights.ucf.edu>, December 2015.

import numpy as np
import os
import time

def check_packet_drops(seq_numbers):
    """Checks for dropped packets"""
    lost = []
    for seq in xrange(len(seq_numbers) - 1):
        cur = int(seq_numbers[seq])
        _next = int(seq_numbers[seq + 1])
        if ((cur + 1) % 128) != _next:
            lost.append((cur + 1) % 128)
    return lost

def get_level(raw_data, bits):
    """Returns signal level from raw_data frame."""
    level = 0
    for i in range(13, -1, -1):
        level <<= 1
        b, o = (bits[i] / 8) + 1, bits[i] % 8
        b = int(b)
        level |= (ord(str(raw_data[b])[0]) >> o) & 1
    return 0.51*level
