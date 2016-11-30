#!/usr/bin/env python
import numpy as np
__author__ = 'Wolf Vos, Casper Thuis, Alexander van Someren, Jeroen Rooijmans'


def get_index_from_click_pattern(click_pattern, location=1):
    """
    Function that returns index given a click pattern
    """
    index = (location) * 1024
    index += int(''.join([str(i) for i in click_pattern]), 2)
    return index


def get_click_pattern_from_index(index):
    """
    Function that returns click pattern from given index
    """
    loc = index / 1024
    index -= (loc * 1024)
    click_pattern = map(int, np.binary_repr(index, width=10))
    return click_pattern, loc
