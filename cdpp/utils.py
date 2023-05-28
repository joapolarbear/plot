import math
import numpy as np
import matplotlib.pyplot as plt

def fig_base(fig_num, row_first=True):
    if row_first:
        row_num = math.ceil(math.sqrt(fig_num))
        col_num = math.ceil(fig_num / row_num)
    else:
        col_num = math.ceil(math.sqrt(fig_num))
        row_num = math.ceil(fig_num / col_num)
    return row_num * 100 + col_num * 10

font_size = 36

def reduce_tick_num(num, axis="y", low=1, high=1, type=None):
    if axis == "y":
        locs, labels = plt.yticks()
    elif axis == "x":
        locs, labels = plt.xticks()
    else:
        raise ValueError(axis)
    _min = min(locs)
    _max = max(locs)
    _mid = (_min + _max) / 2
    _range = (_max - _min)
    low *= (_mid - 1.1 * _range / 2)
    high *= _max
    new_locs = np.arange(low, high, step=(high-low)/float(num))
    new_locs = new_locs.astype(type)
    # new_ticks = (new_locs / 1e4).astype(int)
    if axis == "y":
        plt.yticks(new_locs, fontsize=font_size-2)
    else:
        plt.xticks(new_locs, fontsize=font_size-2)