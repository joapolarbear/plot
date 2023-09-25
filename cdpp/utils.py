import math
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")

marks = ["/", "-", "\\", "x", "+", "."]
plt.rcParams['hatch.linewidth'] = 1.5
linemarks = ["o", 'X', 'd', 'v', '*', 's', 'p', '^', 'h', 'P']
marksize = 10

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
linewidth = 1.5
linestyle_str = [
    # Named
	('solid', 'solid'),      # Same as (0, ()) or '-'
	('dashed', 'dashed'),    # Same as '--'
 	('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
	('dashdot', 'dashdot'),  # Same as '-.'

	# Parameterized
 	('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
  	('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
	# ('loosely dotted',        (0, (1, 10))),
	# ('dotted',                (0, (1, 1))),
	# ('densely dotted',        (0, (1, 1))),
	('long dash with offset', (5, (10, 3))),
	# ('loosely dashed',        (0, (5, 10))),
	# ('dashed',                (0, (5, 5))),
	('densely dashed',        (0, (5, 1))),

	# ('loosely dashdotted',    (0, (3, 10, 1, 10))),
	('dashdotted',            (0, (3, 5, 1, 5))),
	# ('densely dashdotted',    (0, (3, 1, 1, 1))),

	
	('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
 ]

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