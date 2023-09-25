
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from utils import *

fig_dir = os.path.join(os.path.dirname(__file__), "fig/cdpp_tir")
os.makedirs(fig_dir, exist_ok=True)

barwidth = 0.25
font_size = 36

########################### MAPE ##############################
mape_data = pd.read_csv(os.path.join("data", "cdpp_tir_mape.csv"))
mape_data.set_index("Devices", inplace=True)

method_names = list(mape_data.keys()) # ["Habitat", "TLP", "Ours"]
devices = np.array(mape_data.index)

def _plot(fig_name: str, pd_data: pd.DataFrame, st_ed, sub_fig_id):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    st, ed = st_ed
    _devices = devices[np.arange(st, ed)]
    x = np.arange(len(_devices))
    for idx in range(len(method_names)):
        _method = method_names[idx]
        data_on_all_devices = np.array(pd_data.loc[_devices, _method])
        bars = ax.bar(x + idx*barwidth, data_on_all_devices,
            width=barwidth, label=_method)
        for bar in bars:
            bar.set_hatch(marks[idx])
        if fig_name.startswith("MAPE"):
            for i, v in enumerate(data_on_all_devices):
                if v == 0:
                    continue
                ax.text(x[i] + (idx-0.3)*barwidth, v + 10, str(int(v)),
                    fontsize=font_size * 0.8, rotation=90,
                    # color='blue'
                    # fontweight='bold'
                )
    ax.grid(axis="x")
    plt.ylabel(fig_name, fontsize=font_size+2)
    # import math
    # max_y_value = int(math.ceil(1.05*np.max(_tir_time)/100)*100)
    # ax.set_yticks(np.arange(0, max_y_value+1, int(max_y_value/4)))
    plt.ylim(0, 1.65*np.max(pd_data.loc[_devices, :]))
    # ymajorLocator = MultipleLocator(int(1.1*np.max(_tir_time)/4//10*10))
    # ax.yaxis.set_major_locator(ymajorLocator) 
    plt.xticks(x + (len(method_names)/2-0.5)*barwidth, devices[st:ed],
            fontsize=font_size-2, rotation=0)
    plt.xlabel("Target Device", fontsize=font_size)
    plt.yticks(fontsize=font_size-2)
    reduce_tick_num(5, low=0, high=1.3, type=int)
    plt.legend(ncol=2, fontsize=font_size-6)
    # legend = plt.legend(ncol=3, fontsize=font_size)

    # plt.subplots_adjust(left=0.1, bottom=0.15, right=0.88, top=0.95,
                        # wspace=0.2, hspace=0.4)
    plt.tight_layout()
    _key_replace = fig_name.split(" ")[0]
    plt.savefig("{}/{}_{}.pdf".format(fig_dir, _key_replace, 
            sub_fig_id), bbox_inches='tight')
    plt.close()

_plot("MAPE (%)", mape_data, (0, 5), 0)
_plot("MAPE (%)", mape_data, (5, 9), 1)
