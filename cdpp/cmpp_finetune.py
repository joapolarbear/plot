
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from utils import *

fig_dir = os.path.join(os.path.dirname(__file__), "fig/cmpp_finetune")
os.makedirs(fig_dir, exist_ok=True)

barwidth = 0.2
font_size = 36

def _plot(mape_df, device):
    mape_df.set_index("Networks", inplace=True)
    method_names = list(mape_df.keys()) # ["Habitat", "TLP", "Ours"]
    networks = np.array(mape_df.index)
    # networks = np.array(["ResNet-18", "BERT-tiny"])

    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    x = np.arange(len(networks))
    for idx in range(len(method_names)):
        _method = method_names[idx]
        data_on_all_devices = np.array(mape_df.loc[networks, _method])
        bars = ax.bar(x + idx*barwidth, data_on_all_devices,
            width=barwidth, label=_method)
        for bar in bars:
            bar.set_hatch(marks[idx])
        for i, v in enumerate(data_on_all_devices):
            ax.text(x[i] + (idx-0.3)*barwidth, v + 10, f"{float(v):.1f}",
                fontsize=font_size * 0.8, rotation=90,
                # color='blue'
                # fontweight='bold'
            )
    ax.grid(axis="x")
    plt.ylabel("MAPE (%)", fontsize=font_size+2)
    plt.ylim(0, 1.8*np.max(mape_df.loc[networks, :]))
    plt.xticks(x + (len(method_names)/2-0.5)*barwidth, networks,
            fontsize=font_size-2, rotation=0)
    plt.xlabel("Target Network", fontsize=font_size)
    plt.yticks(fontsize=font_size-2)
    reduce_tick_num(5, type=int, low=0, high=1.3)
    plt.legend(fontsize=font_size-6)
    # legend = plt.legend(ncol=3, fontsize=font_size)

    # plt.subplots_adjust(left=0.1, bottom=0.15, right=0.88, top=0.95,
                        # wspace=0.2, hspace=0.4)
    plt.tight_layout()
    plt.savefig("{}/cmpp_finetune_{}.pdf".format(fig_dir,
        device), bbox_inches='tight')
    plt.close()

mape_t4 = pd.read_csv(os.path.join("data", "cmpp_finetune_t4.csv"))
mape_epyc = pd.read_csv(os.path.join("data", "cmpp_finetune_epyc.csv"))
_plot(mape_t4, "t4")
_plot(mape_epyc, "epyc")