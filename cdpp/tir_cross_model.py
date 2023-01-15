
import os
# import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

os.makedirs("fig/tir-cm", exist_ok=True)

mpl.rcParams['hatch.linewidth'] = 0.5

# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
# sns.set_theme(style="darkgrid", color_codes=True)
tips = sns.load_dataset("tips")

# plt.rcParams["font.sans-serif"] = "Simhei"
# marks = ["o","X","+","*","O","."]
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.3
font_size = 36

x_name = ["XGBoost", "Tiramisu", "Ours"]
devices = [
    "T4",
    "V100",
    "A100",
    "K80",
    "P100", 
    "HL-100"
]
x = np.arange(len(devices))

tir_time = {
    "MAPE (%)": np.array([
        [57.41, 77.615, 12.65],
        [51.76, 483.3, 14.84],
        [254.32, 444.6, 14.57],
        [40.87, 462.6, 12.89],
        [44.89, 520.02, 19.04],
        [75.78, 349.31, 26.77]
    ]),
    "Throughput in log\nscale(sample/s)": np.array([
        [-0.293423235, 1.089481203, 3.963337687],
        [-0.2835490008, 0.9816917873, 4.040566353],
        [-0.298517176, 0.9764416894, 4.031522259],
        [-0.305216441, 1.075692918, 4.045338076],
        [-0.3075144694, 0.9880235619, 4.032015806],
        [-0.2946772447, 0.9888912835, 4.032449884]
    ])
}

final_mse = None
for _key, _tir_time in tir_time.items():

    fig = plt.figure(figsize=(15, 6))
    ax = plt.subplot(111)
    for idx in range(len(x_name)):
        bars = ax.bar(
            x + idx*barwidth, _tir_time[:, idx],
            width=barwidth, label=x_name[idx])
        if _key.startswith("MAPE"):
            for i, v in enumerate(_tir_time[:, idx]):
                ax.text(x[i] + idx*barwidth - barwidth*0.3, v + 20, str(int(v)),
                    fontsize=font_size * 0.8, rotation=90,
                    # color='blue'
                    # fontweight='bold'
                )
    ax.grid(False)
    plt.ylabel(_key, fontsize=font_size+2)
    # import math
    # max_y_value = int(math.ceil(1.05*np.max(_tir_time)/100)*100)
    # ax.set_yticks(np.arange(0, max_y_value+1, int(max_y_value/4)))
    plt.ylim(0, 1.65*np.max(_tir_time))
    # ymajorLocator = MultipleLocator(int(1.1*np.max(_tir_time)/4//10*10))
    # ax.yaxis.set_major_locator(ymajorLocator) 
    plt.xticks(x + (len(x_name)/2-0.5)*barwidth, devices,
               fontsize=font_size-2, rotation=0)
    plt.xlabel("Devices", fontsize=font_size)
    plt.yticks(fontsize=font_size-2)
    plt.legend(ncol=3, fontsize=font_size)
    # legend = plt.legend(ncol=3, fontsize=font_size)

    # plt.subplots_adjust(left=0.1, bottom=0.15, right=0.88, top=0.95,
                        # wspace=0.2, hspace=0.4)
    plt.tight_layout()
    _key_replace = _key.split(" ")[0]
    plt.savefig("fig/tir-cm/{}.pdf".format(_key_replace), bbox_inches='tight')
    plt.close()
