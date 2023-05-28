
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


from utils import fig_base, reduce_tick_num

fig_dir = os.path.join(os.path.dirname(__file__), "fig/ablation")
os.makedirs(fig_dir, exist_ok=True)

mpl.rcParams['hatch.linewidth'] = 0.5
# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
# sns.set_theme(style="darkgrid", color_codes=True)
tips = sns.load_dataset("tips")

# plt.rcParams["font.sans-serif"] = "Simhei"
# marks = ["o","X","+","*","O","."]
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.25
font_size = 36


df_data = pd.read_csv(os.path.join("data", "pe.csv"))
df_data.set_index("Device", inplace=True)
method_names = list(df_data.keys())
devices = sorted(df_data.index)
x = np.arange(len(devices))

print(df_data)

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

for idx in range(len(method_names)):
    _method = method_names[idx]
    data_on_all_devices = np.array(df_data.loc[devices, _method])
    bars = ax.bar(x + idx*barwidth, data_on_all_devices,
        width=barwidth, label=_method)
    for i, v in enumerate(data_on_all_devices):
        if v == 0:
            continue
        ax.text(x[i] + (idx-0.3)*barwidth, v+1, str(int(v)),
            fontsize=font_size, rotation=0,
            # color='blue'
            # fontweight='bold'
        )
ax.grid(axis="x")

plt.ylabel("MAPE (%)", fontsize=font_size+2)
plt.xlabel("Device", fontsize=font_size+2)
plt.yticks(fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.xticks(x + (len(method_names)/2-0.5)*barwidth, devices,
    fontsize=font_size, rotation=0)
reduce_tick_num(5, low=0, high=2, type=int)
plt.legend(ncol=2, fontsize=font_size)
plt.tight_layout()
plt.savefig("{}/{}.pdf".format(fig_dir, "pe"), bbox_inches='tight')
