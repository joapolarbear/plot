
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

font_size = 36

df_data = pd.read_csv(os.path.join("data", "finetune.csv"))
df_data.set_index("N_sample_tasks", inplace=True)
method_names = list(df_data.keys())
sampled_num = sorted(df_data.index)

print(df_data)

fig = plt.figure(figsize=(12, 5))
ax = plt.subplot(111)

for i, method in enumerate(method_names):
    ax.plot(sampled_num, df_data.loc[sampled_num, method], markersize=marksize,
            linewidth=3, label=method, marker=linemarks[i])

ax.grid(axis="x")
plt.ylabel("Finetuning\nresult MAPE(%)", fontsize=font_size+2)
plt.xlabel("# of sampled tasks", fontsize=font_size)
plt.yticks(fontsize=font_size-2)
plt.xticks(fontsize=font_size-2)
reduce_tick_num(5)
plt.legend(fontsize=font_size)
plt.tight_layout()
plt.savefig("{}/{}.pdf".format(fig_dir, "finetune"), bbox_inches='tight')
