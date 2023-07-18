
import sys, os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from help.utils import fig_base, reduce_tick_num

fig_dir = os.path.join(os.path.dirname(__file__), "fig/time2loss")
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

####################################################
#                      Read data
####################################################
# df_data = pd.read_csv(os.path.join("data", "finetune.csv"))
# df_data.set_index("N_sample_tasks", inplace=True)
# method_names = list(df_data.keys())
# sampled_num = sorted(df_data.index)
# print(df_data)

data_dir = "./data"
method_names = []
all_df = []
for csv_file in os.listdir(data_dir):
    if not csv_file.endswith(".csv"):
        continue
    method_names.append(csv_file.split(".csv")[0])
    df_data = pd.read_csv(os.path.join(data_dir, csv_file))
    # df_data.set_index("Time", inplace=True)
    all_df.append(df_data)

####################################################
#                      Plot figures
####################################################
fig = plt.figure(figsize=(12, 5))
ax = plt.subplot(111)

for idx, method in enumerate(method_names):
    df_data = all_df[idx]
    ax.plot(df_data.loc[:, "Time"], df_data.loc[:, "Loss"], markersize=20,
            linewidth=3, label=method)

ax.grid(axis="x")
plt.ylabel("Loss", fontsize=font_size+2)
plt.xlabel("Wall Time (s)", fontsize=font_size)
plt.yticks(fontsize=font_size-2)
plt.xticks(fontsize=font_size-2)
reduce_tick_num(5)
plt.legend(fontsize=font_size)
plt.tight_layout()
plt.savefig("{}/{}.pdf".format(fig_dir, "time2loss"), bbox_inches='tight')
