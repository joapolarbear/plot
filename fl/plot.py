
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

fig_dir = os.path.join(os.path.dirname(__file__), "fig")
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

data_dir = os.path.join(os.path.dirname(__file__), "data")
method_names = []
all_df = []
for csv_file in sorted(os.listdir(data_dir)):
    if not csv_file.endswith(".csv"):
        continue
    method_names.append(csv_file.split(".csv")[0])
    df_data = pd.read_csv(os.path.join(data_dir, csv_file))
    # df_data.set_index("Time", inplace=True)
    all_df.append(df_data)

####################################################
#                      Plot figures
####################################################
AXIS2LABEL = {
    "_runtime": "Wall Time (s)",
    "_step": "Step",
    "Test/Loss": "Test Loss",
    "Test/Acc": "Test Acc. (%)",
    "Train/Loss": "Train Loss",
    "Train/Acc": "Train Acc. (%)",
}

FIXED_PLOT_KWARGS = {
    "Eff-FL": {"color": "r"}
}

def axis2data(df_data, axis):
    if "Acc" in axis:
        _data = df_data.loc[:, axis] * 100
    else:
        _data = df_data.loc[:, axis]
    return _data


def plot_figure(x_axis, y_axis, file_name, 
        ytick_kwarg={"num": 5, "high": 2, "type": int}):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    for idx, method in enumerate(method_names):
        df_data = all_df[idx]
        x_data = axis2data(df_data, x_axis)
        y_data = axis2data(df_data, y_axis)
        ax.plot(x_data, y_data, markersize=20,
                linewidth=3, label=method, **FIXED_PLOT_KWARGS.get(method, {}))

    ax.grid(axis="x")
    plt.ylabel(AXIS2LABEL[y_axis], fontsize=font_size+2)
    plt.xlabel(AXIS2LABEL[x_axis], fontsize=font_size)
    plt.yticks(fontsize=font_size-2)
    plt.xticks(fontsize=font_size-2)
    reduce_tick_num(**ytick_kwarg)
    plt.legend(fontsize=font_size-4, ncol=2)
    plt.tight_layout()
    plt.savefig("{}/{}.pdf".format(fig_dir, file_name), bbox_inches='tight')


plot_figure("_runtime", "Test/Loss", "time2test_loss", ytick_kwarg={"num": 5, "high": 2, "type": int})
plot_figure("_runtime", "Test/Acc", "time2test_acc", ytick_kwarg={"num": 5, "high": 2})
plot_figure("_runtime", "Train/Loss", "time2train_loss", ytick_kwarg={"num": 5, "high": 2, "type": int})
plot_figure("_runtime", "Train/Acc", "time2train_acc", ytick_kwarg={"num": 5, "high": 2})

plot_figure("_step", "Test/Loss", "step2test_loss", ytick_kwarg={"num": 5, "high": 2, "type": int})
plot_figure("_step", "Test/Acc", "step2test_acc", ytick_kwarg={"num": 5, "high": 2})
plot_figure("_step", "Train/Loss", "step2train_loss", ytick_kwarg={"num": 5, "high": 2, "type": int})
plot_figure("_step", "Train/Acc", "step2train_acc", ytick_kwarg={"num": 5, "high": 2})