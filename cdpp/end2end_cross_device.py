
import os
# import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from utils import reduce_tick_num

os.makedirs("fig/end2end-cd", exist_ok=True)
# os.system("rm -rf fig/end2end-cd/*")

mpl.rcParams['hatch.linewidth'] = 0.5

# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
# sns.set_theme(style="darkgrid", color_codes=True)
tips = sns.load_dataset("tips")

# plt.rcParams["font.sans-serif"] = "Simhei"
# marks = ["o","X","+","*","O","."]
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.2
font_size = 36

column_names = ["Ground Truth", "Habitat", "CDMPP"]
row_names = [
    # "ResNet-50\\BS=1",
    "ResNet-50\nBS=4",
    # "ResNet-50\nBS=8",
    "Inception\nBS=8",
    "DCGAN\nBS=4",
    "Transfor-\nmer BS=8"
]
x = np.arange(len(row_names))

tir_time = {
    "Source:T4,Target:P100": np.array([
        # [29.4294281, 20.0272051, 33.73526472],
        [29.1389974, 22.21372248, 32.51237205],
        # [33.79794057, 27.00420771, 39.35006099],
        [60.4691925, 45.63072769, 70.56950545],
        [16.36102422, 8.607468405, 19.15828547],
        [79.79638163, 63.59388863, 94.05066886],
    ]),
    "Source:T4,Target:V100": np.array([
        # [30.43709819, 17.37647555, 30.62],
        [25.73649089, 16.79324395, 26.29],
        # [34.1810023, 18.56355602, 35.43],
        [60.44047038, 34.90898648, 64.20],
        [16.64358393, 7.313034538, 17.95],
        [84.04999288, 60.03894419, 89.63],
    ]),
    "Source:V100,Target:P100": np.array([
        # [29.4294281, 26.23348307, 32.49],
        [29.1389974, 39.28034011, 32.57],
        # [33.79794057, 58.20243493, 36.40],
        [60.4691925, 87.57110272, 60.99],
        [16.36102422, 11.9969674, 16.61],
        [79.79638163, 90.5076854, 88.72],
    ])
}

max_error = 0
max_error_daydream = 0

fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(111)
_tir_time = list(tir_time.values())[0]
for idx in range(len(column_names)):
    bars = ax.bar(
        x + idx*barwidth, _tir_time[:, idx],
        width=barwidth, label=column_names[idx])
    # for bar in bars:
    #     bar.set_hatch(marks[idx])
ax.plot(x + barwidth, _tir_time[:, 1], '-o', color='red',
             linewidth=6, markersize=20, label="CDMPP")

# plt.xlabel(title)
plt.yticks(fontsize=font_size)
# legend = plt.legend(ncol=4, fontsize=font_size*20, frameon=False)
label_params = ax.get_legend_handles_labels()
figl, axl = plt.subplots(figsize=(40, 2))
axl.axis(False)

label_params = list(label_params)
for _list in label_params:
    _list.append(_list.pop(0))

axl.legend(*label_params,
    ncol=5, 
    loc="center", 
    bbox_to_anchor=(0.5, 0.5), 
    frameon=False,
    fontsize=font_size*2,
    # prop={"size":50}
    )
figl.savefig("fig/end2end-cd/legend.pdf")
plt.close()

def plot_group_bar(_data, row_names, column_names, save_name, xaxis_name):
    base = _data[:, 0].reshape(_data.shape[0], 1)
    mape = 100 * np.abs(_data - base) / base
    # max_error = max(max_error, max((mape[:, 2] - mape[:, 1]) / mape[:, 1]))
    # max_error = max(max_error, max((mape[:, 2] - mape[:, 1]) / mape[:, 1]))
    xaxis = np.arange(len(row_names))
    # if "habana" in save_name:
    #     fig = plt.figure(figsize=(16, 5))
    # else:
    #     fig = plt.figure(figsize=(12, 5))
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    for idx in range(len(column_names)):
        bars = ax.bar(
            xaxis + idx*barwidth, _data[:, idx],
            width=barwidth, label=column_names[idx])
        # for bar in bars:
        #     bar.set_hatch(marks[idx])
    ax.grid(False)
    plt.ylabel("Time Cost (ms)", fontsize=font_size+2)
    plt.xlabel(xaxis_name, fontsize=font_size+2)
    if "habana" in save_name:
        plt.xticks(xaxis + (len(column_names)/2-0.5)*barwidth, row_names,
               fontsize=font_size-10, rotation=0)
    else:
        plt.xticks(xaxis + (len(column_names)/2-0.5)*barwidth, row_names,
               fontsize=font_size-10, rotation=0)
        
    plt.ylim(0, 1.2*np.max(_data))
    reduce_tick_num(5, low=0)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Prediction Error (%)', fontsize=font_size)  # we already handled the x-label with ax1
    ax2.plot(xaxis + barwidth, mape[:, -1], '-o', color='red',
             linewidth=3, markersize=10)
    ax2.tick_params(axis='y')
    for label in ax2.yaxis.get_majorticklabels():
        label.set_fontsize(font_size+2)
        # label.set_fontname('courier')
    plt.ylim(0, 1.2*np.max(mape[:, -1]))
    reduce_tick_num(5, low=0, type=int)

    plt.tight_layout()
    plt.savefig("fig/end2end-cd/{}.pdf".format(save_name), bbox_inches='tight')


for _key, _tir_time in tir_time.items():
    plot_group_bar(_tir_time, row_names, column_names, "_".join(_key.split(" ")), "Networks")

### Habana's results
row_names = [
    "ResNet\n-50(1)",
    "ResNet\n-50(4)",
    "ResNet\n-50(8)",
    "BERT\nBase(1)",
    "BERT\nBase(4)",
    "BERT\nBase(8)",
]

data = np.array([
    [1.015, 0, 1.510021688],
    [0.216, 0, 0.3792396067],
    [0.374, 0, 0.6340079261],
    [5.412, 0, 7.531204057],
    [5.46, 0, 8.703897827],
    [5.568, 0, 9.744085006],
])

plot_group_bar(data, row_names, column_names, "habana", "Networks")


