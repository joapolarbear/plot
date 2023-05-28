
import os
# import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from utils import reduce_tick_num

os.makedirs("fig/end2end-cm", exist_ok=True)
# os.system("rm -rf fig/end2end-cm/*")

mpl.rcParams['hatch.linewidth'] = 0.5

# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
# sns.set_theme(style="darkgrid", color_codes=True)
tips = sns.load_dataset("tips")

# plt.rcParams["font.sans-serif"] = "Simhei"
# marks = ["o","X","+","*","O","."]
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.22
font_size = 36

column_names = ["Ground Truth", "XGBoost", "Tiramisu", "CDMPP"]
row_names = [
    "T4",
    "V100",
    "A100",
    "EPYC",
    # "P100",
]
x = np.arange(len(row_names))

tir_time = {
    "ResNet-50(1)": np.array([
        [175.00504, 255.8372041, 253.5493811, 187.448],
        [196.210745, 248.3647383, 799.559789, 212.0826803],
        [194.9012648, 514.8944657, 1060.215571, 215.8999819],
        [651.3704429, 835.5116171, 3589.510492, 709.5742603],
        # [300.8287629, 373.8343374, 1222.720392, 339.8169114],
    ]),
    "ResNet-50(4)": np.array([
        [198.1819, 286.318968, 324.055909, 190.36],
        [207.4652966, 282.2909338, 967.3403779, 234.9072438],
        [197.2925151, 624.8606449, 812.6060442, 222.4040226],
        [667.697267, 836.8035326, 2851.732555, 735.8178977],
        # [302.4443561, 393.5747092, 1848.966461, 349.2627403],
    ]),
    "ResNet-50(8)": np.array([
        [286.0293, 24.882488, 400.1491795, 267.17],
        [313.3044606, 20.4947453, 1700.776722, 337.1323856],
        [286.9266111, 61.609916, 1491.016143, 325.8666581],
        [952.9218473, 227.371756, 4599.508863, 1054.822682],
        # [461.566367, 99.377137, 2012.367014, 548.5597178],
    ]),
    "InceptionV3(1)": np.array([
        [829.956, 1140.094934, 1369.092042, 817.54],
        [799.5544654, 1201.305266, 4351.208078, 912.6003303],
        [838.2931112, 2629.783735, 3255.045694, 959.3138511],
        [3547.203921, 4779.628916, 14562.3598, 3810.311035],
        # [1357.954846, 1875.584272, 7262.650041, 1601.3538],
    ]),
    "BERT Base(1)": np.array([
        [499.6468358, 674.0647417, 714.3578115, 534.0902336],
        [521.2542, 668.1383451, 1824.399656, 597.1141112],
        [523.8977452, 1310.550614, 2844.427604, 569.7503971],
        [2151.402671, 2706.504784, 7663.655476, 2355.425039],
        # [851.3043024, 1231.835698, 4777.842837, 1012.793059],
    ]),
    "BERT Base(4)": np.array([
        [271.2892325, 365.1498349, 476.2553096, 296.8680112],
        [253.7262, 356.6292914, 1037.556483, 289.8806233],
        [237.0362396, 558.1573123, 1190.592095, 258.7866288],
        [847.4667479, 1059.073645, 4304.042909, 923.4012201],
        # [383.891531, 528.7043173, 1959.230476, 443.8191725],
    ])
}

max_error = 0
max_error_daydream = 0

### Plot legends separately
fig = plt.figure(figsize=(9, 5))
ax = plt.subplot(111)
_tir_time = list(tir_time.values())[0]
for idx in range(len(column_names)):
    bars = ax.bar(
        x + idx*barwidth, _tir_time[:, idx],
        width=barwidth, label=column_names[idx])
    # for bar in bars:
    #     bar.set_hatch(marks[idx])
# ax.plot(x + barwidth, _tir_time[:, -3],
#              linewidth=6, markersize=20, label=column_names[-3])
# ax.plot(x + barwidth, _tir_time[:, -2],
#              linewidth=6, markersize=20, label=column_names[-2])
ax.plot(x + barwidth, _tir_time[:, -1], '-o', color='red',
             linewidth=6, markersize=20, label=column_names[-1])
    
# plt.xlabel(title)
plt.yticks(fontsize=font_size)
# legend = plt.legend(ncol=4, fontsize=font_size*20, frameon=False)
label_params = ax.get_legend_handles_labels()
figl, axl = plt.subplots(figsize=(50, 2))
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
figl.savefig("fig/end2end-cm/legend.pdf")


def plot_group_bar(_data, row_names, column_names, save_name, xaxis_name):
    base = _data[:, 0].reshape(_data.shape[0], 1)
    mape = 100 * np.abs(_data - base) / base
    # max_error = max(max_error, max((mape[:, 2] - mape[:, 1]) / mape[:, 1]))
    # max_error = max(max_error, max((mape[:, 2] - mape[:, 1]) / mape[:, 1]))
    xaxis = np.arange(len(row_names))
    if "habana" in save_name:
        fig = plt.figure(figsize=(14, 6))
    else:
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
               fontsize=font_size-4, rotation=0)
    else:
        plt.xticks(xaxis + (len(column_names)/2-0.5)*barwidth, row_names,
               fontsize=font_size, rotation=0)
        
    plt.ylim(0, 1.2*np.max(_data))
    reduce_tick_num(5, low=0)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Prediction Error (%)', fontsize=font_size)  # we already handled the x-label with ax1
    # ax2.plot(xaxis + barwidth, mape[:, -3],
    #          linewidth=3, markersize=10)
    # ax2.plot(xaxis + barwidth, mape[:, -2],
    #          linewidth=3, markersize=10)
    ax2.plot(xaxis + barwidth, mape[:, -1], '-o', color='red',
             linewidth=3, markersize=10)
    ax2.tick_params(axis='y')
    for label in ax2.yaxis.get_majorticklabels():
        label.set_fontsize(font_size+2)
        # label.set_fontname('courier')
    plt.ylim(0, 1.2*np.max(mape[:, -1]))
    reduce_tick_num(5, low=0)

    plt.tight_layout()
    plt.savefig("fig/end2end-cm/{}.pdf".format(save_name), bbox_inches='tight')

for _key, _tir_time in tir_time.items():
    plot_group_bar(_tir_time, row_names, column_names, "_".join(_key.split(" ")), "Devices")

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
    [1.015,	1.56650058, 3.953101158, 1.15963693],
    [0.216,	0.320005927, 0.648448837, 0.2624160732],
    [0.374,	0.6347236451, 1.301384088, 0.4445724538],
    [5.412,	8.671341188, 17.54843386, 6.738998461],
    [5.46,	9.04603098, 15.04737393, 6.810274279],
    [5.568,	8.119463644, 23.75258667, 6.389355137],
])

if __name__ == '__main__':
    plot_group_bar(data, row_names, column_names, "habana", "Target Network")

