
import os
# import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

if not os.path.exists("fig/replay"):
    os.mkdir("fig/replay")
# os.system("rm -rf fig/replay/*")

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

x_name = ["Ground Truth", "dPRO", "Daydream"]
configs = [
    "HVD+TF",
    "HVD+MX",
    "BPS+TF",
]
x = np.arange(len(configs))

iter_time = {
    "ResNet50+TCP": np.array([
        [149.900667, 147.672557, 110.568491],
        [129.6621, 130.661992, 103.945101],
        # [159.565992, 137.730744, 131.800293],
        [152.024125, 144.231128, 131.800293]
    ]),
    "ResNet50+RDMA": np.array([
        [138.644111, 142.31318, 109.186041],
        [111.2145, 109.545927, 100.510745],
        # [123.904031, 121.373756, 118.420341],
        ### barrier
        [116.151836, 117.749927, 118.420341]
    ]),
    "BERT Base+TCP": np.array([
        [551.82822, 519.15503, 355.863745], 
        [452.1809, 439.683609, 274.140731],
        # [570.86468, 503.970155, 400.534164],
        [545.058222, 513.983792, 400.534164]
    ]),
    "BERT Base+RDMA": np.array([
        [459.616222, 453.83346, 345.679386],
        [311.2835, 292.813982, 270.489931],
        # [471.485102, 414.402197, 396.490628]
        [499.669979, 465.116912, 396.490628]
    ]),
    "VGG16+TCP": np.array([
        [457.319333, 451.519756, 136.24037],
        [409.3765, 412.67577, 146.519312],
        [431.146258, 449.179687, 148.294287],  
    ]),
    "VGG16+RDMA": np.array([
        [227.036875, 219.808715, 139.084833],
        [219.4006, 219.93695, 146.466721],
        [232.83875, 230.381764, 152.170895]
    ]),
    "InceptionV3+TCP": np.array([
        [137.427889, 142.892577, 100.749748],
        [118.0224, 117.169653, 104.275933],
        # [137.650344, 122.680793, 122.285834],
        [134.852951, 124.937119, 122.285834]
    ]),
    "InceptionV3+RDMA": np.array([
        [130.983111, 138.94257, 98.911402],
        [116.2637, 113.208896, 101.794944],
        # [122.834844, 113.02411, 113.156396]
        [115.772117, 110.8646, 113.156396]
    ])
}

max_error = 0
max_error_daydream = 0

fig = plt.figure(figsize=(9, 5))
ax = plt.subplot(111)
_iter_time = iter_time["ResNet50+TCP"]
for idx in range(len(x_name)):
    bars = ax.bar(
        x + idx*barwidth, _iter_time[:, idx],
        width=barwidth, label=x_name[idx])
    # for bar in bars:
    #     bar.set_hatch(marks[idx])
ax.plot(x + barwidth, _iter_time[:, 1], '-o', color='red',
             linewidth=6, markersize=20, label="dPRO Prediction Error")

# plt.xlabel(title)
plt.yticks(fontsize=font_size)
# legend = plt.legend(ncol=4, fontsize=font_size*20, frameon=False)
label_params = ax.get_legend_handles_labels()
figl, axl = plt.subplots(figsize=(70, 2))
axl.axis(False)


label_params = list(label_params)
for _list in label_params:
    _list.append(_list.pop(0))

axl.legend(*label_params,
    ncol=4, 
    loc="center", 
    bbox_to_anchor=(0.5, 0.5), 
    frameon=False,
    fontsize=font_size*2,
    # prop={"size":50}
    )
figl.savefig("fig/replay/legend.pdf")

final_mse = None
for _key, _iter_time in iter_time.items():
    base = _iter_time[:, 0].reshape(_iter_time.shape[0], 1)
    mse = 100 * np.abs(_iter_time - base) / base
    if final_mse is None:
        final_mse = mse
    else:
        final_mse = np.concatenate((final_mse, mse), axis=0)
    # max_error = max(max_error, max((mse[:, 2] - mse[:, 1]) / mse[:, 1]))
    # max_error = max(max_error, max((mse[:, 2] - mse[:, 1]) / mse[:, 1]))

    fig = plt.figure(figsize=(9, 5))
    ax = plt.subplot(111)
    for idx in range(len(x_name)):
        bars = ax.bar(
            x + idx*barwidth, _iter_time[:, idx],
            width=barwidth, label=x_name[idx])
        # for bar in bars:
        #     bar.set_hatch(marks[idx])
    ax.grid(False)
    plt.ylabel("Iteration Time (ms)", fontsize=font_size+2)
    import math
    max_y_value = int(math.ceil(1.05*np.max(_iter_time)/100)*100)
    ax.set_yticks(np.arange(0, max_y_value+1, int(max_y_value/4)))
    # plt.ylim(0, 1.2*np.max(_iter_time))
    # ymajorLocator = MultipleLocator(int(1.1*np.max(_iter_time)/4//10*10))
    # ax.yaxis.set_major_locator(ymajorLocator) 
    plt.xticks(x + (len(x_name)/2-0.5)*barwidth, configs,
               fontsize=font_size, rotation=0)
    # plt.xlabel(title)
    plt.yticks(fontsize=font_size)
    # legend = plt.legend(ncol=3, fontsize=font_size)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('dPRO Error (%)', fontsize=font_size)  # we already handled the x-label with ax1
    ax2.plot(x + barwidth, mse[:, 1], '-o', color='red',
             linewidth=3, markersize=10)
    ax2.tick_params(axis='y')
    for label in ax2.yaxis.get_majorticklabels():
        label.set_fontsize(font_size+2)
        # label.set_fontname('courier')
    # plt.ylim(0, 20)
    ax2.set_yticks(np.arange(0, 25, 5))

    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.88, top=0.95,
                        wspace=0.2, hspace=0.4)
    _key_replace = "_".join(_key.split(" "))
    plt.savefig("fig/replay/replay_{}.pdf".format(_key_replace), bbox_inches='tight')

    # fig = plt.figure(figsize=(12, 5))
    # ax = plt.subplot(111)
    # for idx in range(len(x_name)):
    #     bars = ax.bar(
    #         x + idx*barwidth, mse[:, idx], width=barwidth, label=x_name[idx])
    #     for bar in bars:
    #         bar.set_hatch(marks[idx])
    # plt.ylabel("MAPE (%)", fontsize=font_size)
    # plt.ylim(0, 1.4*np.max(mse))
    # plt.xticks(x + (len(x_name)/2)*barwidth, configs,
    #            fontsize=font_size, rotation=0)
    # # plt.xlabel(title)
    # plt.yticks(fontsize=font_size)
    # plt.legend(ncol=3, fontsize=font_size)

    # plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95,
    #                     wspace=0.2, hspace=0.4)
    # _key_replace = "_".join(_key.split(" "))
    # plt.savefig("fig/replay/replay_error_{}".format(_key_replace))

print("max_error: {}".format(max_error))
# plt.title(title, fontsize=font_size)


### Large scale data with 128 GPUs
_iter_time = np.array([
    [146.2487526, 141.924559, 113.766854],
    [114.963532, 112.8880371, 112.838],
    [525.4864351, 529.1857233, 137.659056],
    [232.8874756, 234.526943, 138.2075],
    [159.0888947, 154.3139158, 98.772111],
    [147.6898751, 153.6753495, 97.447889],
    [841.276743, 794.041738, 347.878962],
    [1205.85906, 1215.724106, 348.226841],
])
base = _iter_time[:, 0].reshape(_iter_time.shape[0], 1)
mse = 100 * np.abs(_iter_time - base) / base
final_mse = np.concatenate((final_mse, mse), axis=0)

print("ave", np.average(final_mse, axis=0))
print("max", np.max(final_mse, axis=0))