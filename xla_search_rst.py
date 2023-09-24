import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from help.utils import *

if not os.path.exists("fig/opfs"):
    os.mkdir("fig/opfs")
os.system("rm -rf fig/opfs/*")
# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")

x_name = np.array(["No Fusion", "TF XLA", "dPRO_OPFS"])

_filter = np.array([0, 1, 2])

configs = [
    "hvd+TCP",
    "hvd+RDMA",
    "BytePS+TCP",
    "BytePS+RDMA"
]

USE_THROUGHPUT = True
BATCH_SIZE = 32

iter_time = {
    "ResNet50": np.array([
        [187.94175, 230.108086, 183.1359463],
        [134.5339617, 149.4408847, 126.422159],
        [196.0744385, 267.215776, 179.6238305],
        [106.7313433, 131.8673533, 100.0797827],
    ]),
    "VGG16": np.array([
        [412.6467783, 491.9420797, 423.568797],
        [261.206738, 222.943266, 231.733346],
        [523.1078147, 588.9855147, 510.751112],
        [221.690941, 291.430354, 203.2869497],
    ]),
    "InceptionV3": np.array([
        [169.061017, 206.3683907, 167.4394923],
        [119.9843963, 149.7025093, 118.847235],
        [111.2059433, 172.1465507, 109.173385],
        [92.544087, 113.9632467, 91.95433433],
    ]),
    "BERT Base": np.array([
        [538.362813, 566.147264, 527.4501007],
        [371.276363, 337.6628557, 328.1833887],
        [649.079728, 741.331482, 591.633844],
        [1000000, 1000000, 1000000],
    ])
}

max_speedup = 0
fontsize = 24
barwidth = 0.2
for _key, _iter_time in iter_time.items():
    for i in range(len(_iter_time)):
        max_speedup = max(
            max_speedup, 100 * (_iter_time[i][1] - min(_iter_time[i][2:4])) / _iter_time[i][1])

    base = _iter_time[:, 0].reshape(len(configs), 1)
    speedup = 100 * (base - _iter_time) / base

    yaxis_data = 1000 * BATCH_SIZE / _iter_time if USE_THROUGHPUT else _iter_time
    yaxis_name = "Throughput\n(samples/s)" if USE_THROUGHPUT else "Iteration Time (ms)"
    
    x = np.arange(len(configs))

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(111)
    for idx in range(yaxis_data[:, _filter].shape[1]):
        bars = ax.bar(
            x + idx*barwidth, yaxis_data[:, _filter][:, idx], width=barwidth, label=x_name[idx])
        for bar in bars:
            bar.set_hatch(marks[idx])
    plt.ylabel(yaxis_name, fontsize=fontsize)
    plt.ylim(0, 1.4*np.max(yaxis_data[:, _filter]))
    plt.legend(fontsize=fontsize)
    if _key == "BERT Base":
        plt.xticks(x[:-1] + (_iter_time.shape[1]/2)*barwidth,
                   configs[:-1], fontsize=fontsize, rotation=0)
        
    else:
        plt.xticks(x + (_iter_time.shape[1]/2)*barwidth,
                   configs, fontsize=fontsize, rotation=0)
    plt.yticks(fontsize=fontsize)
    plt.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.95,
                        wspace=0.2, hspace=0.4)
    _key_replace = "_".join(_key.split(" "))
    plt.savefig("fig/opfs/opfs_{}.pdf".format(_key_replace), bbox_inches='tight')

    # fig = plt.figure(figsize=(12, 5))
    # ax = plt.subplot(111)
    # for idx in range(speedup.shape[1]):
    #     bars = ax.bar(
    #         x + idx*barwidth, speedup[:, idx], width=barwidth, label=x_name[idx])
    #     for bar in bars:
    #         bar.set_hatch(marks[idx])
    # plt.ylabel("Speedup to No Fusion (%)", fontsize=fontsize)
    # plt.xticks(x + (speedup.shape[1]/2) * barwidth, configs, fontsize=fontsize, rotation=0)
    # plt.yticks(fontsize=fontsize)
    # plt.legend()
    # plt.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.95,
    #                     wspace=0.2, hspace=0.4)
    # _key_replace = "_".join(_key.split(" "))
    # plt.savefig("fig/opfs/opfs_speedup_{}.pdf".format(_key_replace), bbox_inches='tight')

print("max_speedup:{}".format(max_speedup))

