import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
if not os.path.exists("fig/tsfs_opfs"):
    os.mkdir("fig/tsfs_opfs")
# os.system("rm -rf fig/tsfs_opfs/*")
# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")

plt.rcParams["font.sans-serif"] = "Simhei"
#填充符号
# marks = ["o","X","+","*","O","."]
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.15
font_size = 24
max_speedup = 0

USE_THROUGHPUT = True
BATCH_SIZE = 32
Normalize = True

strategy = np.array([
    "No Fusion",
    "TF XLA",
    "dPRO_OPFS",
    "dPRO_TSFS",
    "dPRO_OPFS_TSFS",
])

_filter = np.array([0, 1, 2, 4])

# ''' original
iter_time = np.array([
    [134.5339617, 149.4408847, 126.422159, 111.1744007, 104.223156],
    [187.94175, 230.108086, 183.1359463, 114.379899, 110.323135],
    [119.9843963, 149.7025093, 118.847235, 97.980992, 100.031773],
    [169.061017, 206.3683907, 167.4394923, 101.240524, 105.283936],

    [371.276363, 337.6628557, 328.1833887, 359.2547573, 315.7150427],
    [538.362813, 566.147264, 527.4501007, 437.6139797, 460.6041033],
    [261.206738, 222.943266, 231.733346, 238.3698783, 227.633818],
    [412.6467783, 491.9420797, 423.568797, 410.6463987, 402.038121],
    
])
dataset = np.array([
    "ResNet50\n+RDMA",
    "ResNet50\n+TCP",
    "InceptionV3\n+RDMA",
    "InceptionV3\n+TCP",

    "BERT Base\n+RDMA",
    "BERT Base\n+TCP",
    "VGG16\n+RDMA",
    "VGG16\n+TCP",
])
base = iter_time[:, 0].reshape(len(dataset), 1)
speedup = 100 * (base - iter_time) / base
x = np.arange(len(dataset))

for i in range(len(iter_time)):
    max_speedup = max(max_speedup, 100 * (max(iter_time[i]) - iter_time[i][1]) / max(iter_time[i]))

fig = plt.figure(figsize=(15, 4))
ax = plt.subplot(111)
yaxis_data = 1000 * BATCH_SIZE / iter_time if USE_THROUGHPUT else iter_time
yaxis_name = "Throughput\n(samples/sec)" if USE_THROUGHPUT else "Iteration Time (ms)"

if Normalize:
    base = yaxis_data[:, 0].reshape(len(dataset), 1)
    yaxis_data = yaxis_data / base

for idx in range(yaxis_data[:, _filter].shape[1]):
    bars = ax.bar(
        x + idx*barwidth, yaxis_data[:, _filter][:, idx], width=barwidth, label=strategy[_filter][idx])
    for bar in bars:
        bar.set_hatch(marks[idx])
plt.ylabel(yaxis_name, fontsize=font_size)
plt.xticks(x + (iter_time.shape[1]/2)*barwidth, dataset, fontsize=font_size*0.75, rotation=0)
plt.yticks(fontsize=font_size)
plt.legend(bbox_to_anchor=(0., 1.08, 1., .102), ncol=4, fontsize=font_size * 0.75)

# ax = plt.subplot(211)
# for idx in range(speedup.shape[1]):
#     bars = ax.bar(
#         x + idx*barwidth, speedup[:, idx], width=barwidth, label=strategy[idx])
#     for bar in bars:
#         bar.set_hatch(marks[idx])
# plt.ylabel("Speedup to 1 group (%)", fontsize=font_size)
# plt.xticks(x + (speedup.shape[1]/2) * barwidth, dataset, fontsize=font_size, rotation=0)
# plt.yticks(fontsize=font_size)
# plt.legend(fontsize=font_size)

plt.subplots_adjust(left=0.13, bottom=0.1, right=0.95, top=0.95,
                    wspace=0.2, hspace=0.3)
plt.savefig("fig/tsfs_opfs/tsfs_opfs_all.pdf", bbox_inches='tight')


