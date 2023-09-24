import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from help.utils import *

if not os.path.exists("fig/tsfs"):
    os.mkdir("fig/tsfs")
os.system("rm -rf fig/tsfs/*")
# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")

barwidth = 0.2
font_size = 24
max_speedup = 0

USE_THROUGHPUT = True
BATCH_SIZE = 32

strategy = [
    "Fuse all Tensors",
    "dPRO_TSFS",
    "No Tensor Fusion",
    "Default Horovod"
]

# ''' original
iter_time = np.array([
    [125.9144863, 111.1744007, 134.264199, 116.9958433],
    [168.236701, 114.379899, 182.6664527, 128.5568717],

    [103.6967753, 97.980992, 119.481985, 99.82658233],
    [146.550695, 101.240524, 166.5328183, 111.4922287],

    [388.0105733, 359.2547573, 376.116896, 375.4114943],
    [590.6265023, 437.6139797, 531.2120917, 460.7490457],
    [234.4891707, 238.3698783, 258.4212383, 214.021341],
    [522.6683217, 410.6463987, 412.5351587, 400.3735223],
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
yaxis_name = "Throughput\n(samples/s)" if USE_THROUGHPUT else "Iteration Time (ms)"

for idx in range(iter_time.shape[1]):
    bars = ax.bar(
        x + idx*barwidth, yaxis_data[:, idx], width=barwidth, label=strategy[idx])
    for bar in bars:
        bar.set_hatch(marks[idx])
plt.ylabel(yaxis_name, fontsize=font_size)
plt.xticks(x + (iter_time.shape[1]/2)*barwidth, dataset, fontsize=font_size*0.73, rotation=0)
plt.yticks(fontsize=font_size)
plt.legend(bbox_to_anchor=(0., 1.08, 1., .102), ncol=4, fontsize=font_size*0.75)

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

plt.subplots_adjust(left=0.13, bottom=0.2, right=0.95, top=0.95,
                    wspace=0.2, hspace=0.3)
plt.savefig("fig/tsfs/tsfs_all.pdf", bbox_inches='tight')


