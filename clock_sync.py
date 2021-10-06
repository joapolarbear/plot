import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from help.utils import set_hierarchical_xlabels

sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")
plt.rcParams["font.sans-serif"] = "Simhei"

USE_THROUGHPUT = False
BATCH_SIZE = 32
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.6
font_size = 24

save_dir = "./fig/clock_sync"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dataset = np.array([
    "ResNet50",
    "InceptionV3",
    "VGG16",
    "BERT_Base",
])
dataset_level2 = ["HVD TCP", "BPS RDMA"]


strategy = np.array(["Ground Truth", "dPRO", "dPRO w/o sync"])
_filter = np.array([1, 2])
iter_time = np.array([
    [212.7014063, 215.322057, 249.0422906],
    [843.313688, 858.2334128, 897.0615402],
    [213.173699, 206.3683907, 218.473699],
    [841.820594, 853.5631494, 948.0208912],
    [283.7625, 285.972318, 290.082053],
    [1181.111772, 1208.783676, 1238.807215],
    [282.004563, 284.2043743, 290.8038892],
    [1349.104047, 1349.231957, 1364.619495],
])

x = np.arange(iter_time.shape[0])
yaxis_data = 1000 * BATCH_SIZE / iter_time[:, _filter] if USE_THROUGHPUT else iter_time[:, _filter]
yaxis_name = "Throughput\n(samples/s)" if USE_THROUGHPUT else "Iteration Time (ms)"

if True:
    base = iter_time[:, 0].reshape(iter_time.shape[0], 1)
    mse = 100 * np.abs(iter_time[:, _filter] - base) / base
    yaxis_data = mse
    yaxis_name = "Prediction Error (%)"
    print(np.max(yaxis_data))

print(yaxis_data.shape)
a = pd.DataFrame(yaxis_data,
                 index=pd.MultiIndex.from_product([dataset_level2, dataset]),
                 columns=strategy[_filter])

ax = a.plot.bar(figsize=(12, 4), legend=False, width=barwidth)
set_hierarchical_xlabels(a.index, font_size, bar_yinterval=0.12)
ax.grid(axis='x')

# for idx in range(1, yaxis_data[:, _filter].shape[1]):
#     bars = ax.bar(
#         x + idx*barwidth, yaxis_data[:, _filter][:, idx], width=barwidth, label=strategy[_filter][idx])
#     # for bar in bars:
#         # bar.set_hatch(marks[idx])
# plt.xticks(x + (iter_time.shape[1]/2-0.5)*barwidth, dataset, fontsize=font_size*0.65, rotation=25)
plt.xticks(fontsize=font_size*0.65)
plt.ylabel(yaxis_name, fontsize=font_size)
plt.yticks(np.arange(0, 21, 5), fontsize=font_size-2)
plt.legend(fontsize=font_size*0.75, frameon=False)
plt.subplots_adjust(left=0.13, bottom=0.2, right=0.95, top=0.95,
                    wspace=0.2, hspace=0.3)
plt.savefig("fig/clock_sync/clock_sync.pdf", bbox_inches='tight')

# bias_range_path = "../../bias_range.json" 
# with open(bias_range_path, 'r') as fp:
#     bias_range = json.load(fp)


# a_id = 0
# b_id = 1
# bias_dict = bias_range[str(b_id)][str(a_id)]

# fig = plt.figure(figsize=(12, 4))
# ax = plt.subplot(111)
# try:
#     upper_ts, uppers = zip(*bias_dict["upper"])
#     ax.scatter(upper_ts, uppers, label="upper")
# except:
#     pass
# try:
#     lower_ts, lowers = zip(*bias_dict["lower"])
#     print(len(lowers))
#     ax.scatter(lower_ts, lowers, label="lower")
# except:
#     pass
# plt.xlabel("Clock Time on host id {}".format(a_id))
# plt.ylabel("t_b - t_a")
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "upper_lower_bound.png"))