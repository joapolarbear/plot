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

xaxis_name = np.array(["HVD TCP", "BPS RDMA"])
xaxis_name_level2 = np.array([
    "ResNet50",
    "VGG16",
    "InceptionV3",
    "BERT_Base",
])

legends = np.array(["Ground Truth", "dPRO", "w/o Time Alignment"])
_filter = np.array([1, 2])
iter_time = np.array([
    [212.7014063, 203.134016, 234.9455571],
    [283.7625, 285.972318, 290.082053],
    [843.313688, 856.395883, 951.733479],
    [1181.111772, 1208.783676, 1238.807215],
    [205.715219, 213.173699, 218.473699],
    [282.004563, 284.2043743, 290.8038892],
    [841.820594, 873.95378, 950.748692],    
    [1349.104047, 1361.724845, 1377.25486],
])


iter_time_8w1g1m = np.array([
    [229.134391, 238.356007, 263.202757],
    [1196.517728, 1138.173024, 1354.325316],
    [322.1244375, 339.56886, 716.41511],
    [920.993125, 964.503067, 1109.757307],
])

def plot(_iter_time, name="clock_sync", _xaxis_name=None, _xaxis_size=None, bar_yinterval=None):
    x = np.arange(_iter_time.shape[0])
    yaxis_data = 1000 * BATCH_SIZE / _iter_time[:, _filter] if USE_THROUGHPUT else _iter_time[:, _filter]
    yaxis_name = "Throughput\n(samples/s)" if USE_THROUGHPUT else "Iteration Time (ms)"

    if True:
        base = _iter_time[:, 0].reshape(_iter_time.shape[0], 1)
        mse = 100 * np.abs(_iter_time[:, _filter] - base) / base
        yaxis_data = mse
        yaxis_name = "Prediction Error (%)"
        print(np.max(yaxis_data))

    # print(yaxis_data.shape)
    a = pd.DataFrame(yaxis_data,
                    index=pd.MultiIndex.from_product(
                        [xaxis_name_level2, _xaxis_name if _xaxis_name is not None else xaxis_name]),
                    columns=legends[_filter])

    ax = a.plot.bar(figsize=(12, 4), legend=False, width=barwidth)
    set_hierarchical_xlabels(a.index, font_size, bar_yinterval=0.12 if bar_yinterval is None else bar_yinterval)
    ax.grid(axis='x')

    # for idx in range(1, yaxis_data[:, _filter].shape[1]):
    #     bars = ax.bar(
    #         x + idx*barwidth, yaxis_data[:, _filter][:, idx], width=barwidth, label=legends[_filter][idx])
    #     # for bar in bars:
    #         # bar.set_hatch(marks[idx])
    # plt.xticks(x + (_iter_time.shape[1]/2-0.5)*barwidth, xaxis_name, fontsize=font_size*0.65, rotation=25)
    plt.xticks(fontsize=font_size*0.65 if _xaxis_size is None else _xaxis_size)
    plt.ylabel(yaxis_name, fontsize=font_size)
    plt.yticks(np.arange(0, 21, 5), fontsize=font_size-2)
    plt.ylim((0, 25))
    plt.legend(fontsize=font_size*0.75, frameon=False)
    plt.subplots_adjust(left=0.13, bottom=0.2, right=0.95, top=0.95,
                        wspace=0.2, hspace=0.3)
    plt.savefig("fig/clock_sync/{}.pdf".format(name), bbox_inches='tight')

plot(iter_time, name="clock_sync")

new_axis_name = np.array(["BPS\nRDMA", "HVD\nTCP", "HVD\nTCP*"])
# import code
# code.interact(local=locals())
new_iter_time = None
for row_level2 in range(xaxis_name_level2.shape[0]):
    rows = np.array([1, 0]) + row_level2 * xaxis_name.shape[0]
    append_array = iter_time[rows, :].reshape((xaxis_name.shape[0], -1))
    if new_iter_time is None:
        new_iter_time = append_array
    else:
        new_iter_time = np.concatenate(
            (new_iter_time, append_array), axis=0)
    append_array = iter_time_8w1g1m[row_level2, :].reshape((1, -1))
    if new_iter_time is None:
        new_iter_time = append_array
    else:
        new_iter_time = np.concatenate(
            (new_iter_time, append_array), axis=0)
print(new_iter_time.shape)
plot(new_iter_time, name="clock_sync2", 
    _xaxis_name=new_axis_name, _xaxis_size=font_size*0.65,
    bar_yinterval=0.2)
# plot(iter_time_large_scale, name="clock_sync_large_scale")

############################################################
xaxis_name = "# of workers x # of GPUs per worker"
barwidth = 0.2
legends = np.array(["Ground Truth", "dPRO", "w/o Time Alignment"])
_filter = np.array([1, 2])
xaxis_name = np.array([
    "2x8",
    "4x8",
    "8x8",
    "16x8"
])

### Horovod RDMA VGG16
iter_time_vgg16 = np.array([
    [196.51543, 199.4792044, 214.8589746],
    [209.921537, 215.2849317, 233.8804146],
    [215.848908, 219.195641, 260.991813],
    [241.279914, 244.328643, 327.096731],
])

iter_time_bert_base = np.array([
    [510.589281, 520.944097, 537.317855],
    [564.0925925, 581.189598, 637.732206],
    [676.7370552, 691.801164, 865.715783],
    [902.3060954, 917.945275, 1233.377383],
])



def plot(iter_time, name):
    x = np.arange(iter_time.shape[0])

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)
    ax.grid(axis='x')
    yaxis_data = 1000 * BATCH_SIZE / iter_time[:,_filter] if USE_THROUGHPUT else iter_time[:,_filter]
    yaxis_name = "Throughput per GPU\n(samples/sec)" if USE_THROUGHPUT else "Iteration Time (ms)"

    if True:
        base = iter_time[:, 0].reshape(iter_time.shape[0], 1)
        mse = 100 * np.abs(iter_time[:, _filter] - base) / base
        yaxis_data = mse
        yaxis_name = "Prediction Error (%)"
        print(np.max(yaxis_data))

    for idx in range(yaxis_data.shape[1]):
        print(yaxis_data[:, idx])
        bars = ax.bar(
            x + idx*barwidth, yaxis_data[:, idx], width=barwidth, label=legends[_filter][idx])
        # for bar in bars:
        #     bar.set_hatch(marks[idx])
    plt.xticks(x + (iter_time[:,_filter].shape[1]/2)*barwidth,
               xaxis_name, fontsize=font_size*0.75, rotation=0)
    plt.ylabel(yaxis_name, fontsize=font_size)
    plt.xlabel(list(xaxis_name), fontsize=0.85*font_size)
    plt.yticks(np.arange(0, 1.4*np.max(yaxis_data), 1.4* np.max(yaxis_data)/4//10*10), fontsize=font_size)
    plt.legend(fontsize=font_size*0.85, frameon=False, ncol=1)
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.97, top=0.95,
                        wspace=0.2, hspace=0.3)
    plt.savefig("fig/clock_sync/{}.pdf".format(name), bbox_inches='tight')

plot(iter_time_vgg16, "clock_sync_vgg16")
plot(iter_time_bert_base, "clock_sync_bert_base")
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