import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from help.utils import set_hierarchical_xlabels

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
barwidth = 0.7
font_size = 24
max_speedup = 0

USE_THROUGHPUT = True
BATCH_SIZE = 32
Normalize = 1

strategy = np.array([
    "No Fusion",
    "XLA",
    "dPRO_OPFS",
    "dPRO_TSFS",
    "dPRO_OPFS_TSFS",
    "Horovod Autotune",
])
_filter = np.array([0, 1, 2, 3, 4, 5])
dataset = np.array([
    "HVD",
    "BPS",
    # "HVD",
    # "BPS",
    # "HVD",
    # "BPS",
    # "HVD",
    # "BPS",
])
dataset_level2 = np.array([
    "ResNet50",
    "InceptionV3",
    "BERT Base",
    "VGG16"
])
tick_width = 1 / len(dataset_level2)
tick_border = [tick_width * (id + 1) for id in range(len(dataset_level2))]
locator = np.array(tick_border) - tick_width/2


# ''' original
iter_time_rdma = np.array([
    [108.007241, 117.8426155, 98.6042845, 104.096462, 96.757789, 104.4862539],
    [109.874675, 131.8673533, 100.0797827, 107.2866916, 98.74859376, 0],
    [145.191236, 151.2675826, 142.5182833, 122.008442, 120.9608959, 144.0316344],
    [151.3811675, 190.1337183, 149.73572, 150.9447707, 148.2589382, 0],
    [496.0678063, 507.4753043, 441.1685575, 474.9022466, 421.1896469, 490.6773843],
    [438.915068, 453.9115632, 430.0181796, 398.2888203, 390.6056488, 0],
    [208.937031, 227.8026453, 183.6807798, 184.0641224, 187.035415, 201.7059142],
    [341.146100, 350.9571147, 297.2897195, 317.0380551, 281.8065267, 0],
])

iter_time_tcp = np.array([
    [130.311237, 161.202021, 114.1515095, 118.5679035, 103.064695, 124.7491198],
    [132.317435, 183.488, 120.841, 127.7619627, 116.8345863, 0],
    [159.964836, 214.4858717, 153.1182195, 134.314232, 131.6302005, 153.3524422],
    [155.496753, 222.797, 151.3811675, 149.1754926, 147.4056218, 0],
    [538.362813, 566.147264, 527.4501007, 511.4446724, 460.6041033, 543.7507104],
    [608.525883, 716.9374083, 554.1429844, 535.9590281, 503.6792646, 0],
    [402.0644841, 511.7555598, 391.5570669, 392.9137975, 392.243396, 396.054662],
    [464.463340, 535.741, 446.094, 457.839977, 443.870815, 0],
])

def trial(iter_time, pdf_name="tsfs_opfs_all_tcp", legend=True):
    global max_speedup
    base = iter_time[:, 0].reshape(iter_time.shape[0], 1)
    speedup = 100 * (base - iter_time) / base
    x = np.arange(iter_time.shape[0])

    for i in range(len(iter_time)):
        max_speedup = max(max_speedup, 100 * (max(iter_time[i]) - iter_time[i][1]) / max(iter_time[i]))

    yaxis_data = 1000 * BATCH_SIZE / iter_time if USE_THROUGHPUT else iter_time
    yaxis_name = "Throughput" if USE_THROUGHPUT else "Iteration Time (ms)"

    # fig = plt.figure(figsize=(15, 4))
    # ax = plt.subplot(111)
    # ax.grid(axis='x')
    if Normalize is not None:
        base = yaxis_data[:, Normalize].reshape(iter_time.shape[0], 1)
        yaxis_data = yaxis_data / base
        yaxis_name = "Normalized " + yaxis_name
    
    _base = yaxis_data[:, 1].reshape(iter_time.shape[0], 1)

    a = pd.DataFrame(yaxis_data[:, _filter],
                 index=pd.MultiIndex.from_product([dataset_level2, dataset]),
                 columns=strategy[_filter])

    
    ax = a.plot.bar(figsize=(15, 4), legend=False, width=barwidth)
    set_hierarchical_xlabels(a.index, font_size)
    ax.grid("x")

    # for idx in range(yaxis_data[:, _filter].shape[1]):
    #     bars = ax.bar(
    #         x + idx*barwidth, yaxis_data[:, _filter][:, idx], width=barwidth, label=strategy[_filter][idx])
    #     # for bar in bars:
    #     #     bar.set_hatch(marks[idx])
    # plt.xticks(x + (iter_time.shape[1]/2)*barwidth, dataset, fontsize=font_size*0.7, rotation=0)
    plt.xticks(fontsize=font_size*0.8)
    plt.ylabel(yaxis_name, fontsize=font_size)
    plt.yticks(np.arange(0, 1.6, 0.3), fontsize=font_size * 0.8)
    if legend:
        plt.legend(bbox_to_anchor=(0., 1.08, 1., .102), ncol=3, fontsize=font_size * 0.75, frameon=False)
        plt.subplots_adjust(left=0.13, bottom=0.1, right=0.95, top=0.95,
                        wspace=0.2, hspace=0.3)
    else:
        plt.subplots_adjust(left=0.13, bottom=0.1, right=0.95, top=0.99,
                        wspace=0.2, hspace=0.3)
    plt.savefig("fig/tsfs_opfs/{}.pdf".format(pdf_name), bbox_inches='tight')


trial(iter_time_tcp, "tsfs_opfs_all_tcp")
trial(iter_time_rdma, "tsfs_opfs_all_rdma", legend=False)
print("max_speedup: {}".format(max_speedup))