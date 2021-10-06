import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from help.utils import set_hierarchical_xlabels
if not os.path.exists("fig/large_scale"):
    os.mkdir("fig/large_scale")
# os.system("rm -rf fig/large_scale/*")
# Set the palette using the name of a palette:
rc = {'axes.spines.left': True,
     'axes.spines.right': True,
     'axes.spines.top': False,
    }
sns.set_theme(style="whitegrid", color_codes=True, rc=rc)
tips = sns.load_dataset("tips")
plt.rcParams["font.sans-serif"] = "Simhei"

USE_THROUGHPUT = True
BATCH_SIZE = 32
# marks = ["o","X","+","*","O","."]
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.2
font_size = 24
max_speedup = 0

x_name = ["Ground Truth", "dPRO", "Daydream"]
configs = [
    "TCP",
    "RDMA",
]
dataset_level2 = np.array([
    "ResNet50",
    "VGG16",
    "InceptionV3",
    "BERT Base",
])

### Replay error
_iter_time = np.array([
    [249.997734, 242.605954, 113.766854],
    [218.768534, 202.659429, 112.838],
    [562.137923, 534.052102, 137.659056],
    [259.038545, 250.489765, 141.2075],
    [298.886305, 273.505068, 98.772111],
    [236.704021, 246.297, 97.447889],
    [841.276743, 794.041738, 347.878962],
    [541.290223, 514.828754, 359.668395],
])

base = _iter_time[:, 0].reshape(_iter_time.shape[0], 1)
mse = 100 * np.abs(_iter_time - base) / base
x = np.arange(_iter_time.shape[0])
max_speedup = max(max_speedup, max((mse[:, 2] - mse[:, 1]) / mse[:, 1]))

fig = plt.figure(figsize=(15, 5))
ax = plt.subplot(111)

a = pd.DataFrame(_iter_time,
                 index=pd.MultiIndex.from_product([dataset_level2, configs]),
                 columns=x_name)
ax = a.plot.bar(figsize=(15, 4), legend=False)
set_hierarchical_xlabels(a.index, font_size)
# for idx in range(len(x_name)):
#     bars = ax.bar(
#         x + idx*barwidth, _iter_time[:, idx],
#         width=barwidth, label=x_name[idx])
#     # for bar in bars:
#     #     bar.set_hatch(marks[idx])
# plt.xticks(x + (len(x_name)/2)*barwidth, configs,
#             fontsize=font_size*0.75, rotation=0)
ax.grid(False)
plt.xticks(fontsize=font_size*0.75, rotation=0)
plt.ylabel("Iteration Time (ms)", fontsize=font_size)
plt.ylim(0, 1.4*np.max(_iter_time))
plt.yticks(np.arange(0, 1201, 300), fontsize=font_size)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
color = 'red'
# we already handled the x-label with ax1
ax2.set_ylabel('dPRO Error (%)', fontsize=font_size)
ax2.plot(x + barwidth, mse[:, 1], '-o',
            color=color, linewidth=3, markersize=10, label="Prediction Error")
ax2.tick_params(axis='y')
for label in ax2.yaxis.get_majorticklabels():
    label.set_fontsize(font_size)
ax2.set_yticks(np.arange(0, 25, 5))

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, 
    ncol=4, fontsize=font_size*0.85, frameon=False)
# plt.legend(ncol=4, fontsize=font_size, frameon=False)
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.93, top=0.95,
                    wspace=0.2, hspace=0.4)
plt.savefig("fig/large_scale/replay_{}.pdf".format("128g"), bbox_inches='tight')

print("Replayer max_speedup:{}".format(max_speedup))

### Tensor Fusion
'''
strategy = [
    "Fuse all Tensors",
    "dPRO_TSFS",
    "No Tensor Fusion",
]
iter_time = np.array([
    [127.5404137, 114.9568243, 645.2837707],
    [188.4394883, 165.9778517, 965.1486237],
    [111.28904, 100.320816, 536.9357267],
    [169.5764857, 131.5945867, 883.0202417],
    [314.9751027, 280.3203263, 358.8552],
    [584.4459613, 489.7807123, 641.9551847],
    [433.046508, 361.900409, 882.7339093],
    [656.5911693, 555.7611863, 1567.038186]
])
dataset = np.array([
    "ResNet50\n+RDMA",
    "ResNet50\n+TCP",
    "InceptionV3\n+RDMA",
    "InceptionV3\n+TCP",
    "VGG16\n+RDMA",
    "VGG16\n+TCP",
    "BERT Base\n+RDMA",
    "BERT Base\n+TCP",
])
base = iter_time[:, 0].reshape(iter_time.shape[0], 1)
speedup = 100 * (base - iter_time) / base
x = np.arange(iter_time.shape[0])

fig = plt.figure(figsize=(15, 5))
ax = plt.subplot(111)
ax.grid(axis='x')
yaxis_data = 1000 * BATCH_SIZE / iter_time if USE_THROUGHPUT else iter_time
yaxis_name = "Throughput\n(samples/s)" if USE_THROUGHPUT else "Iteration Time (ms)"

for idx in range(iter_time.shape[1]):
    bars = ax.bar(
        x + idx*barwidth, yaxis_data[:, idx], width=barwidth, label=strategy[idx])
    # for bar in bars:
    #     bar.set_hatch(marks[idx])
plt.ylabel(yaxis_name, fontsize=font_size)
plt.xticks(x + (iter_time.shape[1]/2)*barwidth,
           dataset, fontsize=font_size*0.75, rotation=0)
plt.yticks(np.arange(0, 321, 80), fontsize=font_size)
plt.legend(fontsize=font_size, frameon=False)
for i in range(len(iter_time)):
    max_speedup = max(
        max_speedup, 100 * (max(iter_time[i]) - iter_time[i][1]) / max(iter_time[i]))

plt.subplots_adjust(left=0.12, bottom=0.2, right=0.99, top=0.95,
                    wspace=0.2, hspace=0.4)
plt.savefig("fig/large_scale/tsfs_128g.pdf", bbox_inches='tight')
print("TSFS: max_speedup: {}".format(max_speedup))
'''
dataset = np.array([
    "RDMA",
    "TCP",
])
dataset_level2 = np.array([
    "ResNet50",
    "InceptionV3",
    "VGG16",
    "BERT Base"
])
strategy = np.array([
    "Default",
    "TF XLA",
    "dPRO_OPFS",
    "dPRO_TSFS",
    "dPRO_OPFS_TSFS",
])

# iter_time = np.array([
#     [628.5137093, 691.314427, 643.516008, 114.9568243, 123.2982717],
#     [967.2492107, 1158.26052, 1014.416098, 171.0253317, 149.3234237],
#     [507.5661023, 558.45387, 503.941862, 100.320816, 108.5660617],
#     [872.7584123, 1000.551224, 864.531843, 131.5945867, 145.3171253],
#     [444.2450127, 422.887222, 425.1975617, 280.3203263, 348.5196193],
#     [615.2580977, 668.4013923, 588.671001, 489.7807123, 475.139348],
#     [874.7569403, 914.0408277, 844.8319357, 361.900409, 380.821546],
#     [1488.883273, 1536.316546, 1495.518239, 555.7611863, 520.8536623],
# ])
_filter = np.array([1, 2, 4])
iter_time = np.array([
[628.5137093, 691.314427, 619.0860037, 125.1759103, 123.2982717],
[967.2492107, 1158.26052, 952.7404725, 171.0253317, 149.3234237],
[507.5661023, 558.45387, 503.941862, 118.0945928, 108.5660617],
[872.7584123, 1000.551224, 864.531843, 158.4359028, 145.3171253],
[444.2450127, 422.887222, 425.1975617, 348.5196193, 280.3203263],
[615.2580977, 668.4013923, 588.671001, 489.7807123, 475.139348],
[874.7569403, 914.0408277, 844.8319357, 380.821546, 361.900409],
[1488.883273, 1536.316546, 1495.518239, 555.7611863, 520.8536623],
])
_filter = np.array([0, 1, 4])
# dataset = dataset
base = iter_time[:, 0].reshape(iter_time.shape[0], 1)
speedup = 100 * (base - iter_time) / base
x = np.arange(iter_time.shape[0])

for i in range(len(iter_time)):
    max_speedup = max(
        max_speedup, 100 * (max(iter_time[i]) - iter_time[i][1]) / max(iter_time[i]))
fig = plt.figure(figsize=(15, 5))
ax = plt.subplot(111)
ax.grid(axis='x')
yaxis_data = 1000 * BATCH_SIZE / iter_time[:,_filter] if USE_THROUGHPUT else iter_time[:,_filter]
yaxis_name = "Throughput\n(samples/sec)" if USE_THROUGHPUT else "Iteration Time (ms)"

a = pd.DataFrame(yaxis_data,
                 index=pd.MultiIndex.from_product([dataset_level2, dataset]),
                 columns=strategy[_filter])
ax = a.plot.bar(figsize=(15, 4), legend=False)
set_hierarchical_xlabels(a.index, font_size)
ax.grid(axis='x')
# for idx in range(yaxis_data.shape[1]):
#     bars = ax.bar(
#         x + idx*barwidth, yaxis_data[:, idx], width=barwidth, label=strategy[_filter][idx])
#     # for bar in bars:
#     #     bar.set_hatch(marks[idx])
# plt.xticks(x + (iter_time.shape[1]/2)*barwidth,
#            dataset, fontsize=font_size*0.75, rotation=0)
plt.xticks(fontsize=font_size*0.75, rotation=0)
plt.ylabel(yaxis_name, fontsize=font_size)
plt.yticks(np.arange(0, 301, 75), fontsize=font_size)
plt.legend(fontsize=font_size, frameon=False)
plt.subplots_adjust(left=0.12, bottom=0.15, right=0.97, top=0.95,
                    wspace=0.2, hspace=0.3)
plt.savefig("fig/large_scale/tsfs_opfs_128g.pdf", bbox_inches='tight')
