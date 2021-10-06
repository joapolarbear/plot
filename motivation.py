import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")
plt.rcParams["font.sans-serif"] = "Simhei"

plt.rcParams["font.sans-serif"] = "Simhei"
# marks = ["o","X","+","*","O","."]
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.2
font_size = 24

x_name = ["Ground Truth", "Daydream"]
configs = [
    "Horovod+RDMA\n16 GPUs",
    "Horovod+TCP\n16 GPUs",
    "BytePS+RDMA\n16 GPUs",
    "Horovod+RDMA\n128 GPUs",
]
_iter_time = np.array([
    [138.644111, 109.186041],
    [149.900667, 110.568491],
    [123.904031, 118.420341],
    [218.768534, 112.838]
])
base = _iter_time[:, 0].reshape(len(configs), 1)
mse = 100 * np.abs(_iter_time - base) / base
x = np.arange(len(configs))

fig = plt.figure(figsize=(10, 4))
ax = plt.subplot(111)
for idx in range(len(x_name)):
    bars = ax.bar(
        x + idx*barwidth, _iter_time[:, idx],
        width=barwidth, label=x_name[idx])
    # for bar in bars:
    #     bar.set_hatch(marks[idx])
ax.grid(False)
plt.ylabel("Iteration Time (ms)", fontsize=font_size)
plt.ylim(0, 1.4*np.max(_iter_time))
plt.xticks(x + (len(x_name)/2)*barwidth, configs,
            fontsize=font_size*0.75, rotation=0)
# plt.xlabel(title)
plt.yticks(np.arange(0, 301, 60), fontsize=font_size-4)
plt.legend(loc=2, ncol=2, fontsize=font_size, frameon=False)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
color = 'firebrick'
ax2.set_ylabel('MAPE (%)', color=color, fontsize=font_size)  # we already handled the x-label with ax1
ax2.plot(x + barwidth, mse[:, 1], '-o',
            color=color, linewidth=3, markersize=10)
ax2.tick_params(axis='y', labelcolor=color)
for label in ax2.yaxis.get_majorticklabels():
    label.set_fontsize(font_size-4)
    # label.set_fontname('courier')
# plt.ylim(0, 20)
ax2.set_yticks(np.arange(0, 60, 10))

plt.subplots_adjust(left=0.13, bottom=0.2, right=0.90, top=0.9,
                    wspace=0.2, hspace=0.4)
plt.savefig("fig/motivation_daydream.pdf", bbox_inches='tight')


### Linear speedup
gpu_num = np.array([1, 8, 16, 32, 64, 128])
throughput = np.array([327.3159523, 2470.337113, 4332.907091, 8694.129102, 17206.04534, 29853.6365])
linear = np.array([327.3159523, 2618.527619, 5237.055237, 10474.11047, 20948.22095, 41896.4419])
fig = plt.figure(figsize=(10, 4))
ax = plt.subplot(111)
ax.plot(gpu_num, throughput / 10000, '.-', linewidth=3, markersize=10, label="Actual throughput")
ax.plot(gpu_num, linear / 10000, '--', linewidth=3, markersize=10)

plt.ylabel("Throughput\n(10000 samples/s)", fontsize=font_size)
plt.xlabel("# of GPUs", fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xticks(fontsize=font_size)
ax.set_xticks(gpu_num)
plt.legend(loc=2, ncol=2, fontsize=font_size)
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.99, top=0.9,
                    wspace=0.2, hspace=0.4)
plt.savefig("fig/linear_speedup.pdf", bbox_inches='tight')
# plt.show()
