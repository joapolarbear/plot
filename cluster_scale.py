import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from help.utils import set_hierarchical_xlabels, cal_speedup
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

xaxis_name = "# of GPUs"
strategy = np.array(["No Fusion", "XLA", "dPRO"])
dataset = np.array([
    "16",
    "32",
    "64",
    "128"
])

### Replay error
_iter_time_vgg16 = np.array([
    [402.0644841, 478.3444817, 392.2433961],
    [454.3473721, 535.1383627, 419.5178496],
    [496.0514575, 504.0772271, 435.1323087],
    [520.2835991, 553.6248561, 473.7746381],
])

_iter_time_bert_base = np.array([
    [538.362813, 574.6751623, 460.6041033], 
    [688.1985281, 648.346473, 466.1738945], 
    [1074.492512, 1030.988961, 516.5604431], 
    [1874.1539, 1830.073563, 538.0927985], 
])

_filter = np.array([0, 1, 2])

def plot(iter_time, name):
    max_speedup = 0
    x = np.arange(iter_time.shape[0])

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)
    ax.grid(axis='x')
    yaxis_data = 1000 * BATCH_SIZE / iter_time[:,_filter] if USE_THROUGHPUT else iter_time[:,_filter]
    yaxis_name = "Throughput per GPU\n(samples/sec)" if USE_THROUGHPUT else "Iteration Time (ms)"

    print(yaxis_data)
    print(np.max(cal_speedup(yaxis_data, 0, small_better=not USE_THROUGHPUT)))

    for idx in range(yaxis_data.shape[1]):
        # print(yaxis_data[:, idx])
        bars = ax.bar(
            x + idx*barwidth, yaxis_data[:, idx], width=barwidth, label=strategy[_filter][idx])
        # for bar in bars:
        #     bar.set_hatch(marks[idx])
    plt.xticks(x + (iter_time.shape[1]/2)*barwidth,
               dataset, fontsize=font_size*0.75, rotation=0)
    plt.ylabel(yaxis_name, fontsize=font_size)
    plt.xlabel(xaxis_name, fontsize=font_size)
    plt.yticks(np.arange(0, 1.4*np.max(yaxis_data), 1.4* np.max(yaxis_data)/4//10*10), fontsize=font_size)
    plt.legend(
        bbox_to_anchor=(0., 1.08, 1., .18), 
        fontsize=font_size*0.85, frameon=False, ncol=3)
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.97, top=0.95,
                        wspace=0.2, hspace=0.3)
    plt.savefig("fig/large_scale/{}.pdf".format(name), bbox_inches='tight')

plot(_iter_time_vgg16, "scale_vgg16")
plot(_iter_time_bert_base, "scale_bert_base")