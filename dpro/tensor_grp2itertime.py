import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# if not os.path.exists("fig/replay"):
#     os.mkdir("fig/replay")
# os.system("rm -rf fig/replay/*")

sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")
font_size = 18
itertime = np.array([
    [125.9144863, 112.4667567, 110.6916983, 110.317214, 111.1744007,
        112.1839047, 112.697633, 113.373979, 120.9951003, 134.264199],
    [168.236701, 120.229268, 112.052838, 114.379899, 118.6657667,
        120.5094577, 125.10465, 131.4329227, 154.5064607, 182.6664527]
])
tensor_grp = [1, 5, 10, 20, 35, 36, 40, 50, 100, 214]
config = ["RDMA", "TCP"]

fig = plt.figure(figsize=(6, 3))
ax = plt.subplot(111)
ax.plot(tensor_grp, itertime[0], '-o', label=config[0])
ax.plot(tensor_grp, itertime[1], '-x', label=config[1])
plt.ylabel("Iteration Time (ms)", fontsize=font_size)
plt.xlabel("Tensor Group Number", fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.ylim(0.85 * np.min(itertime), 1.25 * np.max(itertime))
plt.xticks(fontsize=font_size)
plt.legend(fontsize=font_size, ncol=2)
plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.95,
                    wspace=0.2, hspace=0.4)
plt.savefig("fig/tensor_grp2itertime.pdf", bbox_inches='tight')
