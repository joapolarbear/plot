
import ujson as json
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

files = sys.argv[1]
names = sys.argv[2].split(",")

grid_size_lists = []
for idx, file in enumerate(files.split(",")):
    grid_size_list = []
    with open(file, 'r') as fp:
        traces = json.load(fp)
    if "traceEvents" in traces:
        traces = traces["traceEvents"]
    for event in traces:
        if "Comm" in event["name"] or "Memcpy" in event["name"]:
            continue
        assert "Grid size" in event["args"], event
        grid_size = 1
        for _grid in re.findall("[0-9.]+", event["args"]["Grid size"]):
            grid_size *= float(_grid)
        assert grid_size > 0, event
        grid_size_list.append(grid_size)
    grid_size_lists.append(grid_size_list)
    grid_size_list = np.array(grid_size_list)
    np.random.shuffle(grid_size_list)
    sns.distplot(grid_size_list[:2000], hist=False, label=names[idx])

# shape = (# of trials, # of ops)
# grid_size_lists = np.array(grid_size_lists)

plt.xlim(0, 20000)
plt.xlabel("Grid Size", fontsize=18)
plt.legend()
plt.show()
