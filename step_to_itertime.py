import os, sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_step_iter(s):
    s1 = s.split("Iter #")[1]
    step = int(s1.split(":")[0])
    iter_time = float(s1.split("iteration time ")[1].split(" ms")[0])
    return [step, iter_time]

files = sys.argv[1].split(",")
for file in files:
    with open(file, 'r') as fp:
        logs = fp.read()
    lines = re.findall("Iter #[0-9]+: [0-9]+.[0-9] img/sec per GPU, iteration time [0-9.]+ ms", logs)
    ### shape = (num_steps, (xy))
    step2iter_list = np.array([parse_step_iter(s) for s in lines]).T
    l1 = plt.plot(step2iter_list[0], step2iter_list[1], label=file.split("/")[-1])

plt.legend()
plt.xlabel("# of steps")
plt.ylabel("Iteration time (ms)")
plt.show()
