import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
if not os.path.exists("fig/opfs2"):
    os.mkdir("fig/opfs2")
# os.system("rm -rf fig/opfs2/*")
# Set the palette using the name of a palette:
sns.set_theme(style="whitegrid", color_codes=True)
tips = sns.load_dataset("tips")
plt.rcParams["font.sans-serif"] = "Simhei"

# iter_time = {
#     "ResNet50": np.array([
#         [187.94175, 230.108086, 183.1359463],
#         [134.5339617, 149.4408847, 126.422159],
#         [196.0744385, 267.215776, 179.6238305],
#         [106.7313433, 131.8673533, 100.0797827],
#     ]),
#     "VGG16": np.array([
#         [412.6467783, 491.9420797, 423.568797],
#         [261.206738, 222.943266, 231.733346],
#         [523.1078147, 588.9855147, 510.751112],
#         [221.690941, 291.430354, 203.2869497],
#     ]),
#     "InceptionV3": np.array([
#         [169.061017, 206.3683907, 167.4394923],
#         [119.9843963, 149.7025093, 118.847235],
#         [111.2059433, 172.1465507, 109.173385],
#         [92.544087, 113.9632467, 91.95433433],
#     ]),
#     "BERT Base": np.array([
#         [538.362813, 566.147264, 527.4501007],
#         [371.276363, 337.6628557, 328.1833887],
#         [649.079728, 741.331482, 591.633844],
#         [1000000, 1000000, 1000000],
#     ])
# }


USE_THROUGHPUT = True
BATCH_SIZE = 32
marks = ["/", "-", "\\", "x", "+", "."]
barwidth = 0.2
font_size = 24

strategy = np.array(["No Fusion", "TF XLA", "dPRO_OPFS"])
_filter = np.array([0, 1, 2])

dataset = np.array([
    "hvd\nResNet50",
    "bps\nResNet50",
    "hvd\nInceptionV3",
    "bps\nInceptionV3",
    "hvd\nBERT Base",
    "bps\nBERT Base",
    "hvd\nVGG16",
    "bps\nVGG16",
])

iter_time = np.array([
    [130.311237, 161.202021, 114.1515095],
    [132.317, 183.488, 120.841],
    [159.964836, 214.4858717, 153.1182195],
    [155.497, 222.797, 151.030],
    [538.362813, 566.147264, 527.4501007],
    [608.526, 716.9374083, 554.1429844],
    [402.0644841, 511.7555598, 391.5570669],
    [464.463, 535.741, 446.094],
])

max_speedup = 0
base = iter_time[:, 0].reshape(len(dataset), 1)
speedup = 100 * (base - iter_time) / base
x = np.arange(len(dataset))
for i in range(len(iter_time)):
    max_speedup = max(
        max_speedup, 100 * (max(iter_time[i]) - iter_time[i][1]) / max(iter_time[i]))

fig = plt.figure(figsize=(12, 4))
ax = plt.subplot(111)
yaxis_data = 1000 * BATCH_SIZE / iter_time if USE_THROUGHPUT else iter_time
yaxis_name = "Throughput\n(samples/s)" if USE_THROUGHPUT else "Iteration Time (ms)"

for idx in range(yaxis_data[:, _filter].shape[1]):
    bars = ax.bar(
        x + idx*barwidth, yaxis_data[:, _filter][:, idx], width=barwidth, label=strategy[_filter][idx])
    for bar in bars:
        bar.set_hatch(marks[idx])
plt.ylabel(yaxis_name, fontsize=font_size)
plt.xticks(x + (iter_time.shape[1]/2-0.5)*barwidth, dataset, fontsize=font_size*0.65, rotation=15)
plt.yticks(fontsize=font_size)
plt.legend(fontsize=font_size*0.75)
plt.subplots_adjust(left=0.13, bottom=0.2, right=0.95, top=0.95,
                    wspace=0.2, hspace=0.3)
plt.savefig("fig/opfs2/opfs_a.pdf", bbox_inches='tight')

### RDMA
iter_time = np.array([
    [108.0072415, 117.8426155, 98.6042845],
    [109.8746753, 131.8673533, 100.0797827],
    [145.191236, 151.2675826, 142.5182833],
    [151.3811675, 190.1337183, 149.73572],
    [438.915068, 507.4753043, 430.0181796],
    [496.0678063, 453.9115632, 441.1685575],
    [208.9370307, 227.8026453, 183.6807798],
    [341.146100, 350.9571147, 297.2897195]
])

max_speedup = 0
base = iter_time[:, 0].reshape(len(dataset), 1)
speedup = 100 * (base - iter_time) / base
x = np.arange(len(dataset))
for i in range(len(iter_time)):
    max_speedup = max(max_speedup, 100 * (max(iter_time[i]) - iter_time[i][1]) / max(iter_time[i]))

fig = plt.figure(figsize=(12, 4))
ax = plt.subplot(111)
yaxis_data = 1000 * BATCH_SIZE / iter_time if USE_THROUGHPUT else iter_time
yaxis_name = "Throughput\n(samples/s)" if USE_THROUGHPUT else "Iteration Time (ms)"

for idx in range(yaxis_data[:, _filter].shape[1]):
    bars = ax.bar(
        x + idx*barwidth, yaxis_data[:, _filter][:, idx], width=barwidth, label=strategy[_filter][idx])
    for bar in bars:
        bar.set_hatch(marks[idx])
plt.ylabel(yaxis_name, fontsize=font_size)
plt.xticks(x + (iter_time.shape[1]/2-0.5)*barwidth, dataset, fontsize=font_size*0.75, rotation=15)
plt.yticks(fontsize=font_size)
plt.subplots_adjust(left=0.13, bottom=0.2, right=0.95, top=0.95,
                    wspace=0.2, hspace=0.3)
plt.savefig("fig/opfs2/opfs_b.pdf", bbox_inches='tight')

print("max_speedup: {}".format(max_speedup))
