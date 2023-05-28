import sys, os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style("white")

os.makedirs("fig/y_distriution", exist_ok=True)

Y = np.load("sample_200_y.npy", allow_pickle=True).astype(float)
print("Y", Y.shape)

font_size = 56

def plot_dist(_y, _name):
    fig = plt.figure(figsize=(15, 8))
    # plt.hist(_y, bins=10, edgecolor='k')

    sns.histplot(data=_y, bins=10, kde=True, 
        line_kws={"linewidth":4, "label": "curve"})
    # If True, compute a kernel density estimate to smooth the 
    # distribution and show on the plot as (one or more) line(s). Only relevant with univariate data.

    plt.xlabel("TIR kernel latency (s)", fontsize=font_size)
    if _name == "y":
        plt.xticks(fontsize=font_size-2, rotation=60)
    else:
        plt.xticks(fontsize=font_size-2)
    locs, labels = plt.yticks()
    new_locs = np.arange(0, max(locs), step=max(locs)/5.)
    new_ticks = (new_locs / 1e4).astype(int)
    plt.yticks(new_locs, new_ticks, fontsize=font_size-2)
    plt.ylabel("Count (1e4)", fontsize=font_size)
    plt.tight_layout()
    plt.savefig(f"fig/y_distriution/{_name}.pdf")
    plt.close()


from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
rng = np.random.RandomState(304)
bc = PowerTransformer(method="box-cox")
yj = PowerTransformer(method="yeo-johnson")
# n_quantiles is set to the training set size rather than the default value
# to avoid a warning being raised by this example
qt = QuantileTransformer(
    n_quantiles=500, output_distribution="normal", random_state=rng)
qt2 = QuantileTransformer(
    n_quantiles=500, output_distribution="uniform", random_state=rng)

plot_dist(Y, "y")
plot_dist(np.log(Y), "log_y")
Y = np.expand_dims(Y, axis=1)
plot_dist(bc.fit(Y).transform(Y).flatten(), "box_cox_y")
plot_dist(yj.fit(Y).transform(Y).flatten(), "yeo_johnson_y")
plot_dist(qt.fit(Y).transform(Y).flatten(), "quantile_y")
plot_dist(qt2.fit(Y).transform(Y).flatten(), "quantile_norm_y")
