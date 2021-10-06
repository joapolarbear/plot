from itertools import groupby
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt

### Refer to https://linuxtut.com/en/92c21048bacadce811ec/
def set_hierarchical_xlabels(index, font_size, ax=None,
                             bar_xmargin=0.1, #Margins on the left and right ends of the line, X-axis scale
                             bar_yinterval=0.12, #Relative value with the vertical spacing of the line and the length of the Y axis as 1?
                            ):
    

    ax = ax or plt.gca()

    assert isinstance(index, pd.MultiIndex)
    labels = ax.set_xticklabels([s for *_, s in index])
    for lb in labels:
        lb.set_rotation(0)

    transform = ax.get_xaxis_transform()

    for i in range(1, len(index.codes)):
        xpos0 = -0.5 #Coordinates on the left side of the target group
        for (*_, code), codes_iter in groupby(zip(*index.codes[:-i])):
            xpos1 = xpos0 + sum(1 for _ in codes_iter) #Coordinates on the right side of the target group
            ax.text((xpos0+xpos1)/2, (bar_yinterval * (-i-0.1)),
                    index.levels[-i-1][code],
                    transform=transform,
                    ha="center", va="top", fontsize=font_size*0.7)
            ax.add_line(Line2D([xpos0+bar_xmargin, xpos1-bar_xmargin],
                               [bar_yinterval * -i]*2,
                               transform=transform,
                               color="k", clip_on=False))
            xpos0 = xpos1