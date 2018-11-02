# Module plot_bars.py
#
# 
#
# AH 01/11/2018

import matplotlib.pyplot as plt
import numpy as np

def plot_bars(x, y, x_label, y_label):#, x_text, y_text):
    """ plot x, y data in form of bars 
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    bar_width = 0.85
    rects1 = ax.bar(x, y, bar_width, alpha=0.3, color='k')
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xticks(x)
    ax.set_yticks(np.linspace(0,100,11))
    #ax.text(x_text, y_text, '%.0f EOFs = %2.1f prct of variance' % (max(x),100*sum(y)))
    ax.set_title('Signal representation by mode')
