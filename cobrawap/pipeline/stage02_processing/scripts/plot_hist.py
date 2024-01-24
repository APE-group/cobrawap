"""
Plot traces
-----------

description...

Input: neo.Block with ...

Output: neo.Block + ...


"""
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import quantities as pq
import random
from utils.io import load_neo, save_plot
from utils.neo_utils import time_slice
from utils.parse import parse_plot_channels, none_or_int, none_or_float


def plot_hist(asig, channel, bins=20, log=False):
    fig, ax = plt.subplots()
    palette = sns.color_palette()
    ax.hist(asig.as_array()[:,channel], bins=bins, density=True)  
    ax.set_ylabel('original signal', color=palette[0])
    ax.tick_params('y', colors=palette[0])
    #ax.plot(asig.as_array()[:,channel],stats.gaussian_kde(asig.as_array()[:,channel]))
    ax.set_title('Channel {}'.format(channel))
    ax.set_xlabel('a.u.')
    x=np.linspace(np.min(asig.as_array()[:,channel]), np.max(asig.as_array()[:,channel]), len(asig.as_array()[:, channel]))
    kde=stats.gaussian_kde(asig.as_array()[:,channel], bw_method=0.2)
    ax.plot(x, kde(x))

    if log:
        ax.set_yscale('log')
    return ax


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output figure")
    CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=0,
                     help="start time in seconds")
    CLI.add_argument("--t_stop",  nargs='?', type=none_or_float, default=None,
                     help="stop time in seconds")
    CLI.add_argument("--channels", nargs='+', type=none_or_int, default=0,
                     help="list of channels to plot")
    args, unknown = CLI.parse_known_args()
    
    print("ARGS:", args)
    asig = load_neo(args.data, 'analogsignal', lazy=True)

    channels = parse_plot_channels(args.channels, args.data)

    
    asig = time_slice(asig, t_start=args.t_start, t_stop=args.t_stop,
                      lazy=True, channel_indexes=channels)
    

    for ch in channels:
        ax = plot_hist(asig, ch, bins=40, log=True)
        plot_name=os.path.normpath(args.output+"/plot_hist_ch"+str(ch)+".png")
        #plot_name=args.output+"/plot_hist_ch"+str(ch)+".png"
        save_plot(plot_name)



