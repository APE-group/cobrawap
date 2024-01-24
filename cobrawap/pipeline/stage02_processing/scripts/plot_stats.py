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
import quantities as pq
import random
from utils.io import load_neo, save_plot
from utils.neo_utils import time_slice
from utils.parse import parse_plot_channels, none_or_int, none_or_float


def plot_stats(asig):
    mean_ch = np.mean(asig.as_array(),axis=0)
    median_ch = np.median(asig.as_array(), axis=0)
    diff_mean_median_ch = mean_ch - median_ch
    # std_ch = np.std(asig.as_array(), axis=0)
    fig, ax1 = plt.subplots()
    palette = sns.color_palette()
    print("media diff: ", np.mean(diff_mean_median_ch))
    print("mediana diff: ", np.median(diff_mean_median_ch))
    print("lunghezza asig:", len(asig.as_array()[0,:]))
    print("diff:", len(diff_mean_median_ch))
    print("i canali outliers sono:", np.where(diff_mean_median_ch<0.005))
    ax1.scatter(np.arange(0,len((diff_mean_median_ch))),diff_mean_median_ch)
    #ax1.errorbar(np.arange(1,len((diff_mean_median_ch))+1),diff_mean_median_ch, std_ch, fmt="o")
    ax1.set_ylabel('Difference', color=palette[0])
    ax1.tick_params('y', colors=palette[0])

    ax1.set_title('Difference between mean and median for each channel')
    ax1.set_xlabel('Channels')

    return ax1

def plot_box(asig):
    mean_ch = np.mean(asig.as_array(),axis=0)
    median_ch = np.median(asig.as_array(), axis=0)
    diff_mean_median_ch = mean_ch - median_ch
    fig, ax2 = plt.subplots()
    palette = sns.color_palette()
    ax2.boxplot(diff_mean_median_ch, showfliers=False, positions=[0])
    ax2.scatter(np.random.normal(0, 0.02, size=len(diff_mean_median_ch)), diff_mean_median_ch, marker='o',s=1, c='black')
    ax2.set_ylabel('Difference', color=palette[0])
    ax2.tick_params('y', colors=palette[0])

    ax2.set_title('Difference between mean and median for each channel')
    ax2.set_xlabel('Channels')
    outl=np.where(diff_mean_median_ch>0.006)
    #ax2.annotate(outl, xy=(0.1, diff_mean_median_ch[outl]))
    return ax2


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



    ax1 = plot_stats(asig)
    plot_name1=args.output+"/plot_stats.png"
    save_plot(plot_name1)

    ax2 = plot_box(asig)
    #plot_name=os.path.normpath(args.output+"/plot_hist_ch"+str(ch)+".png")
    plot_name2=args.output+"/plot_box.png"
    save_plot(plot_name2)



