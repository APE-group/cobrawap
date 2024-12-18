"""
Plots excerpts of the input data with its corresponding metadata.
"""

import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from utils.io_utils import (
    load_neo,
    save_plot
)
from utils.neo_utils import time_slice
from utils.parse import (
    none_or_float,
    none_or_int,
    parse_plot_channels
)

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output_img", nargs='?', type=Path, required=True,
                 help="path of output figure")
CLI.add_argument("--plot_tstart", nargs='?', type=none_or_float, default=0,
                 help="start time in seconds")
CLI.add_argument("--plot_tstop", nargs='?', type=none_or_float, default=10,
                 help="stop time in seconds")
CLI.add_argument("--plot_channels", nargs='+', type=none_or_int, default=0,
                 help="list of channels to plot")

def plot_traces(asig, channels):
    sns.set(style='ticks', palette="deep", context="notebook")
    fig, ax = plt.subplots()

    offset = np.max(np.abs(asig.as_array()[:,channels]))

    for i, signal in enumerate(asig.as_array()[:,channels].T):
        ax.plot(asig.times, signal + i*offset)

    annotations = [f'{k}: {v}' for k,v in asig.annotations.items()
                               if k not in ['nix_name', 'neo_name']]
    array_annotations = [f'{k}: {v[channels]}'
                        for k,v in asig.array_annotations.items()]

    x_coords = asig.array_annotations['x_coords']
    y_coords = asig.array_annotations['y_coords']
    dim_x, dim_y = np.max(x_coords)+1, np.max(y_coords)+1

    ax.text(ax.get_xlim()[1]*1.05, ax.get_ylim()[0],
            f'ANNOTATIONS FOR CHANNEL(s) {channels} \n'\
          +  '\n ANNOTATIONS:\n' + '\n'.join(annotations) \
          +  '\n\n ARRAY ANNOTATIONS:\n' + '\n'.join(array_annotations) +'\n' \
          + f' t_start: {asig.plot_tstart}; t_stop: {asig.plot_tstop} \n' \
          + f' dimensions(x,y): {dim_x}, {dim_y}')

    ax.set_xlabel(f'time [{asig.times.units.dimensionality.string}]')
    ax.set_ylabel(f'channels [in {asig.units.dimensionality.string}]')
    ax.set_yticks([i*offset for i in range(len(channels))])
    ax.set_yticklabels(channels)
    return ax


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    asig = load_neo(args.data, 'analogsignal', lazy=True)

    channels = parse_plot_channels(args.plot_channels, args.data)

    asig = time_slice(asig, t_start=args.plot_tstart, t_stop=args.plot_tstop,
                      lazy=True, channel_indexes=channels)

    ax = plot_traces(asig, channels)
    save_plot(args.output_img)
