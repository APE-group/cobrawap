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
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import quantities as pq
import random
from utils.io import load_neo, write_neo, save_plot
from utils.neo_utils import time_slice
from utils.parse import parse_plot_channels, none_or_int, none_or_float, none_or_str

CLI = argparse.ArgumentParser()
CLI.add_argument("--data",    nargs='?', type=Path, required=True,
                    help="path to input data in neo format")
CLI.add_argument("--output",  nargs='?', type=Path, required=True,
                    help="path of output file")
CLI.add_argument("--output_array",  nargs='?', type=none_or_str,
                    help="path of output numpy array", default=None)


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    signal = asig.as_array()
    mean = np.nanmean(signal, axis=0)
    std = np.nanstd(signal, axis=0)

    if args.output_array is not None:
        dct = {'mean': mean, 'std': std}
        if args.output_array is not None:
            np.save(args.output_array, dct)


    asig.array_annotate(mean=mean) #parte nuova
    asig.array_annotate(std=std) #parte nuova
    asig.name += ""
    block.segments[0].analogsignals[0] = asig
    print("Annotations", asig.array_annotations)
    write_neo(args.output, block)
