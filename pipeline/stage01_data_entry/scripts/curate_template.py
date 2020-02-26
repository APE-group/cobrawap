"""
ToDo: write docstring
"""

import numpy as np
import argparse
import neo
import quantities as pq
import os
import sys
from utils import parse_string2dict, ImageSequence2AnalogSignal, none_or_float,\
                  none_or_int, load_neo, write_neo, time_slice


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--sampling_rate", nargs='?', type=none_or_float,
                     default=None, help="sampling rate in Hz")
    CLI.add_argument("--spatial_scale", nargs='?', type=float, required=True,
                     help="distance between electrodes or pixels in mm")
    CLI.add_argument("--t_start", nargs='?', type=none_or_float, required=True,
                     help="start time, in s, delimits the interval of recordings to be analysed")
    CLI.add_argument("--t_stop", nargs='?', type=none_or_float, required=True,
                     help="stop time, in s, delimits the interval of recordings to be analysed")
    CLI.add_argument("--data_name", nargs='?', type=str, default='None',
                     help="chosen name of the dataset")
    CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=0,
                     help="start time in seconds")
    CLI.add_argument("--t_stop",  nargs='?', type=none_or_float, default=10,
                     help="stop time in seconds")
    CLI.add_argument("--annotations", nargs='+', type=none_or_str, default=None,
                     help="metadata of the dataset")
    CLI.add_argument("--array_annotations", nargs='+', type=none_or_str,
                     default=None, help="channel-wise metadata")
    CLI.add_argument("--kwargs", nargs='+', type=none_or_str, default=None,
                     help="additional optional arguments")
    args = CLI.parse_args()

    # Load data with Neo IO or custom loading routine
    block = load_neo(args.data)
    # If there is no Neo IO for the data type available,
    # the data must be loaded conventioally and added to a newly constructed
    # Neo block. For building a Neo objects, have a look into the documentation
    # https://neo.readthedocs.io/

    # In case the dataset is imagaging data and therefore stored as an
    # ImageSequence object, it needs to be transformed into an AnalogSignal
    # object. To do this use the function ImageSequence2AnalogSignal in utils.py

    asig = block.segments[0].analogsignals[0]

    asig = time_slice(asig, args.t_start, args.t_stop)

    # Add metadata from ANNOTIATION dict
    asig.annotations.update(parse_string2dict(args.annotations))
    asig.annotations.update(spatial_scale=args.spatial_scale*pq.mm)

    # Add metadata from ARRAY_ANNOTIATION dict
    asig.array_annotations.update(parse_string2dict(args.array_annotations))

    # Do custom metadata processing from KWARGS dict (optional)
    # kwargs = parse_string2dict(args.kwargs)
    # ... do something

    # Add description to the Neo object
    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    block.segments[0].description = 'Loaded with neo.io version {}'\
                                    .format(neo.__version__)
    if asig.description is None:
        asig.description = ''
    asig.description += 'some signal. '

    # Update the annotated AnalogSignal object in the Neo Block
    block.segments[0].analogsignals[0] = asig

    # Save data into Nix file
    write_neo(args.output, block)
