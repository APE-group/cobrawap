"""
ToDo: write docstring
"""
import numpy as np
import neo
import matplotlib.pyplot as plt
import json
import os
import sys
import argparse
import quantities as pq
import scipy.io as sio
from utils import parse_string2dict, ImageSequence2AnalogSignal
from utils import none_or_float, none_or_int, none_or_str, load_neo, write_neo, time_slice
from utils import flip_image, rotate_image


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--data_name", nargs='?', type=str, default='None',
                     help="chosen name of the dataset")
    CLI.add_argument("--sampling_rate", nargs='?', type=none_or_float,
                     default=None, help="sampling rate in Hz")
    CLI.add_argument("--spatial_scale", nargs='?', type=float, required=True,
                     help="distance between electrodes or pixels in mm")
    CLI.add_argument("--t_start", nargs='?', type=none_or_float, default=None,
                     help="start time, in s, delimits the interval of recordings to be analysed")
    CLI.add_argument("--t_stop", nargs='?', type=none_or_float, default=None,
                     help="stop time, in s, delimits the interval of recordings to be analysed")
    CLI.add_argument("--orientation_top", nargs='?', type=str, required=True,
                     help="upward orientation of the recorded cortical region")
    CLI.add_argument("--orientation_right", nargs='?', type=str, required=True,
                     help="right-facing orientation of the recorded cortical region")
    CLI.add_argument("--annotations", nargs='+', type=none_or_str, default=None,
                     help="metadata of the dataset")
    CLI.add_argument("--array_annotations", nargs='+', type=none_or_str,
                     default=None, help="channel-wise metadata")
    CLI.add_argument("--kwargs", nargs='+', type=none_or_str, default=None,
                     help="additional optional arguments")
    CLI.add_argument("--emodynamics_correction", nargs='+', type=bool, default=False,
                     help="whether emodynamics correction is applicable")
    CLI.add_argument("--path_to_reflectance_data", nargs='+', type=str, default=None,
                     help="path to reflectance data")
    
    args = CLI.parse_args()

    # Load data
    # loading the data flips the images vertically
    file_path = args.data    
    mat_file_name = file_path
    
    s = np.array(sio.loadmat(mat_file_name)['s'])
    signal = s.T
    signal = signal.astype(float) 
    
    imageSequences = neo.ImageSequence(signal, sampling_rate = args.sampling_rate* pq.Hz,
                       spatial_scale = args.spatial_scale * pq.mm,
                       units='dimensionless')
                       
    
    block = neo.Block()
    seg = neo.Segment(name='segment 0', index=0)
    block.segments.append(seg)
    block.segments[0].imagesequences.append(imageSequences)

    # appendo segnale rifratto
    # Load reflectance data
    # loading the  data flips the images vertically
    file_path = args.path_to_reflectance_data
    mat_file_name = file_path
    r = np.array(sio.loadmat(mat_file_name[0])['s'])
    # for time reasons only 100 frames are selected
    signal_ref = r.T
    signal_ref = signal_ref.astype(float)
    
    imageSequences_Reflectance = neo.ImageSequence(signal_ref, sampling_rate = args.sampling_rate* pq.Hz,
                       spatial_scale = args.spatial_scale * pq.mm,
                       units='dimensionless')
                       
    
    block.segments[0].imagesequences.append(imageSequences_Reflectance)


    # change data orientation to be top=ventral, right=lateral
    imgseq = block.segments[0].imagesequences[0]
    imgseq = flip_image(imgseq, axis=-2)
    imgseq = rotate_image(imgseq, rotation=-90)
    block.segments[0].imagesequences[0] = imgseq
    
    #ruoto segnale rifratto
    imgseq_ref = block.segments[0].imagesequences[1]
    imgseq_ref = flip_image(imgseq_ref, axis=-2)
    imgseq_ref = rotate_image(imgseq_ref, rotation=-90)
    block.segments[0].imagesequences[1] = imgseq_ref
    

    # Transform into analogsignals
    block.segments[0].analogsignals = []
    block = ImageSequence2AnalogSignal(block)

    block.segments[0].analogsignals[0] = time_slice(
                block.segments[0].analogsignals[0], args.t_start, args.t_stop)
    

    # Add metadata from ANNOTATION dict
    if args.annotations is not None:
        block.segments[0].analogsignals[0].annotations.\
                                    update(parse_string2dict(args.annotations))

    
    
    block.segments[0].analogsignals[0].annotations.update(orientation_top=args.orientation_top)
    block.segments[0].analogsignals[0].annotations.update(orientation_right=args.orientation_right)
    


    # Add description to the Neo object
    block.name = args.data_name
    block.segments[0].name = 'Segment 1'
    block.segments[0].description = 'Loaded from mat file. '\
                                    .format(neo.__version__)
    
    if block.segments[0].analogsignals[0].description is None:
        block.segments[0].analogsignals[0].description = ''
    block.segments[0].analogsignals[0].description += 'Ca+ imaging signal '
    
    # Save data to file
    write_neo(args.output, block)
