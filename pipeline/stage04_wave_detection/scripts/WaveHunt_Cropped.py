#import os
import numpy as np
import quantities as pq
import argparse
#import matplotlib.pyplot as plt
#import pandas as pd
#from scipy import io
#import math
from utils.io import load_neo, write_neo, save_plot
from utils.neo import remove_annotations, analogsignals_to_imagesequences
from utils.parse import none_or_str, none_or_float

import neo

from Params_optimization import timelag_optimization, iwi_optimization
from WaveCleaning import RemoveSmallWaves, CleanWave, Neighbourhood_Search

# ======================================================================================#

# LOAD input data
if __name__ == '__main__':
    
    CLI = argparse.ArgumentParser(description=__doc__,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                        help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                        help="path of output file")
    CLI.add_argument("--max_abs_timelag", nargs='?', type=float, default=0.8,
                        help="Maximum reasonable time lag between electrodes (pixels)")
    CLI.add_argument("--acceptable_rejection_rate", nargs='?', type=float, default=0.1,
                        help="acceptable rejection rate when optimizing iwi parameter")
    CLI.add_argument("--min_ch_fraction", nargs='?', type=float, default=0.5,
                        help="minimum percentage of active channels involved in a wave")
  
    # data loading

    args = CLI.parse_args()
    block = load_neo(args.data)

    # get center of mass coordinates for each signal
    try:
        coords = {'x': block.segments[0].analogsignals[0].array_annotations['x_coord_cm'],
                  'y': block.segments[0].analogsignals[0].array_annotations['y_coord_cm'],
                  'radius': block.segments[0].analogsignals[0].array_annotations['pixel_coordinates_L']}
    except KeyError:
        spatial_scale = block.segments[0].analogsignals[0].annotations['spatial_scale']
        coords = {'x': (block.segments[0].analogsignals[0].array_annotations['x_coords']+0.5)*spatial_scale,
                  'y': (block.segments[0].analogsignals[0].array_annotations['y_coords']+0.5)*spatial_scale,
                  'radius': np.ones([len(block.segments[0].analogsignals[0].array_annotations['x_coords'])])}
    block = analogsignals_to_imagesequences(block)
    imgseq = block.segments[0].imagesequences[-1]
    
    dim_x, dim_y = np.shape(imgseq[0])
    spatial_scale = imgseq.spatial_scale
    
    asig = block.segments[0].analogsignals[0]
    evts = [ev for ev in block.segments[0].events if ev.name== 'transitions']
    if len(evts):
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")
    

    # Preliminar measurements
    # ExpectedTrans i.e. estimate of the Number of Waves
    # ExpectedTrans used to estimate/optimize IWI
    TransPerCh_Idx, TransPerCh_Num = np.unique(evts.array_annotations['channels'], return_counts=True)
    ExpectedTrans = np.median(TransPerCh_Num[np.where(TransPerCh_Num != 0)]);
    print('Expected Transitions', ExpectedTrans)
    nCh = len(np.unique(evts.array_annotations['channels'])) # total number of channels

    neighbors = Neighbourhood_Search(coords, evts.annotations['spatial_scale'])
    
    # search for the optimal abs timelag
    Waves_Inter = timelag_optimization(evts, args.max_abs_timelag)
   
    # search for the best max_iwi parameter
    Waves_Inter = iwi_optimization(Waves_Inter, ExpectedTrans, args.min_ch_fraction, nCh, args.acceptable_rejection_rate)

    # Unicity principle refinement
    Waves_Inter = CleanWave(evts.times, evts.array_annotations['channels'], neighbors, Waves_Inter)

    # Globality principle
    Wave = RemoveSmallWaves(evts, args.min_ch_fraction, Waves_Inter, dim_x, dim_y)
    print('number of detected waves', len(Wave))

    Waves = []
    Times = []
    Label = []
    Pixels = []

    for i in range(0,len(Wave)):
        Times.extend(Wave[i]['times'].magnitude)
        Label.extend(np.ones([len(Wave[i]['ndx'])])*i)
        Pixels.extend(Wave[i]['ch'])

    Label = [str(i) for i in Label]
    Times = Times*(Wave[0]['times'].units)
    waves = neo.Event(times=Times.rescale(pq.s),
                    labels=[str(np.int32(np.float64(l))) for l in Label],
                    name='wavefronts',
                    array_annotations={'channels':Pixels,
                                       'x_coords':[p % dim_y for p in Pixels],
                                       'y_coords':[np.floor(p/dim_y) for p in Pixels]},
                    description='Transitions from down to up states. '\
                               +'Labels are ids of wavefronts. '
                               +'Annotated with the channel id ("channels") and '\
                               +'its position ("x_coords", "y_coords").',
                    spatial_scale = evts.annotations['spatial_scale'])

    remove_annotations(evts, del_keys=['nix_name', 'neo_name'])
    waves.annotations.update(evts.annotations)
    
    block.segments[0].events.append(waves)
    #remove_annotations(waves, del_keys=['nix_name', 'neo_name'])
    write_neo(args.output, block)
