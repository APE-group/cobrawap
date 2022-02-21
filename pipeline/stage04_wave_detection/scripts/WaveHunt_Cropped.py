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
    CLI.add_argument("--Max_Abs_Timelag", nargs='?', type=float, default=0.8,
                        help="Maximum reasonable time lag between electrodes (pixels)")
    CLI.add_argument("--Acceptable_rejection_rate", nargs='?', type=float, default=0.1,
                        help=" ")
    CLI.add_argument("--min_ch_num", nargs='?', type=float, default=300,
                        help="minimum number of channels involved in a wave")
  
    # data loading

    args = CLI.parse_args()
    block = load_neo(args.data)
    block = analogsignals_to_imagesequences(block)
    imgseq = block.segments[0].imagesequences[-1]
    
    dim_x, dim_y = np.shape(imgseq[0])
    spatial_scale = imgseq.spatial_scale
    
    asig = block.segments[0].analogsignals[-1]
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
    print('Expected number of waves', ExpectedTrans)
    nCh = len(np.unique(evts.array_annotations['channels'])) # total number of channels

    # search for the optimal abs timelag
    Waves_Inter = timelag_optimization(evts, args.Max_Abs_Timelag)
   
    # search for the best max_iwi parameter
    Waves_Inter = iwi_optimization(Waves_Inter, ExpectedTrans, nCh, args.Acceptable_rejection_rate)

    # Unicity principle refinement
    neighbors = Neighbourhood_Search(dim_x, dim_y)
    Waves_Inter = CleanWave(evts.times, evts.array_annotations['channels'], neighbors, Waves_Inter)

    # Globality principle
    Wave = RemoveSmallWaves(evts, args.min_ch_num, Waves_Inter, dim_x, dim_y)
    print('num waves', len(Wave))

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
