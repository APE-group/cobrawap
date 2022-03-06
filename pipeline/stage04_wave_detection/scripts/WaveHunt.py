import numpy as np
import quantities as pq
import argparse
import matplotlib.pyplot as plt

from utils.io import load_neo, write_neo, save_plot
from utils.neo import remove_annotations, analogsignals_to_imagesequences
from utils.parse import none_or_str, none_or_float

import neo

from WaveHuntUtils import timelag_optimization, iwi_optimization, 
	RemoveSmallWaves, CleanWave, Neighbourhood_Search, PlotDetectedWaves
# =======================================================================================#

if __name__ == '__main__':
    
#--- Parse CLI (input parameters) -------------------------------------------------------- 
    CLI = argparse.ArgumentParser(description=__doc__,
                      formatter_class=argparse.RawDescriptionHelpFormatter)

    #--- Input & Output
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                        help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                        help="path of output file")
    CLI.add_argument("--output_img",  nargs='?', type=none_or_str,
                     help="path of output image", default=None)

    #--- Algorythm Parameters 
    CLI.add_argument("--max_abs_timelag", nargs='?', type=float, default=0.8, #units: [s]
                        help="Max. reasonable timelag among channels (electrodes/pixels)")
    CLI.add_argument("--acceptable_rejection_rate", nargs='?', type=float, default=0.1, #10%
                        help="acceptable rejection rate when optimizing iwi parameter")
    # N.B. parameter should be anchored to physiology inputs and/or inputs from data 
    CLI.add_argument("--min_ch_fraction", nargs='?', type=float, default=0.5,
                        help="minimum percentage of active channels involved in a wave")

    args = CLI.parse_args()

#--- Load Input Data 
#    Collect Geometry Information from data ----------------------------------------------

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    spatial_scale = asig.annotations['spatial_scale'] 
    # spatial_scale = pixel size (for regularly-spaced channel arrays) 

    #--- Get center of mass (cm) coordinates for each signal
    try:
    # This is the case of "smart" sampling, channel size derived from HOS (Hierarchical Optimal Sampling)
        coords = {'x': asig.array_annotations['x_coord_cm'],
                  'y': asig.array_annotations['y_coord_cm'],
                  'radius': asig.array_annotations['pixel_coordinates_L']}
    except KeyError:
    # This is the case of a regular grid/array with regular channel spacing
        coords = {'x': (asig.array_annotations['x_coords']+0.5)*spatial_scale,
                  'y': (asig.array_annotations['y_coords']+0.5)*spatial_scale,
                  'radius': np.ones([len(asig.array_annotations['x_coords'])])}
   
    #---
    block = analogsignals_to_imagesequences(block)
    imgseq = block.segments[0].imagesequences[-1]
    
    dim_x, dim_y = np.shape(imgseq[0])

    #--- Get Events    
    evts = [ev for ev in block.segments[0].events if ev.name== 'transitions']
    if len(evts):
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")

#--- WaveHunt Preliminary Measurements ---------------------------------------------------
# ExpectedTrans = number of expected Waves in the WaveCollection --> used to estimate/optimize IWI
# IWI = Inter Wave Interval = time distance between two consecutive candidate waves
# nCh = total number of active channels, i.e. channels for which an upward transition has been reported at least once    

    TransPerCh_Idx, TransPerCh_Num = np.unique(evts.array_annotations['channels'], return_counts=True)
    ExpectedTrans = np.median(TransPerCh_Num[np.where(TransPerCh_Num != 0)])
    print('Expected Transitions: ', ExpectedTrans)
     
    nCh = len(np.unique(evts.array_annotations['channels'])) # total number of active channels
    print('Active Channels: ', nCh)

# neighbors --> used when facing the "unicity" issue
    neighbors = Neighbourhood_Search(coords, evts.annotations['spatial_scale'])

#--- WaveHunt Core Actions ---------------------------------------------------------------

    # 1) search for the optimal abs timelag
    Waves_Inter = timelag_optimization(evts, args.max_abs_timelag)
   
    # 2) search for the best max_iwi parameter
    Waves_Inter = iwi_optimization(Waves_Inter, ExpectedTrans, args.min_ch_fraction, nCh, args.acceptable_rejection_rate)

    # 3) Unicity principle refinement
    Waves_Inter = CleanWave(evts.times, evts.array_annotations['channels'], neighbors, Waves_Inter)

    # 4) Globality principle
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
                                       'x_coords': asig.array_annotations['x_coords'][Pixels],
                                       'y_coords': asig.array_annotations['y_coords'][Pixels],
                                       'size': coords['radius'][Pixels]},

                    description='Transitions from down to up states. '\
                               +'Labels are ids of wavefronts. '
                               +'Annotated with the channel id ("channels") and '\
                               +'its position ("x_coords", "y_coords").',
                    spatial_scale = evts.annotations['spatial_scale'])

    remove_annotations(evts, del_keys=['nix_name', 'neo_name'])
    waves.annotations.update(evts.annotations)
    
    if args.output_img is not None:
        PlotDetectedWaves(evts, waves)
