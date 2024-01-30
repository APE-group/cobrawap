import numpy as np
import quantities as pq
import argparse
import matplotlib.pyplot as plt

#--- utils_path --------------------------------------------------------------------------
# (for test purposes, comment out when done)
from pathlib import Path
import sys
import os
#utils_path = '/qst0/home1/homedirs/gdebonis/wavescalephant/pipeline/
#sys.path.append(utils_path)
sys.path.append(os.path.join(os.getcwd(), '../..'))
#-----------------------------------------------------------------------------------------

from utils.io import load_neo, write_neo, save_plot
from utils.neo import remove_annotations, analogsignals_to_imagesequences
from utils.parse import none_or_str, none_or_float

import neo

from WaveHuntUtils import (timelag_optimization, iwi_optimization,
        RemoveSmallWaves, CleanWave, Neighbourhood_Search, PlotDetectedWaves)

import random
import glob

# =======================================================================================#
def byebye():
    print('byebye')
    exit()

#------------------------------------------------------------------------
def clearFolder(PathToFolder):
    
    filelist=PathToFolder + '*'
    files = glob.glob(filelist)
    for f in files:
      os.remove(f)
    return

#------------------------------------------------------------------------
def PlotRandomWaves(waves, nPlot, OutputPath):

    widx=random.sample(range(len(waves)), nPlot)

    ymax=0 
    for i in range(len(waves)):
      if max(waves[i]['ch']) > ymax: ymax = max(waves[i]['ch'])

    for i in widx:
      filename = 'wave_'+str(i)
      output = os.path.join(OutputPath, filename)
      plt.figure(figsize=(9, 9))
      plt.plot(waves[i]['times'], waves[i]['ch'], color='black', marker='.', markersize=8, linestyle = 'None')
      plt.xlabel('time (s)', fontsize = 7.)
      plt.ylabel('channel id', fontsize = 7.)
      plt.ylim([0,ymax])
      save_plot(output)
    
    return nPlot

# =======================================================================================#
#--- default input and output
# (for test purposes, remove/comment out when done)
##testInput = '/qst0/home1/homedirs/gdebonis/test-WaveHunt/Output/LENS/stage03_trigger_detection/trigger_times.nix'
##testInput = '/Users/giuliadebonis/gdb/test-WaveHunt/test/input/transitions_downsampling.nix' 
testInput = '/Users/giuliadebonis/gdb/test-CoBraWAP/test-WH-FunctionalAreas/transitions.pkl'

# [options for test: /transitions_nativeReso.nix|transitions_downsampling.nix|transitions_HOS.nix]
##testOutput = '/qst0/home1/homedirs/gdebonis/test-WaveHunt/test/output/waves.nix'
##OutputPath = '/Users/giuliadebonis/gdb/test-WaveHunt/test/output/'
OutputPath = '/Users/giuliadebonis/gdb/test-CoBraWAP/test-WH-FunctionalAreas/Output'
testOutput = '/Users/giuliadebonis/gdb/test-CoBraWAP/test-WH-FunctionalAreas/Output/waves.nix'
testImageOutput = '/Users/giuliadebonis/gdb/test-CoBraWAP/test-WH-FunctionalAreas/Output/image.jpg'
# =======================================================================================#

if __name__ == '__main__':

#--- Parse CLI (input parameters) --------------------------------------------------------    
    CLI = argparse.ArgumentParser(description=__doc__,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    
    #--- Input & Output
    CLI.add_argument("--data", nargs='?', type=str, required=False, default=testInput,
                        help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=False, default=testOutput,
                        help="path of output file")
    CLI.add_argument("--output_img",  nargs='?', type=none_or_str,
                        help="path of output image", default=testImageOutput)
    
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

    nSamples = np.shape(asig)[0]
    nChannels = np.shape(asig)[1]

    print('signal loaded:\n-- signal duration (nsamples) = ', nSamples)
    print('-- nChannels = ',nChannels)

    byebye()

    spatial_scale = asig.annotations['spatial_scale']
    # spatial_scale = pixel size (for regularly -spaced and/or -downsampled channel arrays) 
    # N.B. spatial_scale is set at native resolution for dataset processed with HOS

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

    # 'radius' is the size of the pixel, expressed in terms of number of pixels at the spatial resolution 
    # given by 'spatial_scale', e.g. radius = 2 corresponds to a downsampling into a 2 x 2 macropixel  

    #---
    block = analogsignals_to_imagesequences(block)
    imgseq = block.segments[0].imagesequences[-1]
   
    dim_x, dim_y = np.shape(imgseq[0]) # size of the array/grid expressed in number of channels/pixels

    #--- Printout geometry information and checks 
    print('spatial_scale: ', spatial_scale)
    print('dim_x, dim_y:', dim_x, dim_y)
    
    print('x: ', coords['x'])
    print('y: ', coords['y'])
    print('radius: ', coords['radius'])
    print('min_radius: ', min(coords['radius']))
    print('max_radius: ', max(coords['radius']))

    result = all(element == coords['radius'][0] for element in coords['radius'])
    if (result):
       print("--> homogenous sampling, i.e. regularly-spaced channel grid")
    else:
       print("--> spatial resolution is not constant, dataset went through Hierarchical Optimal Sampling (HOS)\
              \n('spatial_scale' is native spatial resolution, i.e. before the HOS)")

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

    ChRatio = nCh/nChannels
    print('--> Channel Ratio = ', ChRatio)
# ChRatio < 1 means that there are channels among those identified at stage02 (after ROI and downsampling, 
# or after ROI and then selected as non-noisy by HOS) for which no upward transitions (triggers) 
# have been identified at stage03. 

# neighbors --> used when facing the "unicity" issue
    neighbors = Neighbourhood_Search(coords, spatial_scale)

    #--- check if pixels have empty list of neighbors (relevant for HOS)
    emptyneighbors=0
    HowManyNeighbors=[]
    for i in range(len(neighbors)):
      HowManyNeighbors.append(len(neighbors[i]))
      if not neighbors[i]:
        print('WARNING: pixel ', i, '(radius: ', coords['radius'][i], 'has no neighbors')
        emptyneighbors+=1
    #print(HowManyNeighbors) 
    print('neigh=1', len(np.where(np.asarray(HowManyNeighbors)==1)[0]))    
    print('neigh=2', len(np.where(np.asarray(HowManyNeighbors)==2)[0]))
    print('neigh=3', len(np.where(np.asarray(HowManyNeighbors)==3)[0]))
    print('neigh=4', len(np.where(np.asarray(HowManyNeighbors)==4)[0]))
    print('neigh=5', len(np.where(np.asarray(HowManyNeighbors)==5)[0]))
    print('neigh=6', len(np.where(np.asarray(HowManyNeighbors)==6)[0]))
    print('neigh=7', len(np.where(np.asarray(HowManyNeighbors)==7)[0]))
    print('neigh=8', len(np.where(np.asarray(HowManyNeighbors)==8)[0]))
    print('neigh=9', len(np.where(np.asarray(HowManyNeighbors)==9)[0])) 
    print('neigh>9', len(np.where(np.asarray(HowManyNeighbors)>9)[0])) 
    print(len(HowManyNeighbors))
    if not emptyneighbors:
      print('Each pixel has a valid list of neighbors')

    byebye()

#--- WaveHunt Core Actions ---------------------------------------------------------------
    
    # 1) search for the optimal abs timelag
    # (intra-wave interval, i.e. time lag between triggers in the same candidate wave
    Waves_Inter = timelag_optimization(evts, args.max_abs_timelag) #WaveCollection_01 ('Inter' = intermediate step)
    ImgPath = OutputPath + 'WaveCollection_01/'
    clearFolder(ImgPath)
    PlotRandomWaves(Waves_Inter, 20, ImgPath)

    # 2) search for the best max_iwi parameter
    # (IWI = inter wave interval)
    Waves_Inter = iwi_optimization(Waves_Inter, ExpectedTrans, args.min_ch_fraction, nCh, args.acceptable_rejection_rate)
    ImgPath = OutputPath + 'WaveCollection_02/'
    clearFolder(ImgPath)
    PlotRandomWaves(Waves_Inter, 20, ImgPath)

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
        save_plot(args.output_img)

    block.segments[0].events.append(waves)
    write_neo(args.output, block)

