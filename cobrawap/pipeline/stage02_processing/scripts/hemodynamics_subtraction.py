#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import neo
import quantities as pq
from skimage import data, io, filters, measure
from utils.parse import parse_string2dict, none_or_float, none_or_int, none_or_str
from utils.neo import imagesequences_to_analogsignals, analogsignals_to_imagesequences, flip_image, rotate_image, time_slice
from utils.io import load_neo, write_neo
import scipy

def hemodyn_correction(imgseq_fluo, imgseq_refl):

    # Applies the hemodynamic correction on the fluorescence data
   
    
    I = np.array(imgseq_fluo) 
    R = np.array(imgseq_refl)  

    ############  FLUORESCENCE  ############  
    
    I_mean = np.zeros((I.shape[1],I.shape[2]))
    I_over_I0 = np.zeros((I.shape[0],I.shape[1],I.shape[2]))

    

    I_mean = np.nanmean(I, axis = 0)
    #idx = np.where(I_mean == 0)
    #I_mean[idx] = np.nan
    I_over_I0 = I/I_mean[None,:,:]


    ############  REFLECTANCE  ############   
    
    R_mean = np.zeros((R.shape[1],R.shape[2]))
    F_over_F0 = np.zeros((I.shape[0],I.shape[1],I.shape[2]))
    dF_over_F0 = F_over_F0
    
    

    R_mean = np.nanmean(R, axis = 0)
    #idxR = np.where(R_mean == 0)
    #R_mean[idxR] = np.nan

    F_over_F0 = (I/I_mean[None,:,:])/(R/R_mean[None,:,:])
    dF_over_F0 = (F_over_F0-1)*100
          
    imgseq_corrected = neo.ImageSequence(dF_over_F0,
                                   #units=images.units,
                                   #spatial_scale=images.spatial_scale * macro_pixel_dim,
                                   #sampling_rate=images.sampling_rate,
                                   #file_origin=images.file_origin)#,
                                   #**imgseq.annotations)
                                   units=imgseq_fluo.units,
                                   spatial_scale=imgseq_fluo.spatial_scale,
                                   sampling_rate=imgseq_fluo.sampling_rate,
                                   file_origin=imgseq_fluo.file_origin,
                                   t_start=imgseq_fluo.t_start)

    if 'array_annotations' in imgseq_fluo.annotations:
        del imgseq_fluo.annotations['array_annotations']

    imgseq_corrected.annotations.update(imgseq_fluo.annotations)

    imgseq_corrected.name = imgseq_fluo.name + " "
    imgseq_corrected.description = imgseq_fluo.description +                  "hemodynamic correction ({}).".format(os.path.basename(__file__))

    return imgseq_corrected


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    
    args = CLI.parse_args()
    block = load_neo(args.data)
    block = analogsignals_to_imagesequences(block)
    imgseq_fluo = block.segments[0].imagesequences[0]
    imgseq_refl = block.segments[0].imagesequences[1]
    
    imgseq_corrected = hemodyn_correction(imgseq_fluo, imgseq_refl)

    new_block = neo.Block()
    new_segment = neo.Segment()
    new_block.segments.append(new_segment)
    new_block.segments[0].imagesequences.append(imgseq_corrected)
    new_block = imagesequences_to_analogsignals(new_block)

    block.segments[0].analogsignals[0] = new_block.segments[0].analogsignals[0]

    write_neo(args.output, block)

