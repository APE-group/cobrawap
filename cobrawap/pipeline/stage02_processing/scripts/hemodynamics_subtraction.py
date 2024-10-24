"""

"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import neo
import quantities as pq
from skimage import data, io, filters, measure
from utils.parse import parse_string2dict, none_or_float, none_or_int, none_or_str
from utils.neo_utils import imagesequence_to_analogsignal, analogsignal_to_imagesequence, flip_image, rotate_image, time_slice
from utils.io_utils import load_neo, write_neo
import scipy

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='?', type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")

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
                                   #file_origin=images.file_origin),
                                   #**imgseq.annotations)
                                   units=imgseq_fluo.units,
                                   spatial_scale=imgseq_fluo.spatial_scale,
                                   sampling_rate=imgseq_fluo.sampling_rate,
                                   file_origin=imgseq_fluo.file_origin,
                                   t_start=imgseq_fluo.t_start)

    if 'array_annotations' in imgseq_fluo.annotations:
        del imgseq_fluo.annotations['array_annotations']

    imgseq_corrected.annotations.update(imgseq_fluo.annotations)
    if imgseq_fluo.name:
        imgseq_corrected.name = imgseq_fluo.name
    imgseq_corrected.description = imgseq_fluo.description + \
                                   "hemodynamic correction ({}).".format(os.path.basename(__file__))

    return imgseq_corrected


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    imgseq_fluo = analogsignal_to_imagesequence(block.segments[0].analogsignals[0])
    imgseq_refl = analogsignal_to_imagesequence(block.segments[0].analogsignals[1])

    imgseq_corrected = hemodyn_correction(imgseq_fluo, imgseq_refl)

    asig_corrected = imagesequence_to_analogsignal(imgseq_corrected)

    block.segments[0].analogsignals[0] = asig_corrected

    write_neo(args.output, block)
