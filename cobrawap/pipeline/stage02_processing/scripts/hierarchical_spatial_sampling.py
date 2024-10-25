"""
Performs a dynamical downsampling, based on a (customizable) 'quality' evaluation for each channel.
"""

import neo
import numpy as np

import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import warnings
from scipy.stats import (
    ks_1samp,
    ks_2samp,
    shapiro
)
from utils.io_utils import (
    load_neo,
    save_plot,
    write_neo
)
from utils.neo_utils import (
    analogsignal_to_imagesequence,
    imagesequence_to_analogsignal
)
from utils.parse import (
    none_or_float,
    none_or_int,
    none_or_str
)


CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs="?", type=Path, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs="?", type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs="?", type=Path, required=True,
                 help="path of output image", default=None)
CLI.add_argument("--exit_condition", nargs="?", type=none_or_str,
                 choices=["voting","consecutive"], default="consecutive",
                 help="exit condition in the optimal macro-pixel dimension tree search")
CLI.add_argument("--voting_threshold", nargs="?", type=none_or_float, default=0.5,
                 help="threshold of non-informative nodes percentage if \"voting\" method is selected")
CLI.add_argument("--n_bad_nodes", nargs="?", type=none_or_int, default=2,
                 help="number of non-informative nodes to prune branch")
CLI.add_argument("--evaluation_method", nargs="?", type=none_or_str,
                 choices=["shapiro","shapiroplus"], default="shapiro",
                 help="signal-to-noise ratio evaluation method")
CLI.add_argument("--output_array", nargs="?", type=Path, required=True,
                 help="path of output numpy array")


# Definition of custom versions of nanmean and nanstd,
# where automatically catching and ignoring RuntimeWarning warnings
def silent_nanmean(arr, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, **kwargs)


def silent_nanstd(arr, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(arr, **kwargs)


def next_power_of_2(n):
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1


def ComputeCenterOfMass(s, scale):
    # compute the center of mass of a macropixel con nan values
    mean = silent_nanmean(s, axis = 2)
    idx = np.where(~np.isnan(mean))
    x_cm = (np.mean(idx[0])+0.5)*scale
    y_cm = (np.mean(idx[1])+0.5)*scale
    if np.isnan(x_cm): x_cm = np.shape(mean)[0]/2
    if np.isnan(y_cm): y_cm = np.shape(mean)[1]/2
    return(x_cm, y_cm)


def above_th_points(y, sampling_frequency, shapiro_plus_th):
    m = silent_nanmean(y)
    sigma = silent_nanstd(y)
    th_min = m + sigma*shapiro_plus_th
    ind_min = np.where(y>th_min)[0]
    return [_/sampling_frequency for _ in ind_min]


def InitShapiroPlus(Input_image, sampling_frequency, shapiro_plus_th):
    means = silent_nanmean(Input_image, axis = 2)
    stds = silent_nanstd(Input_image, axis = 2)
    m = silent_nanmean(means)
    s = silent_nanmean(stds)
    rs = []
    for i in range(np.shape(Input_image)[0]):
        for j in range(np.shape(Input_image)[1]):
            y_0 = Input_image[i,j,:]
            if not np.isnan(y_0).all():
                y_1 = np.roll(y_0,-1)
                rs.append(np.sum(y_0*y_1)/np.sum(y_0*y_0))
    r_mean = silent_nanmean(rs)
    r_std = silent_nanstd(rs)

    # loop per generare N canali sintetici
    bins = np.arange(0, (np.shape(Input_image)[2]+0.5)/sampling_frequency, 0.1)
    bin_size = bins[1]-bins[0]

    av_h = np.zeros(len(bins)-1)
    c = 0

    for i in range(10000):
        red_signal = [np.random.normal(loc=m, scale=s) for t in range(np.shape(Input_image)[2])]
        r = r_mean
        #r = np.random.normal(loc=r_mean, scale=r_std)
        for t in range(1,len(red_signal)):
            red_signal[t] = r*red_signal[t-1] + np.sqrt(1-r**2)*red_signal[t]

        y = np.diff(above_th_points(red_signal, sampling_frequency, shapiro_plus_th))

        if len(y) > 0:
            h,b = np.histogram(y, bins=bins, density=True)
            if not np.isnan(np.sum(h)):
                av_h += h
                c += 1
    av_h /= c

    z = np.polyfit(b[:-1], np.cumsum(av_h)*bin_size, 4)
    zz = np.poly1d(z)
    return zz


def EvaluateShapiro(value):
    stat, p = shapiro(value)
    return p


def EvaluateShapiroPlus(value, cumul_distr, sampling_frequency, shapiro_plus_th):
    y = np.diff(above_th_points(value, sampling_frequency, shapiro_plus_th))
    try:
        stat, p = ks_1samp(y, cumul_distr)
        return p
    except Exception:
        return np.nan


def CheckCondition(coords, Input_image, sampling_frequency, evaluation_method, null_distr=None, shapiro_plus_th=None):
    # function to check whether node is compliant with the condition
    value = silent_nanmean(Input_image[coords[0]:coords[0]+coords[2], coords[1]:coords[1]+coords[2]], axis=(0,1))
    # if np.isnan(np.nanmax(value)):
    if np.isnan(value).all():
        return 1
    else:
        if evaluation_method == "shapiro":
            p = EvaluateShapiro(value)
            if p <= 0.05:
                return 0
            else:
                # the pixel is classified as noise
                return 1
        elif evaluation_method == "shapiroplus":
            p_1 = EvaluateShapiro(value)
            p_2 = EvaluateShapiroPlus(value, null_distr, sampling_frequency, shapiro_plus_th)
            if (p_1 > 0.05) and ((p_2 > 0.05) or (p_2 is np.nan)):
                # the pixel is classified as noisy
                return 1
            else:
                return 0


def NewLayer(l, Input_image, sampling_frequency, evaluation_method, null_distr=None, shapiro_plus_th=None):

    new_list = []
    x0 = l[0]
    y0 = l[1]
    L0 = l[2]
    half_L0 = L0//2

    for col in range(2):
        for row in range(2):
            x = x0 + col*half_L0
            y = y0 + row*half_L0
            cond = CheckCondition([x, y, half_L0], Input_image, sampling_frequency, evaluation_method, null_distr, shapiro_plus_th)
            new_list.append([x, y, half_L0, (l[3]+cond)*cond, x0, y0, L0])

    return new_list


def CreateMacroPixel(Input_image, sampling_frequency, exit_condition, evaluation_method, voting_threshold, n_bad_nodes, null_distr=None, shapiro_plus_th=None):
    # initialized node list
    NodeList = []
    MacroPixelCoords = []

    # initialized root
    NodeList.append([0, 0, np.shape(Input_image)[0], 0, 0, 0, np.shape(Input_image)[0]])

    while len(NodeList):

        # create node's children
        Children = NewLayer(NodeList[0], Input_image, sampling_frequency, evaluation_method, null_distr, shapiro_plus_th)
        NodeList.pop(0) # delete investigated node

        #check wether exit condition is met
        if exit_condition == "voting":
            # check how many of children are "bad"
            flag_list = [np.int32(ch[3]>= n_bad_nodes) for ch in Children]
            if np.sum(flag_list) > voting_threshold*len(Children):
                MacroPixelCoords.append(Children[0][4:]) # store parent node
                Children = []
        elif exit_condition == "consecutive":
            # check if some or all children are "bad"
            flag_list = [ch[3]==n_bad_nodes for ch in Children]
            if all(flag_list): # if all children are "bad"
                MacroPixelCoords.append(Children[0][4:]) # store parent node
                Children = []
            else:
                Children = [ch for ch in Children if ch[3] != n_bad_nodes]

        # check whether minimum dimension has been reached
        l_list = [ch[2] == 1 for ch in Children]
        idx = np.where(l_list)[0]
        if len(idx):
            for i in range(0, len(l_list)):
                # check whether minimum-dimension pixels are actually included in the roi
                if l_list[i] == True and not np.isnan(Input_image[Children[i][0],Children[i][1]]).all():
                    MacroPixelCoords.append(Children[i][0:3])
            Children = [ch for ch in Children if ch[2] != 1]

        NodeList.extend(Children)

    return MacroPixelCoords


def plot_masked_image(original_img, MacroPixelCoords):

    NewImage = np.empty([np.shape(original_img)[0], np.shape(original_img)[0]])*np.nan
    for macro in MacroPixelCoords:
        # fill pixels belonging to the same macropixel with the same signal
        m = np.mean(silent_nanmean(original_img[macro[0]:macro[0]+macro[2], macro[1]:macro[1]+macro[2]], axis=(0,1)))
        NewImage[macro[0]:macro[0]+macro[2], macro[1]:macro[1]+macro[2]] = m

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(6,2, forward=True)
    im = axs[0].imshow(silent_nanmean(original_img, axis = 2))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title('Original image', fontsize = 7.)

    im = axs[1].imshow(NewImage)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title('Post sampling', fontsize = 7.)

    log2_sizes = [int(np.log2(mp[2])) for mp in MacroPixelCoords]
    unique, counts = np.unique(log2_sizes, return_counts=True)
    axs[2].bar(unique, counts, width=0.6)
    axs[2].set_yscale('log')
    axs[2].set_xlim([-0.5+np.min(unique),0.5+np.max(unique)])
    axs[2].set_xticks(unique, [str(int(_)) for _ in unique])
    axs[2].set_xlabel('macro-pixel size (log2)', fontsize = 7.)

    plt.tight_layout()
    return axs


if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    #block = analogsignal_to_imagesequence(block)
    sampling_frequency = asig.sampling_rate.magnitude

    # load image sequences at the original spatial resolution
    imgseq = analogsignal_to_imagesequence(asig)
    imgseq_array = np.swapaxes(imgseq.as_array().T, 0, 1)
    dim_x, dim_y, dim_t = imgseq_array.shape

    if args.evaluation_method == "shapiroplus":
        shapiro_plus_th = 2.5
        null_distr = InitShapiroPlus(imgseq_array, sampling_frequency, shapiro_plus_th)
    elif args.evaluation_method == "shapiro":
        shapiro_plus_th = None
        null_distr = None

    # pad image sequences with nans to make it divisible by 2
    N_pad = next_power_of_2(max(dim_x, dim_y))
    padded_image_seq = np.pad(imgseq_array,
                         pad_width = [((N_pad-dim_x)//2, (N_pad-dim_x)//2 + (N_pad-dim_x)%2),
                                      ((N_pad-dim_y)//2, (N_pad-dim_y)//2 + (N_pad-dim_y)%2),
                                      (0,0)], mode = 'constant', constant_values = np.nan)

    # Tree search for the best macro-pixel dimension
    # MacroPixelCoords is a list of lists, each for a macro-pixel,
    # containing following metrics: x, y, L, flag, x_parent, y_parent, L_parent
    MacroPixelCoords = CreateMacroPixel(Input_image = padded_image_seq,
                                        sampling_frequency = sampling_frequency,
                                        exit_condition = args.exit_condition,
                                        evaluation_method = args.evaluation_method,
                                        voting_threshold = args.voting_threshold,
                                        n_bad_nodes = args.n_bad_nodes,
                                        null_distr = null_distr,
                                        shapiro_plus_th = shapiro_plus_th)

    plot_masked_image(padded_image_seq, MacroPixelCoords)
    save_plot(args.output_img)

    N_MacroPixel = len(MacroPixelCoords)
    signal = np.empty([N_MacroPixel, dim_t]) # save data as analogsignal
    coordinates = np.empty([N_MacroPixel, 3]) # pixel coordinates [x,y,L] to retrieve original one
    ch_id = np.empty([N_MacroPixel]) # new channel id
    x_coord = np.empty([N_MacroPixel]) # new x coord
    y_coord = np.empty([N_MacroPixel]) # new y coord
    x_coord_cm = np.empty([N_MacroPixel]) # new x coord
    y_coord_cm = np.empty([N_MacroPixel]) # new y coord

    # for each macro-pixel
    for px_idx, px in enumerate(MacroPixelCoords):
        signal[px_idx, :] = silent_nanmean(padded_image_seq[px[0]:px[0]+px[2],
                                           px[1]:px[1]+px[2]], axis = (0,1))
        x_coord_cm[px_idx], y_coord_cm[px_idx] = \
            ComputeCenterOfMass(padded_image_seq[px[0]:px[0]+px[2], px[1]:px[1]+px[2]],
                                imgseq.spatial_scale)

        coordinates[px_idx] = px
        ch_id[px_idx] = px_idx
        x_coord[px_idx] = (px[0] + px[2]/2.)*imgseq.spatial_scale
        y_coord[px_idx] = (px[1] + px[2]/2.)*imgseq.spatial_scale

    new_evt_ann = {'x_coords': coordinates.T[0],
                   'y_coords': coordinates.T[1],
                   'x_coord_cm': x_coord_cm,
                   'y_coord_cm': y_coord_cm,
                   'pixel_coordinates_L': coordinates.T[2],
                   'channel_id': ch_id}


    new_asig = asig.duplicate_with_new_data(signal.T)
    new_asig.array_annotations.update(new_evt_ann)
    new_asig.description += "Non homogeneous downsampling obtained by checking " + \
                            "the signal-to-noise ratio of macropixels at different sizes."
    block.segments[0].analogsignals[0] = new_asig

    # ToDo:
    # use the CLI arg output_array for saving HOS summary stats

    write_neo(args.output, block)
