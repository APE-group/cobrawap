"""
Performs a dynamical downsampling, based on a (customizable) quality evaluation for each channel.
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
    shapiro,
    zscore
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
    # check if python implementation exists
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1


def ComputeCenterOfMass(s, scale):
    # compute the center of mass of a macropixel con nan values
    avg = np.nanmean(s, axis=0)
    idx = np.where(~np.isnan(avg))
    y_cm = (np.mean(idx[0])+0.5)*scale
    x_cm = (np.mean(idx[1])+0.5)*scale
    if np.isnan(y_cm):
        y_cm = np.shape(avg)[0]/2
    if np.isnan(x_cm):
        x_cm = np.shape(avg)[1]/2
    return y_cm, x_cm


def above_thr_points(y, sampling_frequency, shapiro_plus_th):
    avg = silent_nanmean(y)
    sigma = silent_nanstd(y)
    th_min = avg + sigma*shapiro_plus_th
    ind_min = np.where(y>th_min)[0]
    return [_/sampling_frequency for _ in ind_min]


def InitShapiroPlus(Input_image, sampling_frequency, shapiro_plus_th):
    signal = zscore(Input_image, axis=0)
    avg = 0
    std = 1
    rs = []
    for j in range(np.shape(signal)[1]):
        for i in range(np.shape(signal)[2]):
            trace = signal[:,j,i]
            if not np.isnan(trace).all():
                rolled_trace = np.roll(trace,-1)
                rs.append(np.sum(trace*rolled_trace)/np.sum(trace*trace))
    r_mean = silent_nanmean(rs)
    r_std = silent_nanstd(rs)

    # loop per generare N canali sintetici
    bins = np.arange(0, (np.shape(signal)[0]+0.5)/sampling_frequency, 0.1)
    bin_size = bins[1]-bins[0]

    avg_hist = np.zeros(len(bins)-1)
    count = 0

    for i in range(10000):
        synth_signal = [np.random.normal(loc=avg, scale=std) for t in range(np.shape(signal)[0])]
        #r = r_mean
        r = np.random.normal(loc=r_mean, scale=r_std)
        for t in range(1,len(synth_signal)):
            synth_signal[t] = r*synth_signal[t-1] + np.sqrt(1-r**2)*synth_signal[t]

        intervals = np.diff(above_thr_points(synth_signal, sampling_frequency, shapiro_plus_th))

        if len(intervals) > 0:
            h,b = np.histogram(intervals, bins=bins, density=True)
            if not np.isnan(np.sum(h)):
                avg_hist += h
                count += 1
    avg_hist /= count

    z = np.polyfit(b[:-1], np.cumsum(avg_hist)*bin_size, 4)

    return np.poly1d(z)


def EvaluateShapiro(value):
    stat, p = shapiro(value)
    return p


def EvaluateShapiroPlus(trace, cumul_distr, sampling_frequency, shapiro_plus_th):
    intervals = np.diff(above_thr_points(zscore(trace), sampling_frequency, shapiro_plus_th))
    try:
        stat, p = ks_1samp(intervals, cumul_distr)
        return p
    except Exception:
        return np.nan


def CheckCondition(coords, Input_image, sampling_frequency, evaluation_method, null_distr=None, shapiro_plus_th=None):
    # function to check whether node is compliant with the condition
    # 0 is returned if pixel is informative, 1 otherwise
    mean_trace = silent_nanmean(Input_image[:, coords[1]:coords[1]+coords[2], coords[0]:coords[0]+coords[2]], axis=(1,2))
    if np.isnan(mean_trace).all():
        return 1
    else:
        if evaluation_method == "shapiro":
            p = EvaluateShapiro(mean_trace)
            if p <= 0.05:
                return 0
            else:
                return 1
        elif evaluation_method == "shapiroplus":
            p_1 = EvaluateShapiro(mean_trace)
            p_2 = EvaluateShapiroPlus(mean_trace, null_distr, sampling_frequency, shapiro_plus_th)
            # pixel is classified as non-informative if both tests fail
            if (p_1 > 0.05) and ((p_2 > 0.05) or (p_2 is np.nan)):
                return 1
            else:
                return 0


def coarseGrain(coords, Input_image, sampling_frequency, evaluation_method, null_distr=None, shapiro_plus_th=None):
    # function to compute how much a signal is significant, according to the chosen criterion

    mean_trace = silent_nanmean(Input_image[:, coords[1]:coords[1]+coords[2], coords[0]:coords[0]+coords[2]], axis=(1,2))
    if np.isnan(mean_trace).all():
        p_value = np.nan
    else:
        if evaluation_method == "shapiro":
            p_value = EvaluateShapiro(mean_trace)
        elif evaluation_method == "shapiroplus":
            p_1 = EvaluateShapiro(mean_trace)
            p_2 = EvaluateShapiroPlus(mean_trace, null_distr, sampling_frequency, shapiro_plus_th)
            p_value = np.nanmin([p_1,p_2])
    return mean_trace,p_value


def NewLayer(macropixel, Input_image, sampling_frequency, evaluation_method, null_distr=None, shapiro_plus_th=None):

    new_list = []
    x0 = macropixel[0]
    y0 = macropixel[1]
    L0 = macropixel[2]
    half_L0 = L0//2

    for col in range(2):
        for row in range(2):
            x = x0 + col*half_L0
            y = y0 + row*half_L0
            cond = CheckCondition([x, y, half_L0], Input_image, sampling_frequency, evaluation_method, null_distr, shapiro_plus_th)
            new_list.append([x, y, half_L0, (macropixel[3]+cond)*cond, x0, y0, L0])

    return new_list


def OldTopDown(Input_image, sampling_frequency, exit_condition, evaluation_method, voting_threshold, n_bad_nodes, null_distr=None, shapiro_plus_th=None):
    # initialized node list
    NodeList = []
    MacroPixelCoords = []

    # Elements in NodeList contain: x, y, L, flag, x_parent, y_parent, L_parent

    # initialized root
    NodeList.append([0, 0, np.shape(Input_image)[2], 0, 0, 0, np.shape(Input_image)[2]])

    while len(NodeList):

        # create node's children
        Children = NewLayer(NodeList[0], Input_image, sampling_frequency, evaluation_method, null_distr, shapiro_plus_th)
        NodeList.pop(0) # delete investigated node

        # check whether exit condition is met
        if exit_condition == "voting":
            # check how many of children are "bad"
            flag_list = [np.int32(ch[3]>=n_bad_nodes) for ch in Children]
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
        l_list = [ch[2]==1 for ch in Children]
        idx = np.where(l_list)[0]
        if len(idx):
            for i in range(len(l_list)):
                # check whether minimum-dimension pixels are actually included in the roi
                if l_list[i] == True and not np.isnan(Input_image[:, Children[i][1], Children[i][0]]).all():
                    MacroPixelCoords.append(Children[i][0:3])
            Children = [ch for ch in Children if ch[2] != 1]

        NodeList.extend(Children)

    return MacroPixelCoords


def build_layers(Input_image, sampling_frequency, exit_condition, evaluation_method, voting_threshold, n_bad_nodes, null_distr=None, shapiro_plus_th=None):

    # depth is the number of hierarchic layers when doubling macro-pixels size from 1 to the whole padded roi
    depth = int(np.log2(Input_image.shape[1]))
    p_values = []
    mean_signals = []

    # d=0 corresponds to largest resolution, i.e. smallest spatial scale
    for d in range(depth+1):
        pixel_size = 2**d
        matrix_size = Input_image.shape[1]//pixel_size
        mean_signal = np.empty([Input_image.shape[0],matrix_size,matrix_size])
        mean_signal[:,:,:] = np.nan
        p_value = np.empty([matrix_size,matrix_size])
        p_value[:,:] = np.nan
        for j in range(matrix_size):
            for i in range(matrix_size):
                mean_signal[:,j,i], p_value[j,i] = coarseGrain([pixel_size*i, pixel_size*j, pixel_size], Input_image, sampling_frequency, evaluation_method, null_distr, shapiro_plus_th)
        p_values.append(p_value)
        mean_signals.append(mean_signal)

    return mean_signals, p_values


def NewTopDown(Input_image, mean_signals, p_values):

    depth = int(np.log2(Input_image.shape[1]))
    MacroPixelCoords = []

    optimal_depth = np.empty([Input_image.shape[1],Input_image.shape[1]], dtype=int)
    optimal_depth[:,:] = depth

    # starting from L=2**depth and moving downwards, comparing at each step d with d-1
    for d in range(depth,0,-1):
        pixel_size = 2**d
        matrix_size = Input_image.shape[1]//pixel_size
        for j in range(matrix_size):
            for i in range(matrix_size):
                if not all(mean_signals[d][:,j,i]!=mean_signals[d][:,j,i]):
                    if (optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)]==d).all():
                        children = [(jj,ii) for jj in range(2*j,2*j+2) for ii in range(2*i,2*i+2)]
                        p_father = p_values[d][j,i]
                        p_children = [p_values[d-1][jj,ii] for (jj,ii) in children]
                        if all([_!=_ for _ in p_children]) or (p_father < np.nanmin(p_children) and p_father > 1e-4):
                            # keep father macro-pixel
                            optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)] = d
                            MacroPixelCoords.append([pixel_size*i,pixel_size*j,pixel_size])
                        else:
                            # move to children macro-pixels
                            optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)] = d-1
                            for ch,(jj,ii) in enumerate(children):
                                if d==1 and not all(mean_signals[d-1][:,jj,ii]!=mean_signals[d-1][:,jj,ii]):
                                    MacroPixelCoords.append([pixel_size//2*ii,pixel_size//2*jj,pixel_size//2])

    return MacroPixelCoords


def NewBottomUp(Input_image, mean_signals, p_values):

    depth = int(np.log2(Input_image.shape[1]))
    MacroPixelCoords = []

    optimal_depth = np.empty([Input_image.shape[1],Input_image.shape[1]], dtype=int)
    optimal_depth[:,:] = 0

    for d in range(1,depth+1):
        # starting from L=2 and moving upwards, comparing at each step d with d-1
        pixel_size = 2**d
        matrix_size = Input_image.shape[1]//pixel_size
        for j in range(matrix_size):
            for i in range(matrix_size):
                if not all(mean_signals[d][:,j,i]!=mean_signals[d][:,j,i]):
                    children = [(jj,ii) for jj in range(2*j,2*j+2) for ii in range(2*i,2*i+2)]
                    p_father = p_values[d][j,i]
                    p_children = [p_values[d-1][jj,ii] for (jj,ii) in children]
                    if (optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)]==d-1).all():
                        if any([_==_ for _ in p_children]) and (np.nanmin(p_children) <= p_father or (np.nanmin(p_children) > p_father and np.nanmin(p_children) < 1e-4)):
                            # keep children macro-pixels
                            optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)] = d-1
                            for ch,(jj,ii) in enumerate(children):
                                if not all(mean_signals[d-1][:,jj,ii]!=mean_signals[d-1][:,jj,ii]):
                                    MacroPixelCoords.append([pixel_size//2*ii,pixel_size//2*jj,pixel_size//2])
                        else:
                            # move to father macro-pixel
                            optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)] = d
                            if d==depth:
                                MacroPixelCoords.append([pixel_size*i,pixel_size*j,pixel_size])
                    else:
                        # some children already stopped at a higher resolution
                        for ch,(jj,ii) in enumerate(children):
                            if (optimal_depth[pixel_size//2*jj:pixel_size//2*(jj+1),pixel_size//2*ii:pixel_size//2*(ii+1)]==d-1).all() and not all(mean_signals[d-1][:,jj,ii]!=mean_signals[d-1][:,jj,ii]):
                                MacroPixelCoords.append([pixel_size//2*ii,pixel_size//2*jj,pixel_size//2])

    return MacroPixelCoords


def plot_masked_image(original_img, MacroPixelCoords):

    NewImage = np.empty([np.shape(original_img)[1], np.shape(original_img)[2]])*np.nan
    for mp in MacroPixelCoords:
        # fill pixels belonging to the same macropixel with the same signal
        avg = np.mean(silent_nanmean(original_img[:, mp[1]:mp[1]+mp[2], mp[0]:mp[0]+mp[2]], axis=(1,2)))
        NewImage[mp[1]:mp[1]+mp[2], mp[0]:mp[0]+mp[2]] = avg

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(6, 2, forward=True)
    im = axs[0].imshow(silent_nanmean(original_img, axis=0))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("Original image", fontsize=7)

    im = axs[1].imshow(NewImage)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("Post sampling", fontsize=7)

    log2_sizes = [int(np.log2(mp[2])) for mp in MacroPixelCoords]
    unique, counts = np.unique(log2_sizes, return_counts=True)
    axs[2].bar(unique, counts, width=0.6)
    axs[2].set_yscale("log")
    axs[2].set_xlim([-0.5+np.min(unique),0.5+np.max(unique)])
    axs[2].set_xticks(unique, [str(int(_)) for _ in unique])
    axs[2].set_xlabel("macro-pixel size (log2)", fontsize=7)

    plt.tight_layout()
    return axs


if __name__ == "__main__":
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]
    sampling_frequency = asig.sampling_rate.magnitude
    spatial_scale = asig.annotations["spatial_scale"]

    # load image sequences at the original spatial resolution
    imgseq = analogsignal_to_imagesequence(asig)
    imgseq_array = imgseq.as_array()
    dim_t, dim_y, dim_x = imgseq_array.shape
    if args.evaluation_method == "shapiroplus":
        shapiro_plus_th = 2.5
        null_distr = InitShapiroPlus(imgseq_array, sampling_frequency, shapiro_plus_th)
    elif args.evaluation_method == "shapiro":
        shapiro_plus_th = None
        null_distr = None

    # pad image sequences with nans to make it divisible by 2
    N_pad = next_power_of_2(max(dim_x, dim_y))
    padded_image_seq = np.pad(imgseq_array,
                              pad_width = [(0,0),
                                           ((N_pad-dim_y)//2, (N_pad-dim_y)//2+(N_pad-dim_y)%2),
                                           ((N_pad-dim_x)//2, (N_pad-dim_x)//2+(N_pad-dim_x)%2)],
                              mode="constant", constant_values=np.nan)

    # Tree search for the best macro-pixel dimension
    # MacroPixelCoords is a list of lists, each for a macro-pixel,
    # containing following metrics: x, y, L
    mean_signals, p_values = build_layers(Input_image = padded_image_seq,
                                          sampling_frequency = sampling_frequency,
                                          exit_condition = args.exit_condition,
                                          evaluation_method = args.evaluation_method,
                                          voting_threshold = args.voting_threshold,
                                          n_bad_nodes = args.n_bad_nodes,
                                          null_distr = null_distr,
                                          shapiro_plus_th = shapiro_plus_th)
    #MacroPixelCoords = NewTopDown(padded_image_seq, mean_signals, p_values)
    MacroPixelCoords = NewBottomUp(padded_image_seq, mean_signals, p_values)

    plot_masked_image(padded_image_seq, MacroPixelCoords)
    save_plot(args.output_img)

    N_MacroPixel = len(MacroPixelCoords)
    signal = np.empty([dim_t, N_MacroPixel])
    coordinates = np.empty([N_MacroPixel, 3]) # pixel coordinates [x,y,L] to retrieve original one
    ch_id = np.empty([N_MacroPixel])
    x_coord = np.empty([N_MacroPixel])
    y_coord = np.empty([N_MacroPixel])
    x_coord_cm = np.empty([N_MacroPixel])
    y_coord_cm = np.empty([N_MacroPixel])

    # for each macro-pixel px (containing x,y,L)
    for px_idx, px in enumerate(MacroPixelCoords):
        signal[:, px_idx] = np.nanmean(padded_image_seq[:, px[1]:px[1]+px[2],
                                                        px[0]:px[0]+px[2]],
                                       axis=(1,2))
        y_coord_cm[px_idx], x_coord_cm[px_idx] = \
            ComputeCenterOfMass(padded_image_seq[:, px[1]:px[1]+px[2], px[0]:px[0]+px[2]],
                                spatial_scale)
        coordinates[px_idx] = px
        ch_id[px_idx] = px_idx
        y_coord[px_idx] = (px[1]+0.5*px[2])*spatial_scale
        x_coord[px_idx] = (px[0]+0.5*px[2])*spatial_scale

    # macropixels details are stored as array_annotations
    mp_annot = {"x_coords": coordinates.T[0],
                "y_coords": coordinates.T[1],
                "x_coord_cm": x_coord_cm,
                "y_coord_cm": y_coord_cm,
                "pixel_coordinates_L": coordinates.T[2],
                "channel_id": ch_id}

    new_asig = asig.duplicate_with_new_data(signal)
    new_asig.array_annotations.update(mp_annot) # check!
    new_asig.description += "Non homogeneous downsampling obtained by checking " + \
                            "the signal-to-noise ratio of macropixels at different sizes."
    block.segments[0].analogsignals[0] = new_asig

    # ToDo:
    # use the CLI arg output_array for saving HOS summary stats
    # e.g. compute hist of mp linear sizes
    log2_sizes = [int(np.log2(mp[2])) for mp in MacroPixelCoords]
    unique, counts = np.unique(log2_sizes, return_counts=True)
    print(f"unique = {unique}, counts = {counts}")
    # if args.output_array is not None:
    np.save(args.output_array, np.array([unique,counts]))

    write_neo(args.output, block)
