"""
Performs a dynamical downsampling, based on a (customizable) quality evaluation for each channel.
"""

import neo
import numpy as np
import pandas as pd
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
CLI.add_argument("--pruning_direction", nargs="?", type=none_or_str,
                 choices=["old-top-down","top-down","bottom-up"], default="top-down",
                 help="direction of hierarchic pruning of the tree")
CLI.add_argument("--pruning_method", nargs="?", type=none_or_str, default="all",
                 choices=["all","one","mean","majority","father_father_1e-4","father_child_1e-4"],
                 help="method of pruning in the new approach")
CLI.add_argument("--output_stats", nargs="?", type=Path, required=True,
                 help="path of output files with summary statistics")


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
        return 1,np.nan
    else:
        if evaluation_method == "shapiro":
            p = EvaluateShapiro(mean_trace)
            if p <= 0.05:
                return 0,p
            else:
                return 1,p
        elif evaluation_method == "shapiroplus":
            p_1 = EvaluateShapiro(mean_trace)
            p_2 = EvaluateShapiroPlus(mean_trace, null_distr, sampling_frequency, shapiro_plus_th)
            # pixel is classified as non-informative if both tests fail
            if (p_1 > 0.05) and ((p_2 > 0.05) or (p_2 is np.nan)):
                return 1,np.nanmin([p_1,p_2])
            else:
                return 0,np.nanmin([p_1,p_2])


def coarseGrain(coords, Input_image, sampling_frequency, evaluation_method, null_distr=None, shapiro_plus_th=None):
    # function to compute the coarse-grained signal and how much it is significant, according to the chosen criterion

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
    p0 = macropixel[3]
    half_L0 = L0//2

    for col in range(2):
        for row in range(2):
            x = x0 + col*half_L0
            y = y0 + row*half_L0
            cond, p_value = CheckCondition([x, y, half_L0], Input_image, sampling_frequency, evaluation_method, null_distr, shapiro_plus_th)
            new_list.append([x, y, half_L0, p_value, (macropixel[4]+cond)*cond, x0, y0, L0, p0])

    return new_list


def OldTopDown(Input_image, sampling_frequency, exit_condition, evaluation_method, voting_threshold, n_bad_nodes, null_distr=None, shapiro_plus_th=None):
    # initialized node list
    NodeList = []
    MacroPixelCoords = []

    # Elements in NodeList contain: x, y, L, p_value, flag, x_parent, y_parent, L_parent, p_value_parent

    # initialized root
    cond, p_value = CheckCondition([0, 0, np.shape(Input_image)[2]], Input_image, sampling_frequency, evaluation_method, null_distr, shapiro_plus_th)
    NodeList.append([0, 0, np.shape(Input_image)[2], p_value, cond, 0, 0, np.shape(Input_image)[2], p_value])

    while len(NodeList):

        # create node's children
        Children = NewLayer(NodeList[0], Input_image, sampling_frequency, evaluation_method, null_distr, shapiro_plus_th)
        NodeList.pop(0) # delete investigated node

        # check whether exit condition is met
        if exit_condition == "voting":
            # check how many of children are "bad"
            flag_list = [np.int32(ch[4]>=n_bad_nodes) for ch in Children]
            if np.sum(flag_list) > voting_threshold*len(Children):
                mp = Children[0][5:]
                MacroPixelCoords.append(mp) # store parent node
                Children = []
        elif exit_condition == "consecutive":
            # check if some or all children are "bad"
            flag_list = [ch[4]==n_bad_nodes for ch in Children]
            if all(flag_list): # if all children are "bad"
                mp = Children[0][5:]
                MacroPixelCoords.append(mp) # store parent node
                Children = []
            else:
                Children = [ch for ch in Children if ch[4] != n_bad_nodes]

        # filter out children fully out of the roi
        Children = [ch for ch in Children if ch[3]==ch[3]]

        # check whether minimum dimension has been reached
        MacroPixelCoords.extend([ch[0:4] for ch in Children if ch[2]==1])

        NodeList.extend([ch for ch in Children if ch[2]!=1])

    return MacroPixelCoords


def build_layers(Input_image, sampling_frequency, evaluation_method, null_distr=None, shapiro_plus_th=None):

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


def keep_father(p_father, p_children, incr_res_condition, p_thr=0.05):

    keep = False
    match incr_res_condition:

        # all children are bad
        case "all":
            if np.nanmin(p_children) > p_thr:
                keep = True

        # at least one child is bad
        case "one":
            if np.nanmax(p_children) > p_thr:
                keep = True

        # mean children p_value is bad
        case "mean":
            if np.nanmean(p_children) > p_thr:
                keep = True

        # most of existing children are bad
        case "majority":
            if len([_ for _ in p_children if _==_ and _>p_thr])/len([_ for _ in p_children if _==_])>=0.5:
                keep = True

        # father is better than all children
        case "father_father_1e-4":
            if p_father < np.nanmin(p_children) and p_father > 1e-4:
                keep = True

        # father is better than at least one of children
        case "father_child_1e-4":
            if p_father < np.nanmin(p_children) and np.nanmin(p_children) > 1e-4:
                keep = True

        # default behaviour
        case _:
            keep = False

    return keep


def NewTopDown(Input_image, mean_signals, p_values, incr_res_condition, p_thr):

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
                p_father = p_values[d][j,i]
                if p_father==p_father:
                    if (optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)]==d).all():
                        children = [(jj,ii) for jj in range(2*j,2*j+2) for ii in range(2*i,2*i+2)]
                        p_children = [p_values[d-1][jj,ii] for (jj,ii) in children]
                        if keep_father(p_father, p_children, incr_res_condition=incr_res_condition, p_thr=p_thr):
                            # keep father macro-pixel
                            optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)] = d
                            MacroPixelCoords.append([pixel_size*i, pixel_size*j, pixel_size, p_father])
                        else:
                            # move to children macro-pixels
                            optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)] = d-1
                            for ch,(jj,ii) in enumerate(children):
                                if d==1 and p_children[ch]==p_children[ch]:
                                    MacroPixelCoords.append([pixel_size//2*ii, pixel_size//2*jj, pixel_size//2, p_children[ch]])

    return MacroPixelCoords


def NewBottomUp(Input_image, mean_signals, p_values, incr_res_condition, p_thr):

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
                p_father = p_values[d][j,i]
                if p_father==p_father:
                    children = [(jj,ii) for jj in range(2*j,2*j+2) for ii in range(2*i,2*i+2)]
                    p_children = [p_values[d-1][jj,ii] for (jj,ii) in children]
                    if (optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)]==d-1).all():
                        if keep_father(p_father, p_children, incr_res_condition=incr_res_condition, p_thr=p_thr):
                            # move to father macro-pixel
                            optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)] = d
                            if d==depth:
                                MacroPixelCoords.append([pixel_size*i, pixel_size*j, pixel_size, p_father])
                        else:
                            # keep children macro-pixels
                            optimal_depth[pixel_size*j:pixel_size*(j+1),pixel_size*i:pixel_size*(i+1)] = d-1
                            for ch,(jj,ii) in enumerate(children):
                                if p_children[ch]==p_children[ch]:
                                    MacroPixelCoords.append([pixel_size//2*ii, pixel_size//2*jj, pixel_size//2, p_children[ch]])
                    else:
                        # some children already stopped at a higher resolution
                        for ch,(jj,ii) in enumerate(children):
                            if (optimal_depth[pixel_size//2*jj:pixel_size//2*(jj+1),pixel_size//2*ii:pixel_size//2*(ii+1)]==d-1).all() and p_children[ch]==p_children[ch]:
                                MacroPixelCoords.append([pixel_size//2*ii, pixel_size//2*jj, pixel_size//2, p_children[ch]])

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


def save_mp_stats(padded_image, mp_annot, output_filename):

    # creation of padded roi mask
    padded_roi_mask = np.array(padded_image[0,:,:])
    padded_roi_mask[~np.isnan(padded_roi_mask)] = 1
    number_native_pixels = np.sum(~np.isnan(padded_roi_mask))
    tessellated_area = np.array(padded_image[0,:,:])*np.nan

    # statistics on macro-pixel sizes
    log2_sizes = [int(np.log2(_)) for _ in mp_annot["pixel_coordinates_L"]]
    sides, counts = np.unique(log2_sizes, return_counts=True)
    print(f"sides = {sides}, counts = {counts}")

    # area of macro-pixels actually in roi
    covered_area = np.empty([len(mp_annot["pixel_coordinates_L"])])*np.nan
    for i in range(len(covered_area)):
        x0 = int(mp_annot["x_coords"][i])
        y0 = int(mp_annot["y_coords"][i])
        L0 = int(mp_annot["pixel_coordinates_L"][i])
        covered_area[i] = np.sum(~np.isnan(padded_roi_mask[y0:y0+L0,x0:x0+L0]))
        tessellated_area[y0:y0+L0,x0:x0+L0] = 1
    tessellated_area[np.isnan(padded_roi_mask)] = 1

    stats_dict = {"number_native_pixels": number_native_pixels, \
                  "number_macropixels": len(mp_annot["channel_id"]), \
                  "side_distr": (sides,counts), \
                  "area_covered_by_hos": int(np.sum(covered_area)), \
                  "nans_from_hos": int(np.sum(np.isnan(tessellated_area)))}
    print(f"stats_dict = {stats_dict}")

    stats_filename = Path(output_filename).parent.joinpath(Path(output_filename).stem + ".npy")
    np.save(stats_filename, stats_dict)

    df = pd.DataFrame({"x_anchor_left": mp_annot["x_coords"], \
                       "y_anchor_top": mp_annot["y_coords"], \
                       "side_length": mp_annot["pixel_coordinates_L"], \
                       "p_value": mp_annot["p_values"], \
                       "covered_area": covered_area, \
                       "x_coord_cm": mp_annot["x_coord_cm"], \
                       "y_coord_cm": mp_annot["y_coord_cm"], \
    })

    df_filename = Path(stats_filename).parent.joinpath(Path(stats_filename).stem + ".csv")
    int_cols = ["x_anchor_left","y_anchor_top","side_length","covered_area"]
    df[int_cols] = df[int_cols].astype(int)
    df.to_csv(df_filename, index=False)


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
    if args.pruning_direction=="old-top-down":
        MacroPixelCoords = OldTopDown(Input_image = padded_image_seq,
                                      sampling_frequency = sampling_frequency,
                                      exit_condition = args.exit_condition,
                                      evaluation_method = args.evaluation_method,
                                      voting_threshold = args.voting_threshold,
                                      n_bad_nodes = args.n_bad_nodes,
                                      null_distr = null_distr,
                                      shapiro_plus_th = shapiro_plus_th)
    else:
        mean_signals, p_values = build_layers(Input_image = padded_image_seq,
                                              sampling_frequency = sampling_frequency,
                                              evaluation_method = args.evaluation_method,
                                              null_distr = null_distr,
                                              shapiro_plus_th = shapiro_plus_th)
        if args.pruning_direction=="top-down":
            MacroPixelCoords = NewTopDown(padded_image_seq, mean_signals, p_values, incr_res_condition=args.pruning_method, p_thr=0.05)
        elif args.pruning_direction=="bottom-up":
            MacroPixelCoords = NewBottomUp(padded_image_seq, mean_signals, p_values, incr_res_condition=args.pruning_method, p_thr=0.05)

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
    p_values = np.empty([N_MacroPixel])

    # for each macro-pixel px (containing x,y,L)
    for px_idx, px in enumerate(MacroPixelCoords):
        signal[:, px_idx] = np.nanmean(padded_image_seq[:, px[1]:px[1]+px[2],
                                                        px[0]:px[0]+px[2]],
                                       axis=(1,2))
        y_coord_cm[px_idx], x_coord_cm[px_idx] = \
            ComputeCenterOfMass(padded_image_seq[:, px[1]:px[1]+px[2], px[0]:px[0]+px[2]],
                                spatial_scale)
        coordinates[px_idx] = px[0:3]
        ch_id[px_idx] = px_idx
        y_coord[px_idx] = (px[1]+0.5*px[2])*spatial_scale
        x_coord[px_idx] = (px[0]+0.5*px[2])*spatial_scale
        p_values[px_idx] = px[3]

    # macropixels details are stored as array_annotations
    mp_annot = {"x_coords": coordinates.T[0],
                "y_coords": coordinates.T[1],
                "x_coord_cm": x_coord_cm,
                "y_coord_cm": y_coord_cm,
                "pixel_coordinates_L": coordinates.T[2],
                "channel_id": ch_id,
                "p_values": p_values}

    new_asig = asig.duplicate_with_new_data(signal)
    new_asig.array_annotations.update(mp_annot) # check!
    new_asig.description += "Non homogeneous downsampling obtained by checking " + \
                            "the signal-to-noise ratio of macropixels at different sizes."
    block.segments[0].analogsignals[0] = new_asig

    save_mp_stats(padded_image_seq, mp_annot, args.output_stats)

    write_neo(args.output, block)
