import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import argparse
import scipy
import pandas as pd
import seaborn as sns
import warnings
from utils.io import load_neo, save_plot
from utils.parse import none_or_str


def calc_displacement(times, locations):
    slope, offset, _, _, stderr = scipy.stats.linregress(times, locations)
    d0, d1 = offset + slope*times[0], offset + slope*times[-1]
    displacement = d1 - d0
    displacement_err = np.sqrt(stderr**2 + (stderr*(times[-1]-times[0]))**2)
    return displacement, displacement_err

def trigger_interpolation(evts):
    spatial_scale = evts.annotations['spatial_scale']
    # get center of mass coordinates for each signal
    try:
        coords = {'x': evts.array_annotations['x_coord_cm'],
                  'y': evts.array_annotations['y_coord_cm'],
                  'radius': evts.array_annotations['pixel_coordinates_L']}
    except KeyError:
        spatial_scale = evts.annotations['spatial_scale']
        coords = {'x': (evts.array_annotations['x_coords']+0.5)*spatial_scale,
                  'y': (evts.array_annotations['y_coords']+0.5)*spatial_scale,
                  'radius': np.ones([len(evts.array_annotations['x_coords'])])}

    wave_ids = np.unique(evts.labels)
    
    directions = np.zeros((len(wave_ids), 2), dtype=np.complex_)

    # loop over waves
    for i, wave_i in enumerate(wave_ids):
        print('wave id', wave_i)
        if not np.int32(np.float64(wave_i)) == -1:
            # Fit wave displacement
            idx = np.where(evts.labels == wave_i)[0]
            dx, dx_err = calc_displacement(evts.times[idx].magnitude, coords['x'][idx])
            dy, dy_err = calc_displacement(evts.times[idx].magnitude, coords['y'][idx])
            directions[i] = np.array([dx + 1j*dy, dx_err + 1j*dy_err])

    # transfrom to DataFrame
    df = pd.DataFrame(directions,
                      columns=['direction', 'direction_std'],
                      index=wave_ids)
    df.index.name = 'wave_id'
    return df

def times2ids(time_array, times_selection):
    return np.array([np.argmax(time_array>=t) for t in times_selection])

def calc_flow_direction(evts, asig):
    wave_ids = np.unique(evts.labels)
    directions = np.zeros((len(wave_ids), 2), dtype=np.complex_)
    signals = asig.as_array()
    # loop over waves
    for i, wave_i in enumerate(wave_ids):
        if not int(wave_i) == -1:
            idx = np.where(evts.labels == wave_i)[0]
            t_idx = times2ids(asig.times, evts.times[idx])
            channels = evts.array_annotations['channels'][idx]
            # ToDo: Normalize vectors?
            x_avg = np.nanmean(np.real(signals[t_idx, channels]))
            x_std = np.nanstd(np.real(signals[t_idx, channels]))
            y_avg = np.nanmean(np.imag(signals[t_idx, channels]))
            y_std = np.nanstd(np.imag(signals[t_idx, channels]))
            directions[i] = np.array([x_avg + 1j*y_avg, x_std + 1j*y_std])

    df = pd.DataFrame(directions,
                      columns=['direction', 'direction_std'],
                      index=wave_ids)
    df.index.name = 'wave_id'
    return df

def plot_directions(dataframe, orientation_top=None, orientation_right=None):
    wave_ids = dataframe.index
    directions = dataframe.direction_x + 1j*dataframe.direction_y
    directions_std = dataframe.direction_x_std + 1j*dataframe.direction_y_std

    ncols = int(np.round(np.sqrt(len(wave_ids)+1)))
    nrows = int(np.ceil((len(wave_ids)+1)/ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(3*nrows, 3*ncols))

    rmax = np.max(np.abs(directions))
    for i, (d, d_std) in enumerate(zip(directions, directions_std)):
        row = int(i/ncols)
        if ncols == 1:
            cax = ax[row]
        else:
            col = i % ncols
            cax = ax[row][col]

        cax.plot([0,np.real(d)], [0,np.imag(d)], color='r', alpha=0.8)
        ellipsis = patches.Ellipse(xy=(np.real(d), np.imag(d)),
                                   width=2*np.real(d_std), height=2*np.imag(d_std),
                                   alpha=0.5)
        cax.add_artist(ellipsis)
        cax.set_title('wave {}'.format(wave_ids[i]))
        if np.isfinite(rmax):
            cax.set_ylim((-rmax,rmax))
            cax.set_xlim((-rmax,rmax))
        cax.axhline(0, c='k')
        cax.axvline(0, c='k')
        # cax.axes.get_xaxis().set_visible(False)
        # cax.axes.get_yaxis().set_visible(False)
        cax.set_xticks([])
        cax.set_yticks([])

    if ncols == 1:
        cax = ax[-1]
    else:
        cax = ax[-1][-1]
        for i in range(len(directions), nrows*ncols):
            row = int(i/ncols)
            col = i % ncols
            ax[row][col].set_axis_off()

    cax.axhline(0, c='k')
    cax.axvline(0, c='k')
    cax.set_xlim((-2,2))
    cax.set_ylim((-2,2))
    if orientation_top is not None:
        cax.text(0, 1,orientation_top, rotation='vertical',
                        verticalalignment='center', horizontalalignment='right')
    if orientation_right is not None:
        cax.text(1, 0, orientation_right,
                        verticalalignment='top', horizontalalignment='center')

    sns.despine(left=True, bottom=True)
    return ax

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    CLI.add_argument("--method", "--DIRECTION_METHOD", nargs='?', type=str, default='trigger_interpolation',
                     help="'tigger_interpolation' or 'optical_flow'")
    CLI.add_argument("--event_name", "--EVENT_NAME", nargs='?', type=str, default='wavefronts',
                     help="name of neo.Event to analyze (must contain waves)")
    args, unknown = CLI.parse_known_args()

    block = load_neo(args.data)

    if args.method == 'optical_flow':
        if args.event_name == 'wavemodes':
            warnings.warn('The planar direction of wavemodes can not be '
                          'calculated with the optical_flow method. '
                          'Using trigger_interpolation instead.')
            args.method = 'trigger_interpolation'
        elif not len(block.filter(name='optical_flow', objects="AnalogSignal")):
            warnings.warn('No optical_flow signal could be found for the '
                          'calculation of planar directions. '
                          'Using trigger_interpolation instead.')
            args.method = 'trigger_interpolation'

    evts = block.filter(name=args.event_name, objects="Event")[0]
    evts = evts[evts.labels.astype('str') != '-1']

    if args.method == 'trigger_interpolation':
        directions_df = trigger_interpolation(evts)
        directions = directions_df.to_numpy()
        #directions = directions_df['direction']#trigger_interpolation(evts)
    elif args.method == 'optical_flow':
        asig = block.filter(name='optical_flow', objects="AnalogSignal")[0]
        directions_df = calc_flow_direction(evts, asig)
        directions = directions_df.to_numpy()
    else:
        raise NameError(f'Method name {args.method} is not recognized!')
    df = pd.DataFrame(np.unique(evts.labels), columns=[f'{args.event_name}_id'])
    df['direction_x'] = np.real(directions[:,0])
    df['direction_y'] = np.imag(directions[:,0])
    df['direction_x_std'] = np.real(directions[:,1])
    df['direction_y_std'] = np.imag(directions[:,1])

    if args.output_img is not None:
        orientation_top = evts.annotations['orientation_top']
        orientation_right = evts.annotations['orientation_right']
        plot_directions(df, orientation_top, orientation_right)
        save_plot(args.output_img)
        plt.figure()
        directions = directions_df.direction

    df.to_csv(args.output)
