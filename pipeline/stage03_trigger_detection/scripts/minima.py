import neo
import numpy as np
import quantities as pq
from scipy.signal import argrelmin, argrelmax
import argparse
from utils import load_neo, write_neo, remove_annotations
import scipy.io as sio

def detect_minima(asig, order, interpolation_points, interpolation):
    signal = asig.as_array()
    sampling_time = asig.times[1] - asig.times[0]

    t_idx, channel_idx = argrelmin(signal, order=order, axis=0)
    t_idx_max, channel_idx_max = argrelmax(signal, order=order, axis=0)

    if interpolation:

        # minimum
        fitted_idx_times = np.zeros([len(t_idx)])
        start_arr = t_idx - int(interpolation_points/2)
        start_arr = np.where(start_arr > 0, start_arr, 0)
        stop_arr = start_arr + int(interpolation_points)
        start_arr = np.where(stop_arr < len(signal), start_arr, len(signal)-interpolation_points-1)
        stop_arr = np.where(stop_arr < len(signal), stop_arr, len(signal)-1)

        signal_arr = np.empty((interpolation_points,len(start_arr)))
        signal_arr.fill(np.nan)

        for i, (start, stop, channel_i) in enumerate(zip(start_arr, stop_arr, channel_idx)):
            signal_arr[:,i] = signal[start:stop, channel_i]

        X_temp = range(0, interpolation_points)
        params = np.polyfit(X_temp, signal_arr, 2)

        min_pos = -params[1,:] / (2*params[0,:]) + start_arr
        min_pos = np.where(min_pos > 0, min_pos, 0)
        minimum_times = min_pos * sampling_time
        
        minimum_value = params[0,:]*( -params[1,:] / (2*params[0,:]) )**2 + params[1,:]*( -params[1,:] / (2*params[0,:]) ) + params[2,:]
        
        # maximum
        fitted_idx_times = np.zeros([len(t_idx_max)])
        start_arr = t_idx_max - int(interpolation_points/2)
        start_arr = np.where(start_arr > 0, start_arr, 0)
        stop_arr = start_arr + int(interpolation_points)
        start_arr = np.where(stop_arr < len(signal), start_arr, len(signal)-interpolation_points-1)
        stop_arr = np.where(stop_arr < len(signal), stop_arr, len(signal)-1)

        signal_arr = np.empty((interpolation_points,len(start_arr)))
        signal_arr.fill(np.nan)

        for i, (start, stop, channel_i) in enumerate(zip(start_arr, stop_arr, channel_idx_max)):
            signal_arr[:,i] = signal[start:stop, channel_i]

        X_temp = range(0, interpolation_points)
        params = np.polyfit(X_temp, signal_arr, 2)

        max_pos = -params[1,:] / (2*params[0,:]) + start_arr
        max_pos = np.where(max_pos > 0, max_pos, 0)
        
        maximum_times = max_pos * sampling_time
        maximum_value = params[0,:]*( -params[1,:] / (2*params[0,:]) )**2 + params[1,:]*( -params[1,:] / (2*params[0,:]) ) + params[2,:]

        
        amplitude = []
        ch_arr = []
        min_arr = []
        
        for i in range(len(min_pos)): # for each transition
            ch = channel_idx[i]
            min_time = min_pos[i]
            #print('min time', min_time)
            min_value = minimum_value[i]
            #print('min value', min_value)
            #print('signal', signal[t_idx[i]][ch])
            
            
            ch_idx = np.where(channel_idx_max == ch)[0]
            max_time = max_pos[ch_idx]
            max_value = maximum_value[ch_idx]
            time_idx = np.where(max_time > min_time)[0]
            times = max_time[time_idx]
            
            max_value = max_value[time_idx]
            #print('MAX VALUES', max_value)

            try:
                idx_min_ampl = np.argmin(times)
                amplitude.append(max_value[idx_min_ampl] - min_value)
                #print('time', times[idx_min_ampl])
                #print('max signal', signal[max_value[idx_min_ampl]][ch])

            except (IndexError, ValueError) as e:
                amplitude.append(max_value - min_value)
                #print('time', times)
                
            ch_arr.append(ch)
        #sio.savemat('/Users/chiaradeluca/Desktop/PhD/Wavescalephant/wavescalephant-master/Output/MF_LENS/stage03_trigger_detection/Amplitude.mat', {'Amplitude': amplitude, 'Ch': ch_arr, 'times': minimum_times})
        arr_dict = {'Amplitude': amplitude, 'Ch': ch_arr, 'times': minimum_times}
        
    else:
        minimum_times = asig.times[t_idx]
        maximum_times = asig.times[t_idx_max]
        amplitude = maximum_times-minimum_times
        ch_arr = channel_idx
        arr_dict = {'Amplitude': amplitude, 'Ch': ch_arr, 'times': minimum_times}


    sort_idx = np.argsort(minimum_times)
    
    evt = neo.Event(times=minimum_times[sort_idx],
                    labels=['UP'] * len(minimum_times),
                    name='Transitions',
                    minima_order=order,
                    use_quadtratic_interpolation=interpolation,
                    num_interpolation_points=interpolation_points,
                    array_annotations={'channels':channel_idx[sort_idx]})

    for key in asig.array_annotations.keys():
        evt_ann = {key : asig.array_annotations[key][channel_idx[sort_idx]]}
        evt.array_annotations.update(evt_ann)

    remove_annotations(asig, del_keys=['nix_name', 'neo_name'])
    evt.annotations.update(asig.annotations)
    return evt, arr_dict


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--ampl_file", nargs='?', type=str, required=True,
                     help="path of output array")
    CLI.add_argument("--order", nargs='?', type=int, default=3,
                     help="number of neighbouring points to compare")
    CLI.add_argument("--num_interpolation_points", nargs='?', type=int, default=5,
                     help="number of neighbouring points to interpolate")
    CLI.add_argument("--use_quadtratic_interpolation", nargs='?', type=bool, default=False,
                     help="wether use interpolation or not")

    args = CLI.parse_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    transition_event, array_dict = detect_minima(asig,
                                     order=args.order,
                                     interpolation_points=args.num_interpolation_points, interpolation=args.use_quadtratic_interpolation)

    sio.savemat(args.ampl_file, array_dict)
    
    block.segments[0].events.append(transition_event)
    write_neo(args.output, block)
