import neo
import numpy as np
import quantities as pq
from scipy.signal import argrelmin, argrelmax
import argparse
from utils import load_neo, write_neo, remove_annotations
import scipy.io as sio

def detect_minima(asig, order, interpolation_points,  DT):
    
    signal = asig.as_array()
    print(signal)
    sampling_time = asig.times[1] - asig.times[0]
    signal_times = np.linspace(asig.t_start.magnitude, asig.t_stop.magnitude, np.int32((asig.t_stop.magnitude-asig.t_start.magnitude)/sampling_time))
    DT = DT * pq.s
    
    t_idx_old, channel_idx_old = argrelmin(signal, order=order, axis=0)
    minima_times_old = t_idx_old*sampling_time

    t_idx, channel_idx = argrelmin(signal, order=order, axis=0)
    del_idx = []

    for i, t in enumerate(t_idx):
        s = signal.T[channel_idx[i]]
        depth_ch = s[t+1:t+5]
        if np.min(depth_ch) < s[t]:
            del_idx.append(i)

    t_idx = np.delete(t_idx, del_idx)
    channel_idx = np.delete(channel_idx, del_idx)

    t_idx_1 = t_idx.copy()
    channel_idx_1 = channel_idx.copy()

    minima_times_1 = t_idx*sampling_time
    channel_idx_1 = channel_idx.copy()
    
    # seconda condizione, guardo i massimi
    #DT = 5.# semi ampiezza

    t_idx_max, channel_idx_max = argrelmax(signal, order=order, axis=0)

    del_idx = []

    for i, t in enumerate(t_idx_max): #per ogni massimo
        s = signal.T[channel_idx_max[i]]
        
        if t*sampling_time < DT:
            amplitude = np.max(s[0:np.int32(2*DT/sampling_time)]) - np.min(s[0:np.int32(2*DT/sampling_time)])
            th = np.min(s[0:np.int32(2*DT/sampling_time)]) + amplitude/2.

        elif t*sampling_time > len(s)*sampling_time - DT:
            amplitude = np.max(s[len(s) - np.int32(2*DT/sampling_time):len(s)]) - np.min(s[len(s)-np.int32(2*DT/sampling_time):len(s)])
            th = np.min(s[len(s) - np.int32(2*DT/sampling_time):len(s)]) + amplitude/2.

        else:
            amplitude = np.max(s[t - np.int32(DT/sampling_time):t + np.int32(DT/sampling_time)]) - np.min(s[t - np.int32(DT/sampling_time):t + np.int32(DT/sampling_time)])
            th = np.min(s[t - np.int32(DT/sampling_time):t + np.int32(DT/sampling_time)]) + amplitude/2.
        
        
        if s[t] < th:
            del_idx.append(i)

        
    t_idx_max = np.delete(t_idx_max, del_idx)
    channel_idx_max = np.delete(channel_idx_max, del_idx)
    
    # mA i massimi devono essere almeno a distanza 160ms = 5step
    DeltaT_max = 0.280 #s


    for channel in range(len(signal[0])): # per ogni canale

        s = signal.T[channel]
        del_idx = []
        
        idx_t_max_ch = np.where(channel_idx_max == channel)[0]
        temp = t_idx_max[idx_t_max_ch]
        
        for i in range(0, len(temp)-1): #per ogni massimo
            #print((temp[i+1] - temp[i])/25.)

            if (temp[i+1] - temp[i]+1)*sampling_time <= DeltaT_max:
                if s[temp[i+1]] < s[temp[i]]:
                    del_idx.append(i+1)
                else:
                    del_idx.append(i)

        t_idx_max = np.delete(t_idx_max, idx_t_max_ch[del_idx])
        channel_idx_max = np.delete(channel_idx_max, idx_t_max_ch[del_idx])



   # terza condizione: per ogni coppia di massimi setacciati come in 2 cerchi l'ultimo dei verdi attuali
    t_idx = []
    channel_idx = []

    for channel in range(len(signal[0])): # per ogni canale
        
        s = signal.T[channel]
        time_s = np.linspace(0, len(s), len(s))
        minima_times_ch = np.int32(t_idx_1[np.where(channel_idx_1 == channel)])

        if ~np.isnan(s[0]): # se non sono su un canale mascherato
            ch_idx_max = np.where(channel_idx_max == channel)[0]
            ch_idx_min = np.where(channel_idx == channel)[0]
            
            if t_idx_max[ch_idx_max][0] != 0:
                t_max = np.concatenate([np.array([0]), np.array(t_idx_max[ch_idx_max])])
            if t_idx_max[ch_idx_max][-1] != len(signal):
                t_max = np.concatenate([np.array(t_idx_max[ch_idx_max]), [len(signal)]])
                    

            for i in range(0, len(t_max)-1): #per ogni massimo
                if t_max[i+1]*sampling_time-t_max[i]*sampling_time < 20.: #se non sono troppo distanti
                    #cerco i minimi nel massimo successivo
                    temp1 = minima_times_ch >= (t_max[i] + 2)
                    temp2 = minima_times_ch < (t_max[i+1] - 2)
                    idx = np.where(temp1 & temp2)[0]
                    if len(idx):
                        t_idx.append(minima_times_ch[idx][-1])
                        channel_idx.append(channel)

                else:
                    a = 1
                    
            #t_idx = np.delete(t_idx, del_idx)
            #channel_idx = np.delete(channel_idx, del_idx)
            

    t_idx_2 = t_idx.copy()
    channel_idx_2 = np.array(channel_idx.copy())
    minima_times_2 = np.array(t_idx_2)*sampling_time
    
    # ora faccio il fit parabolico
    # minimum

    t_idx_pruned = np.array(t_idx_2)
    fitted_idx_times = np.zeros([len(t_idx_pruned)])
    start_arr = t_idx_pruned - 1 #int(interpolation_points/2)
    start_arr = np.where(start_arr > 0, start_arr, 0)
    stop_arr = start_arr + int(interpolation_points)

    start_arr = np.where(stop_arr < len(signal), start_arr, len(signal)-interpolation_points-1)
    stop_arr = np.where(stop_arr < len(signal), stop_arr, len(signal)-1)

    signal_arr = np.empty((interpolation_points, len(start_arr)))
    signal_arr.fill(np.nan)

    print(np.shape(signal_arr[:,0]))
    for i, (start, stop, channel_i) in enumerate(zip(start_arr, stop_arr, channel_idx_2)):
        signal_arr[:,i] = signal[start:stop, channel_i]

    X_temp = range(0, interpolation_points)
    params = np.polyfit(X_temp, signal_arr, 2)

    min_pos = -params[1,:] / (2*params[0,:]) + start_arr
    min_pos = np.where(min_pos > 0, min_pos, 0)
    minimum_times = min_pos * sampling_time
    minimum_value = params[0,:]*( -params[1,:] / (2*params[0,:]) )**2 + params[1,:]*( -params[1,:] / (2*params[0,:]) ) + params[2,:]


    minimum_times[np.where(minimum_times > asig.t_stop)[0]] = asig.t_stop
    
    sort_idx = np.argsort(minimum_times)
    print(sort_idx)
    evt = neo.Event(times=minimum_times[sort_idx],
                    labels=['UP'] * len(minimum_times),
                    name='Transitions',
                    minima_order=order,
                    num_interpolation_points=interpolation_points,
                    array_annotations={'channels':channel_idx_2[sort_idx]})

    for key in asig.array_annotations.keys():
        evt_ann = {key : asig.array_annotations[key][channel_idx_2[sort_idx]]}
        evt.array_annotations.update(evt_ann)

    remove_annotations(asig, del_keys=['nix_name', 'neo_name'])
    evt.annotations.update(asig.annotations)
    return evt


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--order", nargs='?', type=int, default=3,
                     help="number of neighbouring points to compare")
    CLI.add_argument("--num_interpolation_points", nargs='?', type=int, default=5,
                     help="number of neighbouring points to interpolate")
    CLI.add_argument("--dt", nargs='?', type=bool, default=False,
                     help="Downstate lenght")

    print('heeey')
    args = CLI.parse_args()
    block = load_neo(args.data)
    asig = block.segments[0].analogsignals[0]

    transition_event = detect_minima(asig,
                                     order=args.order,
                                     interpolation_points=args.num_interpolation_points, DT=args.dt)

    block.segments[0].events.append(transition_event)
    write_neo(args.output, block)
