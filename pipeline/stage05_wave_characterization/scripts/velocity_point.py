import neo
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import os
import argparse
import scipy
import pandas as pd
from utils import load_neo, none_or_str, save_plot
import math 


def calc_point_velocities(evts, output_path):

    #nCh = len(np.unique(ChLabel))
    #RecordingSet = list(range(1,nCh))
    DIM_X = 50 #evts.annotations['Dim_x']
    DIM_Y = 50 #evts.annotations['Dim_y']
    spatial_scale = evts.annotations['spatial_scale']
    
    #FullWaveUnique = evts.annotations['FullWaveUnique']
    #FullWaveSize = evts.annotations['FullWaveSize']
    #FullWaveTime = evts.annotations['FullWaveTime']
    #FullWaveUniqueSize = evts.annotations['FullWaveUniqueSize']
    Wave = []
    for w in np.unique(np.int64(np.float64(evts.labels))):
        w_idx = np.where(np.int64(np.float64(evts.labels)) == w)[0]
        Wave.append({'times': evts.times[w_idx], 'ch': evts.array_annotations['channels'][w_idx], 'x_coords': evts.array_annotations['x_coords'][w_idx], 'y_coords': evts.array_annotations['y_coords'][w_idx]})
        
    #===========================WAVE VELOCITY===========================
    #print '         waves velocity computing...'
    
    NumWave = len(Wave)
    spatial_scale = evts.annotations['spatial_scale']
    v_unit = (spatial_scale.units/evts.times.units).dimensionality.string
    wave_ids = np.unique(evts.labels)
    velocities = np.zeros((len(wave_ids), 2))


    X_velocity = np.zeros(NumWave)
    Y_velocity = np.zeros(NumWave)

    mean_vel = []
    std_vel = []
    
    #ciclo su tutte le onde di una data raccolta dati
    point = []
    point_dir_mean = []

    for w in range(0,NumWave):
        print('w', w)

        grid = np.zeros((DIM_Y,DIM_X))
        
        #creo la griglia
        for k in range(0,len(Wave[w]['times'])):
            x = int(Wave[w]['x_coords'][k])
            print('x', x)
            y = int(Wave[w]['y_coords'][k])
            print('y', y)
            #grid[x, y] = np.around(Wave[w]['times'][k], decimals = 6)
            grid[x, y] = Wave[w]['times'][k]
        
        contatore = 0
        velocity = []
        Tx_temp = 0
        Ty_temp = 0
        
        point_vel_x = []
        point_vel_y = []
        angle = []

        for x in range(1,DIM_Y-1):
            for y in range(1,DIM_X-1):


                if ( grid[x+1,y] != 0  and grid[x-1,y] != 0 and grid[x,y+1] != 0  and grid[x,y-1] != 0) and ( grid[x+1,y] != np.nan  and grid[x-1,y] != np.nan and grid[x,y+1] != np.nan  and grid[x,y-1] != np.nan):
                    Tx_temp=(grid[x+1,y] - grid[x-1,y])/(2*evts.annotations['spatial_scale'])
                    Ty_temp=(grid[x,y+1] - grid[x,y-1])/(2*evts.annotations['spatial_scale'])

                    if np.sqrt(Tx_temp**2+Ty_temp**2) != 0:
                        try:
                            vel = 1./np.sqrt(Tx_temp**2+Ty_temp**2)
                            
                            point_vel_x.append(Tx_temp.magnitude)
                            point_vel_y.append(Ty_temp.magnitude)
                            angle.append(np.arctan2(Tx_temp.magnitude,Ty_temp.magnitude))
                            
                            velocity.append(vel)
                            point.append([vel.magnitude, np.arctan2(Tx_temp.magnitude,Ty_temp.magnitude)])
                            
                            #print('vel', vel)
                            contatore +=1
                        except ZeroDivisionError:
                            1;

        if contatore != 0:
            mean_velocity = np.mean(velocity)
            point_dir_mean = np.mean(angle)
            meanVel_angle = np.arctan2(np.mean(point_vel_x),np.mean(point_vel_y))
            
            np.savetxt(output_path + 'VelWave_' + str(w) + '.txt', [point_vel_x, point_vel_y])
            #qui aggiustiamo le unità di misura
            mean_vel.append([mean_velocity, np.std(velocity)/len(velocity), point_dir_mean, meanVel_angle])
        else:
            mean_vel.append([-1, -1, -1000, -1000])
            
    #plt.figure()
    #plt.hist(mean_vel)
    #plt.title('Mean velocity point')
    #plt.xlabel('velocities [' + str(v_unit) + ']')
    
    
    # transform to DataFrame
    df = pd.DataFrame(mean_vel,
                      columns=['velocity_point', 'velocity_point_std', 'point_dir_mean', 'meanVel_angle'],
                      index=wave_ids)
    df['velocity_unit'] = [v_unit]*len(wave_ids)
    df.index.name = 'wave_id'


    df_point = pd.DataFrame(point,
                      columns=['point_vel', 'point_direction'])
    df_point['velocity_unit'] = [v_unit]*len(point)

    return df, df_point
    

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output2", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_path", nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                     help="path of output image file")
    args = CLI.parse_args()
    print('HEYYYY', args.output_path)
    block = load_neo(args.data)

    evts = [ev for ev in block.segments[0].events if ev.name == 'Wavefronts'][0]

    velocities_df, point_df = calc_point_velocities(evts, args.output_path)

    if args.output_img is not None:
        save_plot(args.output_img)

    point_df.to_csv(args.output2)
    velocities_df.to_csv(args.output)
