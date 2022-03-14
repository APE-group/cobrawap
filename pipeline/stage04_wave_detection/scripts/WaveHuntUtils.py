#########################################################################
# WaveHuntUtils.py
# Collection of functions useful for the WaveHunt
#########################################################################

import numpy as np
from itertools import groupby
from operator import itemgetter
import quantities as pq

#------------------------------------------------------------------------
# Misc Functions
#------------------------------------------------------------------------
def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''
# (fucntion used by timelag_optimization and iwi_optimization)

    setOfElems = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True
        else:
            setOfElems.add(elem)
    return False

#------------------------------------------------------------------------
def PlotDetectedWaves(evts, waves):

    fig, ax = plt.subplots(2, 1, sharex = True)
    fig.set_size_inches(6,4, forward=True)
    ax[0].tick_params(axis='both', which='major', labelsize=6)
    ax[1].tick_params(axis='both', which='major', labelsize=6)

    ax[0].plot(evts.times, evts.array_annotations['channels'], '.', markersize = 0.02, color = 'black')
    ax[0].set_xlim([0, np.max(evts.times)])
    ax[0].set_title('detected transitions', fontsize = 7.) 
    ax[0].set_ylabel('channel id', fontsize = 7.) 

    ax[1].scatter(waves.times, waves.array_annotations['channels'], s=0.02, c=np.int32(waves.labels), cmap = 'prism')
    ax[1].set_xlim([0, np.max(evts.times)])
    ax[1].set_title('detected transitions', fontsize = 7.) 
    ax[1].set_ylabel('channel id', fontsize = 7.) 
    ax[1].set_xlabel('time (s)', fontsize = 7.) 
    plt.tight_layout()
    return ax

#------------------------------------------------------------------------
# Parameter Optimization
#------------------------------------------------------------------------
def timelag_optimization(evts, max_abs_timelag):
# (intra-wave interval, i.e. time lag between triggers in the same candidate wave

    StartingValue = max_abs_timelag
    UpTrans = evts.times.magnitude
    ChLabel = evts.array_annotations['channels']
    sorted_idx = np.argsort(UpTrans)
    UpTrans = UpTrans[sorted_idx]
    ChLabel = ChLabel[sorted_idx]
    #tmp = np.unique([[u,c] for u,c in zip(UpTrans, ChLabel)], axis = 0)
    #UpTrans = tmp.T[0]
    #ChLabel = tmp.T[1]

    DeltaTL = np.diff(UpTrans);
    
    ####################################################
    # Compute the time lags matrix looking for optimal MAX_ABS_TIMELAG...
    # (depends on the distance between electrodes in the array)

    WaveUnique_flag = True;
    
    while WaveUnique_flag: #while there still is at least one non-unique wave...
        WaveUnique_flag = False
        # WnW is True if time distance between two consecutive transitions lies 
        # within the MAX_ABS_TIMELAG parameter
        WnW = DeltaTL<= max_abs_timelag;
        # select indexes associated with consecutive true values as transition 
        # associated to the same wave phoenomenon
        ndx_list_true = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(np.where(WnW)[0]), lambda ix:ix[0]-ix[1])]
        ndx_list_false = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(np.where(~WnW)[0]), lambda ix:ix[0]-ix[1])]

        #merge
        if len(ndx_list_false):
            ndx_list = []
            for t, f in zip(ndx_list_true, ndx_list_false):
                ndx_list.append(t+f)
        else:
            ndx_list = ndx_list_true

        for ndx in ndx_list:
            #print(ndx)
            duplicates = checkIfDuplicates(ChLabel[ndx])
            if duplicates == True and max_abs_timelag > StartingValue*0.0001: # if the wave is not unique
                WaveUnique_flag = True
                max_abs_timelag *= 0.75; # rescale the parameter
                break
    
    print('maximum abs timelag: ', max_abs_timelag) 
        
    Wave = [dict() for i in range(len(ndx_list))]
    # fill the wave candidates dictionary
    for w, ndx in enumerate(ndx_list):
        Wave[w] = {'ndx': ndx,
                   'ch': ChLabel[ndx],
                   'time': UpTrans[ndx],
                   'WaveUnique': len(ChLabel[ndx]) == len(np.unique(ChLabel[ndx])),
                   'WaveSize': len(ndx),
                   'WaveTime': np.mean(UpTrans[ndx]),
                   'WaveUniqueSize': len(np.unique(ChLabel[ndx]))}

    return Wave

#------------------------------------------------------------------------
def iwi_optimization(Wave, ExpectedTrans, min_ch_fraction, nCh, acceptable_rejection_rate):
# IWI = inter wave interval, i.g. distinct waves in the wave collection

    min_ch_num = min_ch_fraction*(nCh + np.sqrt(nCh))
    WaveTime=list(map(lambda x : np.mean(x['time']), Wave))
    IWI = np.diff(WaveTime)
    ##Recollect small-waves in full waves (involving a wider area).
    MAX_IWI = np.max(IWI)*0.5

    OneMoreLoop = True;
    
    while OneMoreLoop: #while there still is at least one non-unique wave...
        
    
        # WnW is True if time distance between two consecutive waves lies 
        # within the MAX_IWI parameter
        WnW = IWI<=MAX_IWI;
        # select indexes associated with consecutive true values as transition 
        # associated to the same wave phoenomenon
        wave_ndx_list = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(np.where(WnW)[0]), lambda ix:ix[0]-ix[1])]

        if len(wave_ndx_list) >= ExpectedTrans: OneMoreLoop = False # if more than expected waves are found, 
                                                                    # exit the loop
        else: # otherwise
            # check the number of non-unique waves
            UniqueWaves = [] # wether a merged wave meets the unicity principle
            WaveSize = [] # involved channels

            for w_ndx in wave_ndx_list: # for each collected wave
                temp_ch_labels = []
                for w in w_ndx:
                    temp_ch_labels.extend(Wave[w]['ch'])#merged channels
                
                duplicates = checkIfDuplicates(temp_ch_labels)
                WaveSize.append(len(np.unique(temp_ch_labels)))

                if duplicates == True: # if the wave is not unique
                    UniqueWaves.append(False)
                else:
                    UniqueWaves.append(True)
            UniqueWaves = np.array(UniqueWaves, dtype = bool)

            # if rejaction rate limit is not met
            n_waves = len(UniqueWaves)

            if n_waves <= ExpectedTrans and  len(np.where(~UniqueWaves)[0])/np.float64(n_waves) > acceptable_rejection_rate: # if there is at least 1 non-unique wave
                MAX_IWI = MAX_IWI*0.75 # reduce max iwi
                OneMoreLoop = True
            elif n_waves > ExpectedTrans and len(np.where(WaveSize < min_ch_num)[0])/np.float64(n_waves) > acceptable_rejection_rate:
                MAX_IWI = MAX_IWI * 1.25
                OneMoreLoop = True
    
    # create new dictionary for candidates waves
    MergedWaves = [dict() for i in range(len(wave_ndx_list))]
    # fill the wave candidates dictionary
    for i, w_ndx in enumerate(wave_ndx_list):
        temp_ch_labels = []
        temp_ndx = []
        temp_times = []
        for w in w_ndx:
            temp_ch_labels.extend(Wave[w]['ch'])#merged channels
            temp_ndx.extend(Wave[w]['ndx'])#merged channels
            temp_times.extend(Wave[w]['time'])#merged channels
        MergedWaves[i] = {'ndx': np.array(temp_ndx),
                   'ch': np.array(temp_ch_labels, dtype = np.int32),
                   'times': np.array(temp_times)*pq.s,
                   'WaveUnique': len(temp_ch_labels) == len(np.unique(temp_ch_labels)),
                   'WaveSize': len(temp_ndx),
                   'WaveTime': np.mean(temp_times),
                   'WaveUniqueSize': len(np.unique(temp_ch_labels))}
    
    return(MergedWaves)

#-----------------------------------------------------------------------------------------
# Wave Cleaning
#-----------------------------------------------------------------------------------------
def Neighbourhood_Search(coords, spatial_scale):
    
    #--- LOCALITY criterion
    # Identify the neighbours of each pixel as those pixels whose barycenter lies within a 
    # radius of pixel size from the given pixel.
    # Pixel size is spatial_scale for uniformly spaced grid, or a multiple ('radius') of 
    # native resolution for dataset processed with HOS (Hierarchical Optimal Sampling) 

    neighbors = []
    for x, y, L in zip(coords['x'], coords['y'], coords['radius']): # for each channel
        idx_x = np.where(np.abs(coords['x']-x) <= L*spatial_scale)[0]
        idx_y = np.where(np.abs(coords['y']-y) <= L*spatial_scale)[0]
        idx = np.intersect1d(idx_x, idx_y)
        neighbors.append(list(idx))
    return(neighbors)

# NOTE. Beside border effects, for uniformly spaced grid each pixel has 4 neighbors, while 
# for HOS if a small (high resolution) pixel is sorrounded by larger pixels (i.e. less 
# informative) it could have an empty set of neighbors. This is in agreement with the 
# assumption made that each pixel can be affected only by pixels (neighbors) having its same
# size, i.e. same information density

#-----------------------------------------------------------------------------------------
def ChannelCleaning(Wave, neighbors):
# (used by CleanWave)
    # --- (1) First CHANNEL CLEANING: check the TIME DISTANCE BETWEEN REPETITIONS IN A WAVE

    chs = Wave['ch'];            
    nhc_pixel, nhc_count = np.unique(chs, return_counts=True)
    rep = np.where(nhc_count > 1.)[0]; # channels index where the wave pass more than once           

    k = 0


    
    while k < len(rep):
        
        i=rep[k];
        
        repeted_index = np.where(chs == nhc_pixel[i])[0] # where i have repetitions
        
        timeStamp = Wave['times'][repeted_index].magnitude #UpTrans[idx];
        
        if np.max(np.diff(timeStamp))<0.125: #if repeted transition happen to be close in time                   
            # delta waves freqeunci is 1-4Hz -> minimum distance between two waves = 0.250s
            # open a time-window around each occurrence of i and check
            # how many members of the clan are in the time-window
            
            nClan=[];
            idx_noClan = []
            neigh_coll =  neighbors[nhc_pixel[i]]
            neigh_coll_idx = []
            for n in neigh_coll:
                neigh_coll_idx.extend(np.where(Wave['ch'] == n)[0])

            for idx, time in enumerate(timeStamp): # for each repetintion in the selected channel 
                time_window=[time-0.125, time+0.125];
                clan = len(np.intersect1d(Wave['times'][neigh_coll_idx] > time_window[0],
                                          Wave['times'][neigh_coll_idx] < time_window[1]))
                if clan == 0:
                    idx_noClan.append(idx)

            if len(idx_noClan)> 0:
                del_idx = repeted_index[idx_noClan]
                          
                del_idx = repeted_index[idx] # indexes to be deleted
                Wave['ndx'] = np.delete(Wave['ndx'], del_idx)
                Wave['times'] = np.delete(Wave['times'], del_idx)
                Wave['ch'] = np.delete(Wave['ch'], del_idx)
                Wave['WaveUnique'] = 1
                Wave['WaveTime'] = np.mean(Wave['times'])
                Wave['WaveSize'] = len(Wave['ndx'])

                chs = Wave['ch'];
                nhc_pixel, nhc_count = np.unique(chs, return_counts=True)
                if nhc_count[i] > 1:
                    k = k-1
        k = k+1
    return(Wave)

#------------------------------------------------------------------------
def Clean_SegmentedWave(seg, neighbors):
# (used by CleanWave)

    delSeg=[];
               
    for i in range(0,len(seg)): # CHECK if SEGMENTS have to be cleaned
        
        # 1) clean non-unique segments
        if len(np.unique(seg[i]['ch'])) != len(seg[i]['ch']): #if  the subset of transition is NOT unique
            # --> 'CLEAN', i.e. scan the transition sequence and
            # find repetition of channels
        
            delCh = Clean_NonUniqueWave(seg[i], neighbors)
            
            if len(delCh):
                seg[i]['ch'] = np.delete(seg[i]['ch'], np.unique(delCh))
                seg[i]['ndx']= np.delete(seg[i]['ndx'], np.unique(delCh)); 
                seg[i]['times']= np.delete(seg[i]['times'], np.unique(delCh)); 


        # 2) clean channels non-LocallyConnected (see neighbors)                    
        if len(seg[i]['ch'])<=5: # 5 is min(len(neighbors{:}{2})
            delList=[];
            for ch_idx, ch in enumerate(seg[i]['ch']):
                if not (np.intersect1d(neighbors[ch], np.setdiff1d(seg[i]['ch'],ch))).size:
                    delList.append(ch_idx);
            if len(delList):
                seg[i]['ch']= np.delete( seg[i]['ch'],delList)
                seg[i]['ndx']= np.delete( seg[i]['ndx'],delList)
                seg[i]['times']= np.delete( seg[i]['times'],delList)

        # 3) prepere to remove empty segments
        if len(seg[i]['ndx'])==0:
            delSeg.append(i);


    # remove empty secments
    if delSeg:
       seg=np.delete(seg,delSeg);
    
    return(seg)

#------------------------------------------------------------------------
def Clean_NonUniqueWave(Wave, neighbors):              
# (used by Clean_SegmentedWave and CleanWave)

    # 1) clean non-unique segments
    delCh=[];

    if len(np.unique(Wave['ch'])) != len(Wave['ch']): #if  the subset of transition is NOT unique
        # --> 'CLEAN', i.e. scan the transition sequence and
        # find repetition of channels
        
        delCh=[];

        involved_ch, repetition_count = np.unique(Wave['ch'], return_counts = True)
        involved_ch = involved_ch[np.where(repetition_count > 1)[0]] # channels non unique

        for rep_ch in involved_ch: # for each non unique channel delete repeted channels                                                  
            t0Clan=0;
            nClan=0;
            neigh = neighbors[rep_ch]
            
            for n in neigh:
                try:
                    tClan = Wave['times'][np.where(Wave['ch']==n)[0]].magnitude
                except AttributeError:
                    tClan = Wave['times'][np.where(Wave['ch']==n)[0]]

                if tClan.size > 0:
                    nClan=nClan+1;
                    t0Clan=t0Clan+np.mean(tClan);
                
            if nClan > 0:
                t0Clan= np.float64(t0Clan)/nClan;
                try:
                    tCh = Wave['times'][np.where(Wave['ch']==rep_ch)[0]].magnitude
                except AttributeError:
                    tCh = Wave['times'][np.where(Wave['ch']==rep_ch)[0]]
                
                index = np.where(Wave['ch']==rep_ch)[0][np.argmin(abs(tCh-t0Clan))];                                
                delCh.extend(np.setdiff1d(np.where(Wave['ch']==rep_ch)[0], index));
                
    return(delCh)

#------------------------------------------------------------------------
def CleanWave(UpTrans,ChLabel, neighbors,  FullWave):

    FullWaveUnique=list(map(lambda x : x['WaveUnique'], FullWave))
    FullWaveSize=list(map(lambda x : x['WaveSize'], FullWave))
    FullWaveTime=list(map(lambda x : x['WaveTime'], FullWave))

    nPixel = len(np.unique(ChLabel))
    nw=0;

    
    while nw<len(FullWave): # for each wave
        
        if len(FullWave[nw]['ch']) != len(np.unique(FullWave[nw]['ch'])):
            
            # CLEAN wave channels
            FullWave[nw] = ChannelCleaning(FullWave[nw], neighbors)

            # Look for CANDIDATE SEGMENTS                            
            #Check the time difference between transitions
            delta=np.diff(FullWave[nw]['times']);
            mD=np.mean(delta);
            stdD=np.std(delta);
            THR=mD+3*stdD;

            Segments_Idx = np.where(delta>THR)[0] # where there is a larger sepration between transitions
            
            if Segments_Idx.size: # --- IDENTIFY SEGMENTS

                #create SEGMENTS i.e. candidate NEW waves
                n_candidate_waves=Segments_Idx.size +1; 
                istart=0;  i=0;

                seg = []
                for b in Segments_Idx:
                    seg.append({'ndx': list(range(istart,b+1)),
                                'ch': FullWave[nw]['ch'][istart:b+1],
                                'times': FullWave[nw]['times'][istart:b+1].magnitude});
                    istart=b+1;
                seg.append({'ndx': list(range(istart,len(FullWave[nw]['ch']))),
                            'ch': FullWave[nw]['ch'][istart:len(FullWave[nw]['ch'])],
                            'times': FullWave[nw]['times'][istart:len(FullWave[nw]['times'])].magnitude})

                
                # --- CLEAN SEGMENTS---
                seg = Clean_SegmentedWave(seg, neighbors)

                ##############################################################
                # fino a qui
                n_candidate_waves=len(seg); # update the value in 'segments' = number of segments

                # coalescence of segments if no repetitions with adjacent(s) one(s)
                # N.B. a "small" intersection is admitted
                i=0;
                while i<(n_candidate_waves-1): # for each wave
                    if len(np.intersect1d(seg[i]['ch'],seg[i+1]['ch'])) <= np.floor(1./4.*np.min([len(seg[i]['ch']),len(seg[i+1]['ch'])])):

                        # CANDIDATE SEGMENTS for COALESCENCE
                        # check also if distance between segments'border is smaller than 250ms = 1/4Hz

                        distance = seg[i+1]['times'][0] - seg[i]['times'][len(seg[i]['times'])-1];

                        if distance>=0.250: # check also if distance between segments'border is smaller than 250ms = 1/4Hz
                            # FREQUENCY ALERT: distance compatible with SWA, the two segments should be kept separated
                            i=i+1; #increment the pointer only if no coalescence is made
                        else:
                            # COALESCENCE
                            # The two segments are close enough that can be merged into a single wave
                            # (consider them separated waves would mean the SWA frequency is larger than 4Hz)
                            # COALESCENCE of consecutive SEGMENTS

                            merged = {'ch': np.append(seg[i]['ch'], seg[i+1]['ch']),
                                      'ndx': np.append(seg[i]['ndx'], seg[i+1]['ndx']),
                                      'times': np.append(seg[i]['times'], seg[i+1]['times'])}
                             
                            # CHECK for REPETITIONS (and treat them as usual...
                            # looking at the meanTime in the Clan)
                            
                            delCh = Clean_NonUniqueWave(merged, neighbors)

                            if len(np.unique(delCh))>0:
                                merged['ch'] = np.delete(merged['ch'], np.unique(delCh));
                                merged['ndx']= np.delete(merged['ndx'], np.unique(delCh));
                                merged['times']= np.delete(merged['times'], np.unique(delCh));

                            seg[i]=merged; # COALESCENCE
                            seg=np.delete(seg, i+1); # coalesced segments are at index i, segment at index i+1 is REMOVED
                            n_candidate_waves=n_candidate_waves-1;


                    else: # consecutive segments intersect too much...
                        i=i+1; #increment the pointer only if no coalescence is made

                # $$$$$ N.B. the number of segments has to be updated
                NewFullWave = []
                for i in range(0,n_candidate_waves):
                    #ndx = []
                    #for elem in seg[i]['ndx']:
                    #    ndx.append(FullWave[nw]['ndx'][elem])
                    ndx = FullWave[nw]['ndx'][seg[i]['ndx']]
                    NewFullWave.append({'ndx': ndx, 'times': UpTrans[ndx], 'ch': ChLabel[ndx],
                                        'WaveUnique':1, 'WaveSize': len(ndx), 'WaveTime': np.mean(UpTrans[ndx])});


                    #NewFullWaveUnique.append(1); # update the logical value (...we are "cleaning" the waves)
            else: # NO SEGMENTS identified -->
                # repeated chiannels are due to noise (and not to the presence of more than one wave)
                # CLEAN the wave, i.e. keep only the first channel occurrance
                delCh = Clean_NonUniqueWave(FullWave[nw], neighbors)
                
                if len(delCh):
                    FullWave[nw]['ch'] = np.delete(FullWave[nw]['ch'], np.unique(delCh))
                    FullWave[nw]['ndx']= np.delete(FullWave[nw]['ndx'], np.unique(delCh)); 
                    FullWave[nw]['times']= np.delete(FullWave[nw]['times'], np.unique(delCh)); 


                # wave is "cleaned"; store the updated wave
                ndx = FullWave[nw]['ndx']
                NewFullWave = [{'ndx': ndx, 'times': UpTrans[ndx], 'ch': ChLabel[ndx],
                               'WaveUnique': 1, 'WaveSize': len(ndx), 'WaveTime':np.mean(UpTrans[ndx])}];


            # --- REPLACE CurrentWave with NewWave(s)
            # [its segments or its 'cleaned' version]

            if nw != 0:

                Pre = FullWave[:].copy()
                FullWave=Pre[0:nw].copy()
                FullWave.extend(NewFullWave)
                FullWave.extend(Pre[nw+1:]);#end

            else:               
                Pre = FullWave[:].copy()
                FullWave = NewFullWave.copy()
                FullWave.extend(Pre[nw+1:]);#end
            # --- INCREMENT the pointer
            if len(NewFullWave)>0: # SEGMENTS ARE NEW WAVES
                nw = nw+len(NewFullWave); # increment (point at the next wave)
            else: # no segments identified, the current wave is a New Wave, because it has been cleaned
                nw=nw+1; # increment (point at the next wave)
        else:
            nw=nw+1; # increment (point at the next wave) [current wave is already unique]

    return(FullWave)

#------------------------------------------------------------------------
def RemoveSmallWaves(Evts_UpTrans, min_ch_fraction, FullWave, dim_x, dim_y):
    
    UpTrans = Evts_UpTrans.times
    ChLabel = Evts_UpTrans.array_annotations['channels']
    
    nCh = len(np.unique(ChLabel))
    min_ch_num = min_ch_fraction*(nCh + np.sqrt(nCh))
    spatial_scale = Evts_UpTrans.annotations['spatial_scale']
    FullWaveUnique=list(map(lambda x : x['WaveUnique'], FullWave))
    FullWaveSize=list(map(lambda x : x['WaveSize'], FullWave))
    FullWaveTime=list(map(lambda x : x['WaveTime'], FullWave))


    # Remove small waves and those rejected...
    temp = [FullWaveUnique[i]==1 and FullWaveSize[i] >= min_ch_num  for i in range(0, len(FullWaveUnique))]
    ndx = np.where(np.array(temp))[0];
    RejectedWaves = len(FullWaveUnique) - len(ndx); # rejected beacuse too small
                                                    # (i.e. involving too few channels)
    Wave = []
    for idx in ndx:
        Wave.append(FullWave[idx]);

    return Wave



