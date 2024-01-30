#########################################################################
# WaveHuntUtils.py
# Collection of functions useful for the WaveHunt
#########################################################################

import numpy as np
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
import quantities as pq
import matplotlib.pyplot as plt

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

    ax[0].plot(evts.times, evts.array_annotations['channels'], '.', markersize = 0.9, color = 'black')
    ax[0].set_xlim([0,13])# np.max(evts.times)])
    ax[0].set_title('detected transitions', fontsize = 7.) 
    ax[0].set_ylabel('channel id', fontsize = 7.) 

    ax[1].scatter(waves.times, waves.array_annotations['channels'], s=0.9, c=np.int32(waves.labels), cmap = 'prism')
    #ax[1].set_xlim([0, np.max(evts.times)])
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

    print('\n(1) TimeLagOptimization')
    
    StartingValue = max_abs_timelag
    UpTrans = evts.times.magnitude
    ChLabel = evts.array_annotations['channels']
    sorted_idx = np.argsort(UpTrans)
    UpTrans = UpTrans[sorted_idx]
    ChLabel = ChLabel[sorted_idx]
    print('transitions:',UpTrans)
    print('Channels:',ChLabel)
# N.B. UpTrans should be already sorted (chronological order)
# a check if sorting is necessary could be introduced, if (UpTrans!=sorted(UpTrans)).any():

    DeltaTL = np.diff(UpTrans);
    print('DeltaTL:',DeltaTL)
    ######################################################################################
    # Compute the time lag matrix looking for optimal MAX_ABS_TIMELAG...
    # (depends on the distance between electrodes in the array, i.d. spatial resolution)

    # UniqueWave = each channel appears only once (unique) in the recostructed wavefront 
    WaveUnique_flag = True; #True if at least one wave is not unique
    count=0
# Starting assumption: MAX_ABS_TIMELAG starting value is large --> starting wave(s) are not unique
    
    while WaveUnique_flag: #while there is still at least one non-unique wave...
        WaveUnique_flag = False #Let's assume that all the waves are unique...
        
        # WnW is True if time distance between two consecutive transitions lies within MAX_ABS_TIMELAG
        WnW = DeltaTL<= max_abs_timelag;
                
# --> consecutive true values are assumed to be transitions of the same wave

# --- Group transition indexes associated with consecutive WnW true values as transitions associated to the same wave phenomenon
        ndx_list_true = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(np.where(WnW)[0]), lambda ix:ix[0]-ix[1])]
        ndx_list_false = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(np.where(~WnW)[0]), lambda ix:ix[0]-ix[1])]
        
        print('ndx_list_true',ndx_list_true)
        print('ndx_list_false',ndx_list_false)
#--- Merge
        if len(ndx_list_false):
            ndx_list = []
            for t, f in zip(ndx_list_true, ndx_list_false):
                ndx_list.append(t+f)
        else:
            ndx_list = ndx_list_true
        
        #print(count)
        #print('len(ndx_list)', len(ndx_list))
        print('ndx_list',ndx_list)
        for ndx in ndx_list:
            #print(ndx)
            duplicates = checkIfDuplicates(ChLabel[ndx]) #bool (True/False)
            if duplicates == True and max_abs_timelag > StartingValue*0.0001: # if True, the wave is not unique
# N.B. even if duplicates, if max_abs_timelag goes below the threshold, iteration (while loop) is stopped 
# (a strict "no duplicates" condition would bring to infinite loops and meaningless wave collections
# if there are zeros in DeltaTL, i.e. simultaneous triggers) 
                WaveUnique_flag = True                
                count+=1
                max_abs_timelag *= 0.75; # rescale the parameter
                break
    
    print('number of iterations: ', count)
    print('maximum abs timelag: ', max_abs_timelag) 
    print('(check: ', StartingValue*pow(0.75,count),') (threshold: ', StartingValue*0.0001,')')        
    print('>>> number of waves in WaveCollection_v1: ', len(ndx_list))
    if duplicates:
      print('    N.B. There are still non-unique waves')
    nUniqueWaves=0 #counter initialised

# --- Create dictionary for candidate waves
    Wave = [dict() for i in range(len(ndx_list))]
# --- Fill the wave candidates dictionary ('Wave' is a list of dictionaries)
    for w, ndx in enumerate(ndx_list):
        Wave[w] = {'ndx': np.array(ndx),
                   'ch': np.array(ChLabel[ndx],dtype = np.int32),
                   'times': np.array(UpTrans[ndx])*pq.s,
                   'WaveUnique': len(ChLabel[ndx]) == len(np.unique(ChLabel[ndx])),
                   'WaveSize': len(ndx),
                   'WaveTime': np.mean(UpTrans[ndx]),
                   'WaveUniqueSize': len(np.unique(ChLabel[ndx]))}

        if Wave[w]['WaveUnique']: nUniqueWaves+=1

    print('    number of UniqueWaves: ', nUniqueWaves, '(ratio: ', nUniqueWaves/len(ndx_list),')') 

    return Wave

#------------------------------------------------------------------------
def iwi_optimization(Wave, ExpectedTrans, min_ch_fraction, nCh, acceptable_rejection_rate):
# IWI = inter wave interval, i.e. distinct waves in the wave collection

    print('\n(2) IWIOptimization')
    
    min_ch_num = int(min_ch_fraction*(nCh + np.sqrt(nCh))) # (Globality parameter)
    WaveTime=list(map(lambda x : x['WaveTime'], Wave))
    IWI = np.diff(WaveTime)
    print(len(IWI))
# --- Merge small waves in larger waves (involving a wider area)
    
    MAX_IWI = np.max(IWI)*0.5 #StartingValue

    OneMoreLoop = True;
    count=1
   
    while OneMoreLoop:

        # WnW is True if time distance between two consecutive waves lies within the MAX_IWI parameter
        WnW = IWI<=MAX_IWI;
        MAX_IWI_loop = MAX_IWI
        # --> consecutive true values are assumed to be portions of the same wave

# --- Group wave indexes associated with consecutive WnW true values as sub-waves associated to the same wave phenomenon
        wave_ndx_list = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(np.where(WnW)[0]), lambda ix:ix[0]-ix[1])]

        print('iteration: ', count, '\tMAX_IWI: ', MAX_IWI, '\tnWaves: ', len(wave_ndx_list))
        
        if len(wave_ndx_list) >= ExpectedTrans: OneMoreLoop = False # if more than expected waves are found, 
                                                                    # exit the loop
        else: # otherwise (we merged too much!)
            
            UniqueWaves = [] # whether a merged wave meets the unicity principle
            WaveSize = [] # number of involved channels ('unique', e.g. without repetitions)

            for w_ndx in wave_ndx_list: # for each collected wave
                temp_ch_labels = []
                for w in w_ndx: # for each wave(portion) in the examined collected wave
                    temp_ch_labels.extend(Wave[w]['ch']) # merge channels
                
                duplicates = checkIfDuplicates(temp_ch_labels)
                WaveSize.append(len(np.unique(temp_ch_labels)))

                if duplicates == True: # if the wave is not unique
                    UniqueWaves.append(False)
                else:
                    UniqueWaves.append(True)

            UniqueWaves = np.array(UniqueWaves, dtype = bool)
            n_waves = len(UniqueWaves) #len(UniqueWaves) is the same as len(wave_ndx_list)

# --- Check the UNICITY (number of non-unique waves) and the GLOBALITY (number of unique channels involved)
            if n_waves <= ExpectedTrans and  len(np.where(~UniqueWaves)[0])/np.float64(n_waves) > acceptable_rejection_rate: 
            # waves in the collection are too few (we merged too much) and 
            # the percentage of non-unique waves is above the acceptable rejection rate (default: 10%)
                MAX_IWI = MAX_IWI*0.75 # >>> reduce MAX_IWI, i.e. some merged waves should be splitted, in order to meet the unicity requirement
                OneMoreLoop = True
            elif n_waves > ExpectedTrans and len(np.where(WaveSize < min_ch_num)[0])/np.float64(n_waves) > acceptable_rejection_rate:
            # waves in the collection are too many (we splitted too much) and 
            # the percentage of waves not fulfilling the globality requirement is above the acceptable rejection rate (default: 10%)
                MAX_IWI = MAX_IWI * 1.25 # >>> increase MAX_IWI, i.e. some splitted waves should bee merged, in order to meet the globality requirement
                OneMoreLoop = True
            count+=1
        if MAX_IWI_loop == MAX_IWI:
                OneMoreLoop = False
    print('number of iterations: ', count)
    print('optimal MAX_IWI: ', MAX_IWI)
    print('>>> number of waves in WaveCollection_v2: ', len(wave_ndx_list))
    nUniqueWaves=0 #counter initialised
 
# --- Create dictionary for candidate waves
    MergedWaves = [dict() for i in range(len(wave_ndx_list))]
# --- Fill the wave candidates dictionary
    for i, w_ndx in enumerate(wave_ndx_list):
        temp_ch_labels = []
        temp_ndx = []
        temp_times = []
        for w in w_ndx:
            temp_ch_labels.extend(Wave[w]['ch'])#merged channels
            temp_ndx.extend(Wave[w]['ndx'])#merged transition ids
            temp_times.extend(Wave[w]['times'])#merged transition times
        MergedWaves[i] = {'ndx': np.array(temp_ndx),
                   'ch': np.array(temp_ch_labels, dtype = np.int32),
                   'times': np.array(temp_times)*pq.s,
                   'WaveUnique': len(temp_ch_labels) == len(np.unique(temp_ch_labels)),
                   'WaveSize': len(temp_ndx),
                   'WaveTime': np.mean(temp_times),
                   'WaveUniqueSize': len(np.unique(temp_ch_labels))}
        if MergedWaves[i]['WaveUnique']: nUniqueWaves+=1    
    print('    number of UniqueWaves: ', nUniqueWaves, '(ratio: ', nUniqueWaves/len(wave_ndx_list),')')

    return(MergedWaves)

#-----------------------------------------------------------------------------------------
# Wave Cleaning
#-----------------------------------------------------------------------------------------
def Neighbourhood_Search(coords, spatial_scale, tolerance = 0.01):
    
    #--- LOCALITY criterion
    # Identify the neighbours of each pixel as those pixels whose barycenter lies within a
    # radius of pixel size from the given pixel.
    # Pixel size is spatial_scale for uniformly spaced grid, or a multiple ('radius') of
    # native resolution for dataset processed with HOS (Hierarchical Optimal Sampling)
    
    # tolerance is set aat 1% to avoid rounding problems for small differences
    neighbors = []
    for idx_ch, (x, y, L) in enumerate(zip(coords['x'], coords['y'], coords['radius'])): # for each channel
        dist = ((coords['x']-x)**2 + (coords['y']-y)**2)
        idx = np.where(dist  <= (L*(spatial_scale + tolerance*spatial_scale))**2)
        # delete itself
        idx = np.delete(idx, np.where(idx == idx_ch)[0])
        neighbors.append(list(idx))
    return(neighbors)

# Beside border effects, for uniformly spaced grid each pixel has up to 4 neighbors
# (including itself), while for HOS if a small (high resolution) pixel is sorrounded by 
# larger pixels (i.e. less informative) it could have an empty set of neighbors 
# (len(neighbors[i] is 1, i.e. itself only). This is in agreement with the assumption made 
# that each pixel can be affected only by pixels (neighbors) having its same size, i.e. 
# same information density.
# N.B. proximity in the grid is evaluated along x and y are separately (not as a radius centered 
# at the pixel c.m., i.e. through a distance computed with a quadratic sum ad a square root  

#-----------------------------------------------------------------------------------------
def ChannelCleaning(Wave, neighbors):
# (used by CleanWave)
    # --- (1) First CHANNEL CLEANING: check the TIME DISTANCE BETWEEN REPETITIONS IN A WAVE

    chs = Wave['ch'];            
    nhc_pixel, nhc_count = np.unique(chs, return_counts=True)
    rep = np.where(nhc_count > 1)[0]; # repeated channels, i.e. where the wave passes more than once           

    k = 0
    while k < len(rep):
        i=rep[k];
        repeted_index = np.where(chs == nhc_pixel[i])[0] # where repetitions of channel nhc_pixel[i] occurs in Wave        
        timeStamp = Wave['times'][repeted_index].magnitude #UpTrans[idx];
        
        if np.max(np.diff(timeStamp))<0.125: #if repeted transitions happen to be close in time                   
            # delta wave frequency is 1-4Hz -> minimum distance between two waves = 0.250s
# --- Open a time-window around each occurrence of i and check how many members of the 'clan' are in the time-window
            
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
def CleanWave(UpTrans, ChLabel, neighbors, FullWave):

    FullWaveUnique=list(map(lambda x : x['WaveUnique'], FullWave))
    FullWaveSize=list(map(lambda x : x['WaveSize'], FullWave))
    FullWaveTime=list(map(lambda x : x['WaveTime'], FullWave))

    nPixel = len(np.unique(ChLabel)) # total number of active channels (nCh)
    nw=0;

    while nw<len(FullWave): # for each wave in the WaveCollection
        
        if len(FullWave[nw]['ch']) != len(np.unique(FullWave[nw]['ch'])): # wave is not unique
            
# --- CLEAN wave channels (using neighbors information)
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

############################## REFINEMENT FUNC - TO BE REVISED #############################

def DetectStationary(times, sampling_rate, transition_th):
    #TODO: optimize this function!!!
    minimum = np.min(times)
    maximum = np.max(times)
    n_bins = np.int32((maximum-minimum)*sampling_rate)

    H, edges = np.histogram(times, range = (minimum, maximum),
                            bins = n_bins)
    CumulativeH = np.cumsum(H)
    DiffCum = np.diff(CumulativeH)
    DiffCum = np.insert(DiffCum, [0, DiffCum.size], [0,0])

    Stationary = DiffCum <= transition_th
    # select indexes associated with consecutive true values as transition 
    # associated to the same wave phoenomenon
    ndx_list_true_tmp = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(np.where(Stationary)[0]), 
                                                                         lambda ix:ix[0]-ix[1])]

    # delete single events
    lenght = [len(n) for n in ndx_list_true_tmp]
    ndx_list_true = [ndx_list_true_tmp[i] for i in np.where(np.array(lenght) > 1)[0]]

    start_time_list = []
    end_time_list = []
    for n in ndx_list_true:
        start_time_list.append(edges[n[-1]])
        end_time_list.append(edges[n[0]])

    if len(start_time_list) == 0:
        start_time_list = [edges[0]]
        end_time_list = [edges[-1]]
    if end_time_list[0] < start_time_list[0]:
        temp = [0]
        start_time_list.insert(0, 0)
    if end_time_list[-1] < start_time_list[-1]:
        end_time_list.extend([edges[-1]])

    return(DiffCum, ndx_list_true, start_time_list, end_time_list, edges)

def CheckWave(Waves, sampling_rate, transition_th):
    #TODO: optimize this function!!!
    if np.int32((np.max(Waves['times'])-np.min(Waves['times']))*sampling_rate) > 1:

        # compute cumulative of transition and identify stationary intervals
        DiffCum, ndx_list_true, start_time_list, end_time_list, edges = DetectStationary(Waves['times'].magnitude, sampling_rate, transition_th)
        # crop wave between start and stop
        new_wave = []
        #ax.plot(edges[:-1], DiffCum)
        #print('start', start_time_list)
        #print('end', end_time_list)
        for s, e in zip(start_time_list, end_time_list):
            temp1 = Waves['times'] <= e
            temp2 = Waves['times'] >= s
            idx = np.where(temp1 & temp2)[0]
            if len(idx) != 0:
                new_wave.append({'ndx': Waves['ndx'][idx],
                                 'ch': Waves['ch'][idx],
                                 'times': Waves['times'][idx],
                                 'WaveUnique': len(Waves['ch'][idx]) == len(np.unique(Waves['ch'][idx])),
                                 'WaveSize': len(idx),
                                 'WaveTime': np.mean(Waves['times'][idx]),
                                 'WaveUniqueSize': len(np.unique(Waves['ch'][idx]))})
    else:
        new_wave = [Waves]
    if len(new_wave) ==0:
        new_wave = [Waves]

    return(new_wave)

def CropDetectedWaves(Wave_coll, sampling_rate, transition_th):
    #TODO: optimize this function!!!
    NWaves = len(Wave_coll)  # whether each wave has been checked
    del_idx = []
    for w_idx in range(0, NWaves): # while there still is a wave to be checked
        new_wave = CheckWave(Wave_coll[w_idx], sampling_rate, transition_th)
        if len(new_wave) == 1:
            Wave_coll[w_idx] = new_wave[0]
        elif len(new_wave) > 1:
            Wave_coll[w_idx] = new_wave[0]
            for w in new_wave[1:]:
                Wave_coll.append(w)
        elif len(new_wave) == 0:
            del_idx.append(w_idx)
    return(Wave_coll)



def reject_outliers(data, m=2):
        return m * np.std(data)

def GetRidOfSlowTransition(Waves, neigh, spatial_scale):
    #TODO: optimize this function!!!
    #TODO: allign this function to non homogeneous sampling
    # compute local velocity of each trnasition
    for w_idx, w in enumerate(Waves):
        # for each transition
        w_vel = []
        vel_ch = []
        for t, ch in zip(w['times'], w['ch']):
            ch_neigh = neigh[ch]

            # get transition time from neigh
            times = []
            for n in ch_neigh:
                idx = np.where(w['ch'] == n)[0]
                if len(idx):
                    times.append(w['times'][idx][0]-t)
            vel_r = times/spatial_scale.magnitude
            vel_ch.append(vel_r)
            w_vel.extend(vel_r)

        # compute wave quartile at a set threshold
        th = reject_outliers(w_vel)
        median = np.median(w_vel)

        # compute the number of outliers per transition
        count_good = []
        for v in vel_ch:
            count_good.append(np.sum(np.abs(v-median)<=th)/len(v))
        #select only transition with more than the half non-outliers neigh
        good_idx = np.where(np.array(count_good) >= 0.5)[0]

        
        w['ndx'] = w['ndx'][good_idx]
        w['ch'] = w['ch'][good_idx]
        w['times'] = w['times'][good_idx]
        w['WaveUnique'] = len(w['ch']) == len(np.unique(w['ch']))
        w['WaveSize'] = len(good_idx)
        w['WaveTime'] = np.mean(w['times'])
        w['WaveUniqueSize'] = len(np.unique(w['ch']))

    return(Waves)


