
import numpy as np
from itertools import groupby
from operator import itemgetter
import quantities as pq

def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''
    setOfElems = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True
        else:
            setOfElems.add(elem)
    return False


# MAX ABST TIMELAG
def timelag_optimization(evts, MAX_ABS_TIMELAG):

    StartingValue = MAX_ABS_TIMELAG
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
        WnW = DeltaTL<=MAX_ABS_TIMELAG;
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
            if duplicates == True and MAX_ABS_TIMELAG > StartingValue*0.0001: # if the wave is not unique
                WaveUnique_flag = True
                MAX_ABS_TIMELAG = MAX_ABS_TIMELAG*0.75; # rescale the parameter
                break
    
    print('maximum abs timelag: ', MAX_ABS_TIMELAG) 
        
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



def iwi_optimization(Wave, ExpectedTrans, min_ch_num, ACCEPTABLE_REJECTION_RATE):

    # compute inter wave interval (IWI)
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

            if n_waves <= ExpectedTrans and  len(np.where(~UniqueWaves)[0])/np.float64(n_waves) > ACCEPTABLE_REJECTION_RATE: # if there is at least 1 non-unique wave
                MAX_IWI = MAX_IWI*0.75 # reduce max iwi
                OneMoreLoop = True
            elif n_waves > ExpectedTrans and len(np.where(WaveSize < min_ch_num)[0])/np.float64(n_waves) > ACCEPTABLE_REJECTION_RATE:
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


