
# MAX ABST TIMELAG
def Optimal_MAX_ABS_TIMELAG(evts, MAX_ABS_TIMELAG):

    import numpy as np
    import quantities as pq
    #from utils import load_neo, write_neo, remove_annotations
    import neo
    StartingValue = MAX_ABS_TIMELAG
    UpTrans = evts.times.magnitude
    ChLabel = evts.array_annotations['channels']
    DeltaTL = np.diff(UpTrans);
    ####################################################
    # Compute the time lags matrix looking for optimal MAX_ABS_TIMELAG...
    # (depends on the distance between electrodes in the array)

    WaveUnique_flag = 0;
    
    while WaveUnique_flag == 0: #while there still is at least one non-unique wave...
        #print('hum')
    
        WaveUnique_flag = 1
        WnW = np.int32(DeltaTL<=MAX_ABS_TIMELAG);
        #print(WnW)
        #ndxBegin_list = np.append(0, np.where(np.diff(WnW)==1)[0]+1)
        #print('begin', ndxBegin_list)
        #ndxEnd_list = np.append(np.where(np.diff(WnW)==-1)[0]+1, len(UpTrans)-1)
        #print('end', ndxEnd_list)
        print('heeeey', len(np.where(np.diff(WnW)==1)[0]))
        if len(np.where(np.diff(WnW)==1)[0]):
            if np.where(np.diff(WnW)==1)[0][0] != 0:
                ndxBegin_list = np.append(0, np.where(np.diff(WnW)==1)[0]+1)
            else:
                ndxBegin_list = np.where(np.diff(WnW)==1)[0] + 1

            #ndxEnd_list = np.append(np.where(np.diff(WnW)==-1)[0]+1, len(UpTrans)-1)
            ndxEnd_list = ndxBegin_list[1:len(ndxBegin_list)]
            ndxEnd_list = np.append(ndxEnd_list, len(UpTrans))

            #if len(ndxBegin_list) != len(ndxEnd_list):
            #    ndxBegin_list = np.append(ndxBegin_list, len(UpTrans)-1)

        else:
            ndxBegin_list = [0]
            ndxEnd_list = [len(UpTrans)]
        
        #print('begin', ndxBegin_list)
        #print('end', ndxEnd_list)
 
        Wave = [dict() for i in range(len(ndxBegin_list))]
        del_idx = []

        for w in range(0, len(ndxBegin_list)):
            try:
                ndx = list(range(ndxBegin_list[w], ndxEnd_list[w]))
            except IndexError:
                ndx = list(range(ndxBegin_list[w], len(DeltaTL)))
            
            if ndx:
                Wave[w] = {'ndx': ndx,
                           'ch': ChLabel[ndx],
                           'time': UpTrans[ndx],
                           'WaveUnique': len(ndx) == len(np.unique(ChLabel[ndx])),
                           'WaveSize': len(ndx),
                           'WaveTime': np.mean(UpTrans[ndx])};

                if len(ndx) != len(np.unique(ChLabel[ndx])) and MAX_ABS_TIMELAG > StartingValue*0.0001:
                    WaveUnique_flag = 0
                    break
            else:
                del_idx.append(w)
                
        for elem in del_idx:
            Wave.pop(elem)
        
        MAX_ABS_TIMELAG = MAX_ABS_TIMELAG*0.75; # rescale the parameter
        #plt.figure()
        #for elem in range(0, 2): #len(Waves_Inter)):
        #    plt.plot(Wave[elem]['time'], Wave[elem]['ch'], '.', markersize = 5.)
    WaveUnique=list(map(lambda x : x['WaveUnique'], Wave))

    return Wave

def Optima_MAX_IWI(UpTrans, ChLabel, Wave, ACCEPTABLE_REJECTION_RATE):

    import numpy as np
    import quantities as pq
    #from utils import load_neo, write_neo, remove_annotations
    import neo

    WaveUnique=list(map(lambda x : x['WaveUnique'], Wave))
    WaveSize=list(map(lambda x : x['WaveSize'], Wave))
    WaveTime=list(map(lambda x : x['WaveTime'], Wave))
    
    nCh = len(np.unique(ChLabel))
    
    
    ####################################################
    # ExpectedTrans i.e. estimate of the Number of Waves
    # ExpectedTrans used to estimate/optimize IWI
    TransPerCh_Idx, TransPerCh_Num = np.unique(ChLabel, return_counts=True)
    ExpectedTrans = np.median(TransPerCh_Num[np.where(TransPerCh_Num != 0)]);
    print('Expected Trans', ExpectedTrans)
    ## Wave Hunt -- step 1: Compute the Time Lags, Find Unique Waves --> WaveCollection1
    IWI = np.diff(WaveTime);          # IWI = Inter-Wave-Interval...


    ##Recollect small-waves in full waves (involving a wider area).
    MAX_IWI = np.max(IWI)*0.5
    OneMoreLoop = 1;
    
    while OneMoreLoop:
        WnW = np.int32(IWI<=MAX_IWI);
        
        if len(np.where(np.diff(WnW)==1)[0]):

            if np.where(np.diff(WnW)==1)[0][0] != 0:
                ndxBegin_list = np.append(0, np.where(np.diff(WnW)==1)[0]+1)
            else:
                ndxBegin_list = np.where(np.diff(WnW)==1)[0] + 1

            #ndxEnd_list = np.append(np.where(np.diff(WnW)==-1)[0]+1, len(WaveTime)-1)
            ndxEnd_list = ndxBegin_list[1:len(ndxBegin_list)]
            ndxEnd_list = np.append(ndxEnd_list, len(WaveTime)-1)

            
            if len(ndxBegin_list) != len(ndxEnd_list):
                ndxBegin_list = np.append(ndxBegin_list, len(WaveTime)-1)
        else:
            ndxBegin_list = [0]
            ndxEnd_list = [len(WaveTime)]
        FullWave = []

        num_iter = -1
        #for w in range(0, len(ndxBegin_list)):
        for w in range(0, len(ndxEnd_list)):

            num_iter = num_iter +1
            ndx = list(range(ndxBegin_list[w],ndxEnd_list[w]))
            Full_ndx = []
            for elem in ndx:
                Full_ndx.extend(Wave[elem]['ndx'])
            Full_ndx = np.int64(Full_ndx)
            
            FullWave.append({'ndx': Full_ndx,
                           'times': UpTrans[Full_ndx],
                           'ch': ChLabel[Full_ndx],
                           'WaveUnique': len(Full_ndx) == len(np.unique(ChLabel[Full_ndx])),
                           'WaveSize': len(Full_ndx),
                           'WaveTime': np.mean(UpTrans[Full_ndx]),
                           'WaveUniqueSize': len(np.unique(ChLabel[Full_ndx]))
                           });

            # save as single waves isolated transitions.
            if len(ndxBegin_list) < w:
                for j in range(ndxEnd_list[w], ndxBegin_list[w+1]):
                    ndx = [j]
                    Full_ndx = []
                    for elem in ndx:
                        Full_ndx.extend(Wave[elem]['ndx'])
                    Full_ndx = np.int64(Full_ndx)
                    
                    FullWave.append({'ndx': Full_ndx,
                                   'times': UpTrans[Full_ndx],
                                   'ch': ChLabel[Full_ndx],
                                   'WaveUnique': len(Full_ndx) == len(np.unique(ChLabel[Full_ndx])),
                                   'WaveSize': len(Full_ndx),
                                   'WaveTime': np.mean(UpTrans[Full_ndx]),
                                   'WaveUniqueSize': len(np.unique(ChLabel[Full_ndx]))
                                   });


        FullWaveUnique = list(map(lambda x : x['WaveUnique'], FullWave))
        BadWavesNum = len(FullWaveUnique) - len(np.where(FullWaveUnique)[0]);
        OneMoreLoop = 0;

        if len(FullWaveUnique) <= ExpectedTrans: # If not we have an artifactual amplification of small waves...
            if float(BadWavesNum)/len(FullWaveUnique) > ACCEPTABLE_REJECTION_RATE:
                if np.min(FullWaveUnique) == 0: # at lest a Wave non-unique
                    MAX_IWI = MAX_IWI*0.75;
                    OneMoreLoop = 1;
                else: # only unique waves
                    if np.max(WaveSize) < nCh: # at least one wave MUST BE GLOBAL (i.e. involving the full set of electrodes)
                        MAX_IWI = MAX_IWI*1.25;
                        OneMoreLoop = 1;
                        
        if len(ndxBegin_list) == 1:
            OneMoreLoop = 0
        print('IWI', MAX_IWI)
    return(FullWave)






