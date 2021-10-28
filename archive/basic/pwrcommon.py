# this is the scheme for metric file naming:

# mode = 'run'
# app = 'pymm'
#prec = 'fp64'
# pstate = 'p0'
# pl = 'tdp'
# mclk = 'm715'
# pclk = 'p1189'
# fPrefix = prec +

import time
import pandas as pd
import numpy as np
from datetime import datetime



def readMetrics(file):
        
        df = pd.read_csv(file,sep=r'\s*,\s*',engine='python')
        total = df[['timestamp','index','power.draw [W]']]
        gpu0Pwr = total.loc[total['index'] == 0]
        gpu0Pwr.reset_index(drop=True, inplace=True)

        #gpu0Pwr['time'] = pd.to_datetime(gpu0Pwr['timestamp']).values.astype(np.int64) // 10 ** 6
        #gpu0Pwr['time'] = pd.to_datetime(gpu0Pwr['timestamp']).dt.second
        
        gpu0Pwr['time'] = gpu0Pwr['timestamp'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f').timestamp())
        gpu0Pwr['pwr'] = gpu0Pwr['power.draw [W]'].str.replace('W','').astype(float)
        gpu0Pwr['time'] =  (gpu0Pwr['time'] - (gpu0Pwr['time'].iloc[0]))

        '''
        print ('\nGPU0:',gpu0Pwr)
        print ('\navgTime:',gpu0Pwr['avgTime'])
        print ('\navgPwr:',gpu0Pwr['avgPwr'])
        '''

        return gpu0Pwr[['time','pwr']]

def getComputationStartEnd (gpu0Pwr, powerThreshold):
                
        cpStartTime = gpu0Pwr.loc[gpu0Pwr['pwr'] > powerThreshold]['time'].iloc[0]
        cpEndTime = gpu0Pwr.iloc[::-1].loc[gpu0Pwr['pwr'] > powerThreshold]['time'].iloc[0]
        gpu0Pwr.set_index(['time'])
        return gpu0Pwr,cpStartTime,cpEndTime

def getStablePwr(gpu0Pwr,cpStartTime,cpEndTime):
        
        mask = (gpu0Pwr['time'] >= cpStartTime) & (gpu0Pwr['time'] <= cpEndTime)
        gpu0Pwr = gpu0Pwr.loc[mask]
        return gpu0Pwr
