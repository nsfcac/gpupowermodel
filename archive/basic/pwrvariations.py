import time
import pandas as pd
from datetime import datetime
import numpy as np

from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import pearsonr
import os
from matplotlib.font_manager import FontProperties
from statistics import stdev 

import pwrcommon as pc

app = 'pymm'
prec = 'fp64'
mode = 'load'

def getVariationSlices (gpu0Pwr,cpStartTime,cpEndTime):
        upMask =  (gpu0Pwr['time'] >= gpu0Pwr['time'].iloc[0]) & (gpu0Pwr['time'] <= cpStartTime)
        stableMask = (gpu0Pwr['time'] >= cpStartTime) & (gpu0Pwr['time'] <= cpEndTime)
        downMask = (gpu0Pwr['time'] >= cpEndTime) & (gpu0Pwr['time'] <= gpu0Pwr['time'].iloc[-1])

        return gpu0Pwr.loc[upMask],gpu0Pwr.loc[stableMask],gpu0Pwr.loc[downMask]
        

def pltVarations(output,fileID,r,pwrThresh = 100):
        file = output+fileID+r
        gpu0Pwr = pc.readMetrics(file)
        # striping non-computing power consumption
        data,start,end = pc.getComputationStartEnd (gpu0Pwr,pwrThresh)
        rampUp, stable, rampDown = getVariationSlices(data,start,end)
    
        '''
        print ('\ntotal length:', len(gpu0Pwr))
        print('\nrampUp:',rampUp)
        print ('\nrampUp length:',len(rampUp))
 
        print('\nStable:',stable)
        print ('\nStable length:',len(stable))
        
        print('\nrampDown:',rampDown)
        print ('\nrampDown length:',len(rampDown))
        '''

        pltRampUp(rampUp,output,fileID)
        pltStable(stable,output,fileID)
        pltRampDown(rampDown,output,fileID)

def pltRampUp(rampUp,output,fileID):
        # power ramping up
        plt.figure()
        plt.plot(rampUp['time'], rampUp['pwr'])
        plt.xlabel('Time(s)')
        plt.ylabel('GPU Power Consumption (W)')
        plt.title('Power Ramping Up Analysis')
        plt.grid()
        plt.ticklabel_format(useOffset=False)
        plt.savefig(output+fileID+'rampup.png')

def pltStable(stable,output,fileID):
        # power stable
        plt.figure()
        plt.plot(stable['time'], stable['pwr'])
        plt.xlabel('Time(s)')
        plt.ylabel('GPU Power Consumption (W)')
        # plt.ylim(200, 275)
        maxp = max(stable['pwr'])
        minp = min(stable['pwr'])
        meanp = stable['pwr'].mean()
        std = stdev(stable['pwr'])
        plt.title( 'Min: %.2f'%(minp)+'W, Mean: %.2f'%(meanp)+'W, Max: %.2f'%(maxp)+'W, Std: %.2f'%(std)+'W')
        plt.suptitle('Peak Power Stablization  Analysis')
        plt.grid()
        plt.ticklabel_format(useOffset=False)
        plt.savefig(output+fileID+'stable.png')
        
def pltRampDown(rampDown,output,fileID):
        # power ramping down
        plt.figure()
        plt.plot(rampDown['time'], rampDown['pwr'])
        plt.xlabel('Time(s)')
        plt.ylabel('GPU Power Consumption (W)')
        # plt.ylim(0, 275)
        plt.title('Power Ramping Down Analysis')
        plt.grid()
        plt.ticklabel_format(useOffset=False)
        plt.savefig(output+fileID+'rampdown.png')
