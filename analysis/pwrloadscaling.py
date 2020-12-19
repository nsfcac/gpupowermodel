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

prec = 'fp64'
app = 'pymm'
mode = 'load'

def pltPwrLoadScaling(output,fileID,loadSize,plotName,pwrThresh=100) :
        loadSize=loadSize[::-1] 
        plt.figure()
        plt.style.use('classic')
        for size in loadSize:
                file = output+fileID+size
                df = pc.readMetrics(file)
        
                # data,start,end = pc.getComputationStartEnd (dff,pwrThresh)
                # df = pc.getStablePwr(data,start,end)

                maxp = max(df['pwr'])
                minp = min(df['pwr'])
                meanp = df['pwr'].mean()
                std = stdev(df['pwr'])
                print ("\nMax: ",maxp,"Min: ",minp,"Mean: ",meanp,"Std: ",std,"\n")
                plt.plot(df['time'], df['pwr'], label = size)
                
        plt.xlabel('Time (s)')
        #plt.xlim(0, 12)
        # Set the y axis label of the current axis.
        plt.ylabel('Power (W)')
        #plt.ylim(0, 275)
 
        plt.ticklabel_format(useOffset=False)

        # Set a title of the current axes.
        #plt.title('Power vs Load')
        # show a legend on the plot
        plt.legend()
        #plt.show()
        plt.grid()
        plt.savefig(output+fileID+plotName)

