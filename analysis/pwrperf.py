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
mode = 'load'
app = 'pymm'

def pltIndividual(output,fileID,size):
        
        file = output+fileID+size
        gpu0Pwr = pc.readMetrics(file)
        GPU0MeanPwr = gpu0Pwr['pwr'].mean()
        plt.figure()
        plt.plot(gpu0Pwr['time'], gpu0Pwr['pwr'],label = "Real-time")
        meanGPUPwr = [GPU0MeanPwr for x in range(len(gpu0Pwr))] 
        plt.plot(gpu0Pwr['time'], meanGPUPwr, linestyle = '--',label = "Mean")
        plt.xlabel('Time')
        plt.ylabel('GPU Power Consumption (W)')
        plt.title('Matrix size: '+size+'x'+size)
        plt.legend()
        plt.grid()
        plt.ticklabel_format(useOffset=False)
        plt.savefig(output+fileID+size+'.png')
        
        meanGPUPwr.clear()


def getPwrPerfCounters(output,fileID,size,pwrThresh = 100):
        #os.chdir('../')
        #baseDir = os.getcwd()
        #file = baseDir+'/test/output/'+app+'-'+prec+'-'+mode+'-'+size
        file = output+fileID+size
        # striping non-computing power consumption
        
        data = pc.readMetrics(file)
        
        gdata, cpStartTime,cpEndTime = pc.getComputationStartEnd (data,pwrThresh)
        df = pc.getStablePwr(gdata,cpStartTime,cpEndTime)
        
        #gpu0Pwr['time'] =  (df['time'] - (df['time'].iloc[0]))
        
        tot_seconds = max(df['time']) - min(df['time'])
        #timedelta = tot_time - datetime(1900, 1, 1)

        #tot_seconds = timedelta.total_seconds()
        print ('\nTotal Seconds',tot_seconds)
        s=0.0
        if size == '16' or size == '32' or size == '48' or size == '64' or  size == '72':
                s = float(size) * 320
        else:
                s = float(size)
        
        gflop = 2*s*s*s/1e9
        print('\nGFlops:',gflop)
        if tot_seconds != 0:
                gflops = gflop/tot_seconds
        else:
                gflops = gflop

        return (df['pwr'].mean()),gflops        



def pltPwrPerf(output,fileID,loadSize,thresh):
        flops = []
        meanPwr = []
        fltPerW = []
        normflops = []
        normMeanPwr = []
        for size in loadSize:
                pwr,gfs = getPwrPerfCounters(output,fileID,size,thresh)
                meanPwr.append(pwr)
                flops.append(gfs)
                fltPerW.append(gfs/pwr)

        # TDP
        tdpY = [250,250,250,250,250]
        print ('\nGfltPerW:',fltPerW)
        print ('\nGFlops:',flops)
        print ('\nMeanPwr:',meanPwr)
        # attainable GFLOPS: 7065
        pltGFLOPS(output,fileID,loadSize,meanPwr,tdpY,flops)
        pltGFLOPSWatt(output,fileID,loadSize, meanPwr,tdpY,fltPerW)

def pltGFLOPS(output,fileID,loadSize, meanPwr,tdpY,flops):        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        #tdp line
        lns1 = ax.plot(loadSize, meanPwr, 'H', color = 'b',label = 'MeanPower(W)')
        lns2 = ax.plot(loadSize, tdpY, '--', color = 'b',label = 'Thermal Design Power(TDP)')
        ax2 = ax.twinx()
        #attainable flops line
        attainableY = [7065,7065,7065,7065,7065]
        lns3 = ax2.plot(loadSize, attainableY, '--', color = 'r',label = 'Attainable GFLOPS')
        lns4 = ax2.plot(loadSize, flops, 'o', color = 'r',label = 'GFLOPS')
                
        # added these four lines
        lns = lns1+lns2+lns3+lns4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=4)
        
        ax.grid()
        ax.set_xlabel("Data Points (FP64)")
        ax.set_ylabel("Mean Power Usage (W)")
        ax2.set_ylabel("GFLOPS")
        ax2.set_ylim(0, 4500)
        ax.set_ylim(0,275)
        plt.title('GFLOPS versus Power Usage')
        plt.savefig(output+fileID+'-gflops.png')
        
def pltGFLOPSWatt(output,fileID,loadSize, meanPwr,tdpY,fltPerW):
        #GFLOPS/WATT
        #attainable GFLOPS/WATT: 7065/250=28.26

        Gflopsperwatt = [28.26,28.26,28.26,28.26,28.26]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        #mean power usage
        lns1 = ax.plot(loadSize, meanPwr, 'H', color = 'b',label = 'Power(W)')
        # tdp
        lns2 = ax.plot(loadSize, tdpY, '--', color = 'b',label = 'TDP')
        ax2 = ax.twinx()
        #attainable flops/watt line
        lns3 = ax2.plot(loadSize, fltPerW, 'o', color = 'r',label = 'GFLOPS/Watt')
        lns4 = ax2.plot(loadSize, Gflopsperwatt, '--', color = 'r',label = 'Attainable GFLOPS/Watt')
        # added these four lines
        lns = lns1+lns2+lns3+lns4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=4)

        ax.grid()
        ax.set_xlabel("Data Points (FP64)")
        ax.set_ylabel("Mean Power")
        ax2.set_ylabel("GFLOPS/Watt")
        ax2.set_ylim(0, 22)
        ax.set_ylim(0,275)
        plt.title('GFLOPS/Watt vs Power Usage')
        
        plt.savefig(output+fileID+'-gflopswatt.png')

