import os.path
from os import path

import pwrcommon as pc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean 

def pltPwrcapTimeIF (arch, app, dataPath, pltPath):
    mean_BW = []
    fID = ""
    pls = []
    for p in np.arange(125, 255, 5):
        pls.append(p)
        pl = str(p)
        
        fID = 'run_triads_pl_'+pl

        ff = open(dataPath+fID, 'r') 
        lines = ff.read().splitlines()
        lines = [float(li) for li in lines]
        mean_BW.append(mean(lines[:3]))

    #print ('\nBW: ',mean_BW,'\n')
    return mean_BW, arch, app, pltPath,'PWRCAP',pls

def pltDVFSTimeIF (arch, app, dataPath, pltPath):
    
    mean_BW = []
    freqs = []

    if arch == 'V100':
        alt = 0
        f = 135
        while f <= 1380:
            fID = 'run_triads_dvfs_'+str(f)
            freqs.append(int(f))

            ff = open(dataPath+fID, 'r') 
            lines = ff.read().splitlines()
            lines = [float(li) for li in lines]
            mean_BW.append(mean(lines[:3]))
    
            if alt == 0:
                f += 7
                alt = 1
            else:
                f += 8
                alt = 0
        #print ('\nBandwidth: ',mean_BW,'\n')
        return mean_BW, arch, app, pltPath,'DVFS',freqs

    elif arch == 'P100':
        f = 544
        while f <= 1329: # 1328
            fID = 'run_triads_dvfs_'+str(f)
            freqs.append(int(f))

            ff = open(dataPath+fID, 'r') 
            lines = ff.read().splitlines()
            lines = [float(li) for li in lines]
            mean_BW.append(mean(lines[:3]))
            
            if f == 1328:
                break
            if (path.exists(dataPath+'run_perf_dvfs_'+str(f+12))):
                f += 12
            elif(path.exists(dataPath+'run_perf_dvfs_'+str(f+13))):
                f += 13
            
        #print ('\nBW: ',mean_BW,'\n')
        return mean_BW, arch, app, pltPath,'DVFS',freqs

def pltData ():
    
    #mean_BW, arch, app, pltPath,mode,freqs
    archV100 = 'V100'
    archP100 = 'P100'

    appDgemm = 'dgemm'
    appStream = 'stream'

    dataPathV100 = "C:/rf/lbnl/data/phase2/v100-1/results/"
    dataPathP100 = "C:/rf/lbnl/data/phase2/p100-1/results/"

    pltPath = "C:/rf/lbnl/plots/examples/"

    fig, axs = plt.subplots(2, 2)

    
    # DVFS for STREAM on P100
    mean_BW, arch, app, pltPath,mode,freqs = pltDVFSTimeIF (archP100, appStream, dataPathP100, pltPath)
    mean_BW = [int(x/1000) for x in mean_BW]
    axs[0, 0].plot(freqs, mean_BW,c='red',label="X",linestyle='dashed', marker='^', markersize=2)

    # Power capping for STREAM on P100
    mean_BW, arch, app, pltPath,mode,freqs = pltPwrcapTimeIF (archP100, appStream, dataPathP100, pltPath)
    mean_BW = [int(x/1000) for x in mean_BW]
    axs[0, 1].plot(freqs, mean_BW,c='blue',label="X",linestyle='dashed', marker='^', markersize=2)

    # DVFS for STREAM on V100
    mean_BW, arch, app, pltPath,mode,freqs = pltDVFSTimeIF (archV100, appStream, dataPathV100, pltPath)
    mean_BW = [int(x/1000) for x in mean_BW]
    axs[1, 0].plot(freqs, mean_BW,c='red',label="X",linestyle='dashed', marker='^', markersize=2)

    # Power capping for STREAM on V100
    mean_BW, arch, app, pltPath,mode,freqs = pltPwrcapTimeIF (archV100, appStream, dataPathV100, pltPath)
    mean_BW = [int(x/1000) for x in mean_BW]
    axs[1, 1].plot(freqs, mean_BW,c='blue',label="X",linestyle='dashed', marker='^', markersize=2)

    axs[0,0].set_ylabel("NVIDIA GP100\nHBM (GB/s)",weight='bold',fontsize=10)
    #ax01.set_ylabel("Runtime (sec)",weight='bold',fontsize=12)

    axs[1,0].set_ylabel("NVIDIA sGV100\nHBM (GB/s)",weight='bold',fontsize=10)
    axs[1,0].set_xlabel("Core Frequency (MHz)",weight='bold',fontsize=10)
    
    axs[1,1].set_xlabel("Power Cap (W)",weight='bold',fontsize=10)
    #ax11.set_ylabel("Runtime (sec)",weight='bold',fontsize=12)
    
    axs[0,0].set_ylim([0, 900])
    axs[0,1].set_ylim([0, 900])
    axs[1,0].set_ylim([0, 900])
    axs[1,1].set_ylim([0, 900])

    plt.suptitle("BabelStream HBM (GB/s)",weight='bold', fontsize=12)
    #,bbox_inches='tight'
    fig.savefig(pltPath+'_'+app+'_'+'BW.png',bbox_inches='tight')

    '''
    #'-ok',
    c = ''
    ax = plt.figure()
    plt.style.use('classic')

    if app == 'dgemm':
        c = 'red'
    elif app == 'stream':
        c = 'blue'   

    # create figure and axis objects with subplots()
    #fig,ax = plt.subplots()
    # make a plot
    #ax.plot(freqs, mean_BW, color=c, marker="o")
    # set x-axis label
    xlabel = "Frequeny (MHz)"
    if mode == 'PWRCAP':
        xlabel = "PowerCapp (W)"

    #ax.set_xlabel(xlabel,weight='bold',fontsize=16)
    # set y-axis label
    #ax.set_ylabel("GB/s",color="red",weight='bold',fontsize=16)
    
    # twin object for two different y-axis on the sample plot
    #ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    #ax2.plot(freqs, tot_time,color="blue",marker="o")
    #ax2.set_ylabel("Execution Time (msec)",color="blue",weight='bold',fontsize=16)
    #plt.grid(True)
    #plt.show()
    # save the plot as a file
    #fig.savefig(pltPath+arch+'_'+app+'_'+mode+'_GB.png',bbox_inches='tight')

    plt.plot(freqs, mean_BW, 'o',  color=c);
    plt.xlabel(xlabel,weight='bold',fontsize=16)
    #plt.xlim(0, 12)
    # Set the y axis label of the current axis.
    plt.ylabel('GB/s',weight='bold',fontsize=16)
    #if mode == 'PWRCAP':
    plt.ylim(0, 900)
    plt.ticklabel_format(useOffset=False)
    # Set a title of the current axes.
    #plt.title('Power vs Load')
    #plt.show()
    
    plt.grid(True)

    plt.savefig(pltPath+arch+'_'+app+'_'+mode+'_GB.png')

    print('\nMode:', mode)
    print('\nArch:', arch)
    print('\nFreqs/PL:',freqs)
    print('\nPowers:',mean_BW)
    print('\nTotalRecords:',len(mean_BW))
    '''

pltData()

'''
archV100 = 'V100'
archP100 = 'P100'

appDgemm = 'dgemm'
appStream = 'stream'

dataPathV100 = "C:/rf/lbnl/data/phase2/v100-1/results/"
dataPathP100 = "C:/rf/lbnl/data/phase2/p100-1/results/"

pltPath = "C:/rf/lbnl/plots/examples/"


# DVFS for STREAM on P100
pltDVFSTimeIF (archP100, appStream, dataPathP100, pltPath)

# Power capping for STREAM on P100
pltPwrcapTimeIF (archP100, appStream, dataPathP100, pltPath)

# DVFS for STREAM on V100
pltDVFSTimeIF (archV100, appStream, dataPathV100, pltPath)

# Power capping for STREAM on V100
pltPwrcapTimeIF (archV100, appStream, dataPathV100, pltPath)
'''

#***** DVFS *****
# DVFS for DGEMM on P100
#pltDVFSTimeIF (archP100, appDgemm, dataPathP100, pltPath)

# DVFS for DGEMM on V100
#pltDVFSTimeIF (archV100, appDgemm, dataPathV100, pltPath)


'''
import pandas as pd

from matplotlib.ticker import PercentFormatter

df = pd.DataFrame({'country': [177.0, 7.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0]})
df.index = ['USA', 'Canada', 'Russia', 'UK', 'Belgium', 'Mexico', 'Germany', 'Denmark']
df = df.sort_values(by='country',ascending=False)
df["cumpercentage"] = df["country"].cumsum()/df["country"].sum()*100


fig, ax = plt.subplots()
ax.bar(df.index, df["country"], color="C0")
ax2 = ax.twinx()
ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
plt.savefig("C:/rf/lbnl/plots/examples/pretoexample.png")
'''

'''
#import statistics
import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np
#from itertools import cycle, islice




dPath = "C:/rf/lbnl/data/phase2/p100-1/results/"
#df = pd.DataFrame()
tot_time = []
mean_pwr = []

for pl in np.arange(125, 255, 5):
    pls = str(pl)
    #fID = 'stream-fp64-run-'+pls+'-P100-1328-1'
    fID = 'stream-fp64-run-'+pls+'-V100-1380-0'
    data = pc.readMetrics(dPath+fID)
    mean_pwr.append(data['pwr'].mean())
    tot_seconds = max(data['time']) - min(data['time'])
    tot_time.append(tot_seconds)
    #df[pls] = data['pwr']
    #df['time'] = data['time']
print ('\nTime: ',tot_time,'\n')
print ('\nPower: ',mean_pwr,'\n')
#df.set_index("time", inplace = True, append = True, drop = False)
#df.set_index("time") 
      
# df = pd.DataFrame({
#    'pig': [20, 18, 489, 675, 1776],
#    'horse': [4, 25, 281, 600, 1900]
#    }, index=[1990, 1997, 2003, 2009, 2014])

#lines = df.plot.line()
#lines.figure.savefig("C:/rf/lbnl/plots/examples/DGEMM_PWR_CAP.png")

# read DGEMM Power capping

#x = np.linspace(0, 10, 30)
#y = np.sin(x)
#'-ok',
plt.plot(tot_time, mean_pwr, 'o',  color='blue');
pltPath = "C:/rf/lbnl/plots/examples/pwr_time_scatter.png"
plt.savefig(pltPath)
'''