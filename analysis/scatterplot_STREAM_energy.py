import os.path
from os import path

import pwrcommon as pc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean 

def pltPwrcapTimeIF (arch, app, dataPath, pltPath,thresh):
    tot_time = []
    mean_pwr = []
    r2r_time = []
    r2r_mean_pwr = []
    mean_energy = []

    fID = ""
    pls = []
    for p in np.arange(125, 255, 5):
        pls.append(p)
        pl = str(p)
        if arch == "P100":
            fID = app+'-fp64-run-'+pl+'-P100-1328-'
        elif arch == "V100":
            fID = app+'-fp64-run-'+pl+'-V100-1380-'
        
        r2r_mean_pwr.clear()
        r2r_time.clear()

        for r in range (3):
            d = pc.readMetrics(dataPath+fID+str(r))
            df,s,e = pc.getComputationStartEnd (d,thresh)
            # striping non-computing power consumption
            data = pc.getStablePwr(df,s,e)

            r2r_mean_pwr.append(data['pwr'].mean())
            seconds = max(data['time']) - min(data['time'])
            r2r_time.append(seconds)
        mpwr = mean(r2r_mean_pwr)
        mTime = mean(r2r_time)
        mean_pwr.append(mpwr)
        tot_time.append(mTime)
        mean_energy.append(mpwr * mTime)
    #print ('\nTime: ',tot_time,'\n')
    #print ('\nPower: ',mean_pwr,'\n')
    return mean_energy,tot_time, mean_pwr, arch, app, pltPath,'PWRCAP',pls

def pltDVFSTimeIF (arch, app, dataPath, pltPath,thresh):
    
    tot_time = []
    mean_pwr = []
    r2r_time = []
    r2r_mean_pwr = []
    mean_energy = []
    freqs = []

    if arch == 'V100':
        alt = 0
        f = 135
        while f <= 1380:
            fID = app+'-fp64-run-250-V100-'+str(f)+'-'
            freqs.append(f)
            r2r_mean_pwr.clear()
            r2r_time.clear()

            for r in range (3):
                d = pc.readMetrics(dataPath+fID+str(r))
                df,s,e = pc.getComputationStartEnd (d,thresh)
                # striping non-computing power consumption
                data = pc.getStablePwr(df,s,e)

                r2r_mean_pwr.append(data['pwr'].mean())
                seconds = max(data['time']) - min(data['time'])
                r2r_time.append(seconds)

            mpwr = mean(r2r_mean_pwr)
            mTime = mean(r2r_time)
            mean_pwr.append(mpwr)
            tot_time.append(mTime)
            mean_energy.append(mpwr * mTime)
    
            if alt == 0:
                f += 7
                alt = 1
            else:
                f += 8
                alt = 0
        print ('\nTime: ',tot_time,'\n')
        print ('\nPower: ',mean_pwr,'\n')
        return mean_energy,tot_time, mean_pwr, arch, app, pltPath,'DVFS',freqs

    elif arch == 'P100':
        f = 544
        while f <= 1329: # 1328
            fID = app+'-fp64-run-250-P100-'+str(f)+'-'
            
            freqs.append(f)

            r2r_mean_pwr.clear()
            r2r_time.clear()

            for r in range (3):
                d = pc.readMetrics(dataPath+fID+str(r))
                df,s,e = pc.getComputationStartEnd (d,thresh)
                # striping non-computing power consumption
                data = pc.getStablePwr(df,s,e)

                r2r_mean_pwr.append(data['pwr'].mean())
                seconds = max(data['time']) - min(data['time'])
                r2r_time.append(seconds)

            mpwr = mean(r2r_mean_pwr)
            mTime = mean(r2r_time)
            mean_pwr.append(mpwr)
            tot_time.append(mTime)
            mean_energy.append(mpwr * mTime)
            
            if f == 1328:
                break
            
            if (path.exists(dataPath+app+'-fp64-run-250-P100-'+str(f+12)+'-0')):
                f += 12
            elif(path.exists(dataPath+'dgemm-fp64-run-250-P100-'+str(f+13)+'-0')):
                f += 13
            
        
        return mean_energy,tot_time, mean_pwr, arch, app, pltPath,'DVFS',freqs

def pltData ():

    k = 1000
    archV100 = 'V100'
    archP100 = 'P100'
    
    appDgemm = 'dgemm'
    appStream = 'stream'

    thresh = 80

    dataPathV100 = "C:/rf/lbnl/data/phase2/v100-1/results/"
    dataPathP100 = "C:/rf/lbnl/data/phase2/p100-1/results/"

    pltPath = "C:/rf/lbnl/plots/examples/"

    fig, axs = plt.subplots(2, 2)

    #DVFS for STREAM on P100
    mean_energy,tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltDVFSTimeIF (archP100, appStream, dataPathP100, pltPath,thresh)
    mean_energy = [int(x/k) for x in mean_energy]
    axs[0, 0].plot(freqs, mean_energy,c='red',label="X",linestyle='dashed', marker='^', markersize=2)

    # Power capping for STREAM on P100
    mean_energy,tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltPwrcapTimeIF (archP100, appStream, dataPathP100, pltPath,thresh)
    mean_energy = [int(x/k) for x in mean_energy]
    axs[0, 1].plot(freqs, mean_energy, c='blue',label='X',linestyle='dashed', marker='^', markersize=2)

    #DVFS for STREAM on V100
    thresh = 46
    mean_energy,tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltDVFSTimeIF (archV100, appStream, dataPathV100, pltPath,thresh)
    mean_energy = [int(x/k) for x in mean_energy]
    axs[1, 0].plot(freqs, mean_energy, c='red',label='X',linestyle='dashed', marker='^', markersize=2)

    # Power capping for STREAM on V100  
    thresh = 80
    mean_energy,tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltPwrcapTimeIF (archV100, appStream, dataPathV100, pltPath,thresh)
    mean_energy = [int(x/k) for x in mean_energy]
    axs[1, 1].plot(freqs, mean_energy, c='blue',label='X',linestyle='dashed', marker='^', markersize=2)

    axs[0,0].set_ylabel("NVIDIA GP100\nEnergy (kJ)",weight='bold',fontsize=10)

    axs[1,0].set_ylabel("NVIDIA GV100\nEnergy (kJ)",weight='bold',fontsize=10)
    axs[1,0].set_xlabel("Core Frequency (MHz)",weight='bold',fontsize=10)

    axs[1,1].set_xlabel("Power Cap (W)",weight='bold',fontsize=10)
    
    axs[0,0].set_ylim([0, 20])
    axs[0,1].set_ylim([0, 20])
    axs[1,0].set_ylim([0, 20])
    axs[1,1].set_ylim([0, 20])

    plt.suptitle("BabelStream Energy Consumption (kJ)",weight='bold',fontsize=12)
    #,bbox_inches='tight'
    fig.savefig(pltPath+'_'+app+'_'+'ENERGY.png',bbox_inches='tight')
    
    '''
    #*************** DGEMM ENERGY PLOT ************************
    fig2, axs2 = plt.subplots(2, 2)

    #DVFS for STREAM on P100
    mean_energy,tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltDVFSTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)
    mean_energy = [int(x / k) for x in mean_energy]
    axs2[0, 0].plot(freqs, mean_energy,c='red',label="X",linestyle='dashed', marker='^', markersize=2)

    # Power capping for STREAM on P100
    mean_energy,tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltPwrcapTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)
    mean_energy = [int(x / k) for x in mean_energy]
    axs2[0, 1].plot(freqs, mean_energy, c='blue',label='X',linestyle='dashed', marker='^', markersize=2)

    #DVFS for STREAM on V100
    thresh = 46
    mean_energy,tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltDVFSTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)
    mean_energy = [int(x / k) for x in mean_energy]
    axs2[1, 0].plot(freqs, mean_energy, c='red',label='X',linestyle='dashed', marker='^', markersize=2)

    # Power capping for STREAM on V100  
    thresh = 80
    mean_energy,tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltPwrcapTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)
    mean_energy = [int(x / k) for x in mean_energy]
    axs2[1, 1].plot(freqs, mean_energy, c='blue',label='X',linestyle='dashed', marker='^', markersize=2)

    axs2[0,0].set_ylabel("NVIDIA GP100\nEnergy (kJ)",weight='bold',fontsize=12)

    axs2[1,0].set_ylabel("NVIDIA GV100\nEnergy (kJ)",weight='bold',fontsize=12)
    axs2[1,0].set_xlabel("Core Frequency (MHz)",weight='bold',fontsize=12)

    axs2[1,1].set_xlabel("Power Cap (W)",weight='bold',fontsize=12)
    
    # axs2[0,0].set_ylim([0, 65])
    # axs2[0,1].set_ylim([0, 65])
    # axs2[1,0].set_ylim([0, 65])
    # axs2[1,1].set_ylim([0, 65])

    plt.suptitle("DGEMM CUBLAS Energy Consumption (kJ)", weight='bold',fontsize=16)
    #,bbox_inches='tight'
    fig2.savefig(pltPath+'_'+app+'_'+'ENERGY.png',bbox_inches='tight')
    '''

    '''
    #'-ok',
    c = ''
    ax = plt.figure()
    plt.style.use('classic')

    if app == 'dgemm':
        c = 'red'
    elif app == 'stream':
        c = 'blue'   
    
    xLabel = None
    if mode == 'PWRCAP':
        xLabel = 'PowerCap (W)'
    else:
        xLabel = 'Frequency (MHz)'

    plt.plot(freqs, mean_energy, 'o',  color=c);
    
    

    plt.xlabel(xLabel,weight='bold',fontsize=16)
    #plt.xlim(0, 12)
    # Set the y axis label of the current axis.
    plt.ylabel('Energy (kJ)',weight='bold',fontsize=16)
    #plt.ylim(0, 275)
    plt.ticklabel_format(useOffset=False)
    # Set a title of the current axes.
    #plt.title('Power vs Load')
    #plt.show()
    plt.grid(True)

    plt.savefig(pltPath+arch+'_'+app+'_'+mode+'_energy.png')
    print('\nMode:', mode)
    print('\nArch:', arch)
    print('\nFreqs/PL:',freqs)
    print('\nPowers:',mean_pwr)
    print('\nTotal Time:',tot_time)
    print('\nTotalRecords:',len(mean_pwr))
    '''

pltData()

'''
archV100 = 'V100'
archP100 = 'P100'

appDgemm = 'dgemm'
appStream = 'stream'

thresh = 85

dataPathV100 = "C:/rf/lbnl/data/phase2/v100-1/results/"
dataPathP100 = "C:/rf/lbnl/data/phase2/p100-1/results/"

pltPath = "C:/rf/lbnl/plots/examples/"


#DVFS for STREAM on P100
pltDVFSTimeIF (archP100, appStream, dataPathP100, pltPath,thresh)

# Power capping for STREAM on P100
pltPwrcapTimeIF (archP100, appStream, dataPathP100, pltPath,thresh)

#DVFS for STREAM on V100
thresh = 46
pltDVFSTimeIF (archV100, appStream, dataPathV100, pltPath,thresh)

# Power capping for STREAM on V100
thresh = 85
pltPwrcapTimeIF (archV100, appStream, dataPathV100, pltPath,thresh)

# Power capping for DGEMM on P100
pltPwrcapTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)

# Power capping for DGEMM on V100
pltPwrcapTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)

#DVFS for DGEMM on V100
thresh = 46
pltDVFSTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)

#DVFS for DGEMM on P100
thresh = 85
pltDVFSTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)
'''



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