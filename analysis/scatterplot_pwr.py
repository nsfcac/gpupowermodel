import os.path
from os import path

import pwrcommon as pc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean 

def readData(dataPathP100):
    freqs = [544, 556, 569, 582, 594, 607, 620, 632, 645, 658, 670, 683, 696, 708, 721, 734, 746, 759, 772, 784, 797, 810, 822, 835, 847, 860, 873, 885, 898, 911, 923, 936, 949, 961, 974, 987, 999, 1012, 1025, 1037, 1050, 1063, 1075, 1088, 1101, 1113, 1126, 1139, 1151, 1164, 1177, 1189, 1202, 1215, 1227, 1240, 1252, 1265, 1278, 1290, 1303, 1316, 1328]
    gFLOPS = []
    for f in freqs:
        f = open(dataPathP100+'run_perf_dvfs_'+str(f), 'r')
        gFLOPS.append(float(f.readline().split(" ")[0]))
        f.close()
        
    print (gFLOPS)
    print (len(gFLOPS))

def pltPwrcapTimeIF (arch, app, dataPath, pltPath,thresh):
    tot_time = []
    mean_pwr = []
    r2r_time = []
    r2r_mean_pwr = []
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

        mean_pwr.append(mean(r2r_mean_pwr))
        tot_time.append(mean(r2r_time))
    
    #print ('\nTime: ',tot_time,'\n')
    #print ('\nPower: ',mean_pwr,'\n')
    return pls, mean_pwr, arch, app, pltPath,'PWRCAP',tot_time

def pltDVFSTimeIF (arch, app, dataPath, pltPath,thresh):
    
    tot_time = []
    mean_pwr = []
    r2r_time = []
    r2r_mean_pwr = []

    freqs = []

    if arch == 'V100':
        alt = 0
        f = 135
        while f <= 1380:
            fID = app+'-fp64-run-250-V100-'+str(f)+'-'
            freqs.append(f)
            r2r_mean_pwr.clear()
            r2r_time.clear()
            
            if f == 1005:
                thresh = 85

            for r in range (3):
                d = pc.readMetrics(dataPath+fID+str(r))
                df,s,e = pc.getComputationStartEnd (d,thresh)
                # striping non-computing power consumption
                data = pc.getStablePwr(df,s,e)

                r2r_mean_pwr.append(data['pwr'].mean())
                seconds = max(data['time']) - min(data['time'])
                r2r_time.append(seconds)

            mean_pwr.append(mean(r2r_mean_pwr))
            tot_time.append(mean(r2r_time))
    
            if alt == 0:
                f += 7
                alt = 1
            else:
                f += 8
                alt = 0
        #print ('\nTime: ',tot_time,'\n')
        #print ('\nPower: ',mean_pwr,'\n')
        return freqs, mean_pwr, arch, app, pltPath,'DVFS',tot_time

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

            mean_pwr.append(mean(r2r_mean_pwr))
            tot_time.append(mean(r2r_time))
            
            if f == 1328:
                break
            
            if (path.exists(dataPath+app+'-fp64-run-250-P100-'+str(f+12)+'-0')):
                f += 12
            elif(path.exists(dataPath+app+'-fp64-run-250-P100-'+str(f+13)+'-0')):
                f += 13
            
        
        return freqs, mean_pwr, arch, app, pltPath,'DVFS',tot_time

def pltData ():
    # freqdata, mean_pwr, arch, app, pltPath,mode,tot_time
    archV100 = 'V100'
    archP100 = 'P100'
    appDgemm = 'dgemm'
    appStream = 'stream'
    appFS = 'firestarter'
    thresh = 80 #46 & 85 for DGEMM DVFS V100 and STREAM on both archs
    dataPathV100 = "C:/rf/lbnl/data/phase2/v100-1/results/"
    dataPathP100 = "C:/rf/lbnl/data/phase2/p100-1/results/"
    pltPath = "C:/rf/lbnl/plots/examples/"

    fig, ax = plt.subplots(2, 2)
    # DVFS for DGEMM on P100
    freqdata, mean_pwr, arch, app, pltPath,mode,tot_time = pltDVFSTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)
    ax[0,0].plot(freqdata, mean_pwr,c='red',linestyle='dashed', marker='^', markersize=2)
    # DVFS for DGEMM on V100
    thresh = 46
    freqdata, mean_pwr, arch, app, pltPath,mode,tot_time = pltDVFSTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)
    ax[0,1].plot(freqdata, mean_pwr, c='red',linestyle='dashed', marker='^', markersize=2)
    
    ax[0,0].set_ylabel("DGEMM CUBLAS\nPower (W)",weight='bold',fontsize=10)
    #ax[].set_xlabel("Core Frequency (MHz)",weight='bold',fontsize=10)
    #ax.set_xlabel("Core Frequency (MHz)",weight='bold',fontsize=10)

    ax[0,0].set_title("NVIDIA GP100",weight='bold',fontsize=10)
    ax[0,1].set_title("NVIDIA GV100",weight='bold',fontsize=10)
    # plt.suptitle("CUBLAS DGEMM and BabelStream Power Consumption (W)",weight='bold', fontsize=12)
    #fig.savefig(pltPath+app+'_'+'PWR.png',bbox_inches='tight')


    #fig2, (ax11, ax22) = plt.subplots(1, 2,sharey=True)
    # DVFS for STREAM on P100
    thresh = 80
    freqdata, mean_pwr, arch, app, pltPath,mode,tot_time = pltDVFSTimeIF (archP100, appStream, dataPathP100, pltPath,thresh)
    ax[1,0].plot(freqdata, mean_pwr,c='red',linestyle='dashed', marker='^', markersize=2)
    # DVFS for STREAM on V100
    thresh = 46
    freqdata, mean_pwr, arch, app, pltPath,mode,tot_time = pltDVFSTimeIF (archV100, appStream, dataPathV100, pltPath,thresh)
    ax[1,1].plot(freqdata, mean_pwr, c='red',linestyle='dashed', marker='^', markersize=2)
    ax[1,0].set_ylabel("BabelStream\nPower (W)",weight='bold',fontsize=10)
    ax[1,0].set_xlabel("Core Frequency (MHz)",weight='bold',fontsize=10)
    ax[1,1].set_xlabel("Core Frequency (MHz)",weight='bold',fontsize=10)

    ax[0,0].set_ylim([0, 260])
    ax[0,1].set_ylim([0, 260])
    ax[1,0].set_ylim([0, 260])
    ax[1,1].set_ylim([0, 260])

    # ax11.set_title("NVIDIA GP100",weight='bold',fontsize=10)
    # ax22.set_title("NVIDIA GV100",weight='bold',fontsize=10)
    # plt.suptitle("BabelStream Power Consumption (W)",weight='bold', fontsize=12)
    plt.suptitle("CUBLAS DGEMM and BabelStream Power Consumption (W)",weight='bold', fontsize=12)
    fig.savefig(pltPath+'PWR.png',bbox_inches='tight')


    '''
    #'-ok',
    c = ''
    ax = plt.figure()
    plt.style.use('classic')

    if app == 'dgemm':
        c = 'red'
    elif app == 'stream':
        c = 'blue'
    else:
        c = 'green'   

    plt.plot(freqdata, mean_pwr, 'o',  color=c);
    if mode == 'PWRCAP':
        xLabel = 'PowerCap (W)'
    else:
        xLabel = 'Frequency (MHz)'
    plt.xlabel(xLabel,weight='bold',fontsize=16)
    #plt.xlim(0, 12)
    # Set the y axis label of the current axis.
    plt.ylabel('Power (W)',weight='bold',fontsize=16)
    #plt.ylim(0, 275)
    #plt.ticklabel_format(useOffset=False)
    # Set a title of the current axes.
    #plt.title('Power vs Load')
    #plt.show()
    plt.grid(True)

    plt.savefig(pltPath+arch+'_'+app+'_'+mode+'_pwr.png')
    print('\App:', app)
    print('\nArch:', arch)
    print('\nFreqs/PL:',freqdata)
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
appFS = 'firestarter'
thresh = 85 #46 & 85 for DGEMM DVFS V100 and STREAM on both archs
dataPathV100 = "C:/rf/lbnl/data/phase2/v100-1/results/"
dataPathP100 = "C:/rf/lbnl/data/phase2/p100-1/results/"
pltPath = "C:/rf/lbnl/plots/examples/"

# DVFS for DGEMM on P100
pltDVFSTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)

# DVFS for STREAM on P100
pltDVFSTimeIF (archP100, appStream, dataPathP100, pltPath,thresh)

# DVFS for DGEMM on V100
pltDVFSTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)

# DVFS for STREAM on V100
pltDVFSTimeIF (archV100, appStream, dataPathV100, pltPath,thresh)
'''
# DVFS for FIRESTARTER on P100
#pltDVFSTimeIF (archP100, appFS, dataPathP100, pltPath, thresh)

# DVFS for FIRESTARTER on V100
#pltDVFSTimeIF (archV100, appFS, dataPathV100, pltPath, thresh)

#***** NO NEED *****
# Power capping for DGEMM on P100
#pltPwrcapTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)

# Power capping for DGEMM on V100
#pltPwrcapTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)

# Power capping for STREAM on P100
#pltPwrcapTimeIF (archP100, appStream, dataPathP100, pltPath,thresh)

# Power capping for STREAM on V100
#pltPwrcapTimeIF (archV100, appStream, dataPathV100, pltPath,thresh)




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