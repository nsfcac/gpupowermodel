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
    return tot_time, mean_pwr, arch, app, pltPath,'PWRCAP',pls

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
        print ('\nPower: ',mean_pwr,'\n')
        
        return tot_time, mean_pwr, arch, app, pltPath,'DVFS',freqs

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
            elif(path.exists(dataPath+'dgemm-fp64-run-250-P100-'+str(f+13)+'-0')):
                f += 13
            
        return  tot_time, mean_pwr, arch, app, pltPath,'DVFS',freqs  
        #pltData (tot_time, mean_pwr, arch, app, pltPath,'DVFS',freqs)

def pltData ():
    
    archV100 = 'V100'
    archP100 = 'P100'

    appDgemm = 'dgemm'
    appStream = 'stream'

    thresh = 80

    dataPathV100 = "C:/rf/lbnl/data/phase2/v100-1/results/"
    dataPathP100 = "C:/rf/lbnl/data/phase2/p100-1/results/"
    pltPath = "C:/rf/lbnl/plots/examples/"

    fig, axs = plt.subplots(2, 2)

    
    # DGEMM
    tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltDVFSTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)
    axs[0, 0].plot(freqs, tot_time,c='red',linestyle='dashed', marker='^', markersize=2)

    tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltPwrcapTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)
    l1 = axs[0, 1].plot(freqs, tot_time, c='blue',linestyle='dashed', marker='^', markersize=2)
    
    thresh = 46
    tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltDVFSTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)
    l1 = axs[1, 0].plot(freqs, tot_time, c='red',label='X',linestyle='dashed', marker='^', markersize=2)
    
    thresh = 80
    tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltPwrcapTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)

    l1 = axs[1, 1].plot(freqs, tot_time, c='blue',label='X',linestyle='dashed', marker='^', markersize=2)

    axs[0,0].set_ylabel("NVIDIA GP100\nRuntime (s)",weight='bold',fontsize=10)
    #ax01.set_ylabel("Runtime (sec)",weight='bold',fontsize=12)

    axs[1,0].set_ylabel("NVIDIA GV100\nRuntime (s)",weight='bold',fontsize=10)
    axs[1,0].set_xlabel("Core Frequency (MHz)",weight='bold',fontsize=10)

    axs[1,1].set_xlabel("Power Cap (W)",weight='bold',fontsize=10)
    #ax11.set_ylabel("Runtime (sec)",weight='bold',fontsize=12)
    
    # axs[0,0].set_ylim([0, 1300])
    # axs[0,1].set_ylim([0, 1300])
    # axs[1,0].set_ylim([0, 1300])
    # axs[1,1].set_ylim([0, 1300])

    plt.suptitle("DGEMM CUBLAS Runtime (s)",weight='bold', fontsize=12)
    
    fig.savefig(pltPath+app+'_'+'RUNTIME.png',bbox_inches='tight')
    

    '''
    #'-ok',
    XLabel = ""
    if mode == 'DVFS':
        XLabel = 'Core Frequency (MHz)'
    else:
        XLabel = 'Power Caps (W)'

    c = ''
    fig,ax = plt.subplots()
    #ax = plt.figure()
    plt.style.use('classic')

    if app == 'dgemm':
        c = 'red'
    elif app == 'stream':
        c = 'blue'   

    #plt.plot(tot_time, mean_pwr, 'o',  color=c);
    l1 = ax.plot(freqs, mean_pwr, 'd',  c='red',label="Power (W)");
    
    #plt.xlabel('Time (s)',weight='bold',fontsize=16)
    ax.set_xlabel(XLabel,weight='bold',fontsize=16)
    #plt.xlim(0, 12)
    # Set the y axis label of the current axis.
    ax.set_ylabel('Power (W)',weight='bold',fontsize=16)
    l3 = ax.axhline(y=250, color='b', lw=2,linestyle='--',label="TDP")
    #ax.legend()
    #plt.ylim(0, 275)
    ax.set_ylim(0, 275)
    #ax.ticklabel_format(useOffset=False)
    # Set a title of the current axes.
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    l2 = ax2.plot(freqs, tot_time,marker="^", c='green',label="Execution Time (sec)")
    
    ax2.set_ylabel("Execution Time (sec)",weight='bold',fontsize=16)
    
    #plt.title('Power vs Load')
    #plt.show()
    lns = l1+l2+[l3]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=7)

    ax.grid(True)
    
    fig.savefig(pltPath+arch+'_'+app+'_'+mode+'_time.png',bbox_inches='tight')
    print('\nMode:', mode)
    print('\nArch:', arch)
    print('\nFreqs/PL:',freqs)
    print('\nPowers:',mean_pwr)
    print('\nTotal Time:',tot_time)
    print('\nTotalRecords:',len(mean_pwr))
'''


#pltData ()

# archV100 = 'V100'
# archP100 = 'P100'

# appDgemm = 'dgemm'
# appStream = 'stream'

# thresh = 85

# dataPathV100 = "C:/rf/lbnl/data/phase2/v100-1/results/"
# dataPathP100 = "C:/rf/lbnl/data/phase2/p100-1/results/"
# pltPath = "C:/rf/lbnl/plots/examples/"

# appFS = 'firestarter'
# thresh = 46
# tot_time, mean_pwr, arch, app, pltPath,mode,freqs = pltDVFSTimeIF (archV100, appFS, dataPathV100, pltPath,thresh)


# pltPwrcapTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)
# pltPwrcapTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)
# pltDVFSTimeIF (archP100, appDgemm, dataPathP100, pltPath,thresh)
# thresh = 46
# pltDVFSTimeIF (archV100, appDgemm, dataPathV100, pltPath,thresh)

# Power capping for STREAM on P100
#pltPwrcapTimeIF (archP100, appStream, dataPathP100, pltPath,thresh)

# Power capping for STREAM on V100
#pltPwrcapTimeIF (archV100, appStream, dataPathV100, pltPath,thresh)

# DVFS for STREAM on P100
#pltDVFSTimeIF (archP100, appStream, dataPathP100, pltPath,thresh)

# DVFS for STREAM on V100
#pltDVFSTimeIF (archV100, appStream, dataPathV100, pltPath,thresh)

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