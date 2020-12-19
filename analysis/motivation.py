import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn.palettes import color_palette

'''
def plotArchCapability():
    plt.figure()
    plt.style.use('classic')
    df = pd.DataFrame({'DGEMM':cats, 'GFLOPs/s':data})
    ax = sns.barplot(x="DGEMM", y="GFLOPs/s", data=df)
    fig = ax.get_figure()
    pltName = "dgemm_performance_variations.png"
    path = results + pltName

    #attainableGFs = 4800 #P100=4800, V100=7065
    #ax.axhline(25,linewidth=3, color='red',label="Exascale Power (MW) Target",linestyle='--')
    yX = 8000 #P100=5500
    plt.ylim(0, yX)
    plt.xlabel('DGEMM', weight='bold')  
    plt.ylabel('GFLOPs/s', weight='bold')
    plt.legend(loc='upper center',prop={'size': 9})
    plt.grid(True)
    fig.savefig(path)

def grouped_DEVbarplot(df, cat, subcat, val, path ):
        
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()

    plt.figure()
    plt.style.use('classic')

    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width,label="{}".format(gr))
    plt.xlabel(cat, weight='bold')
    plt.ylabel(val, weight='bold')
    #plt.ylim(0,  300)
    plt.xticks(x, u)
        
    plt.axhline(20,linewidth=3, color='red',label="Exascale Power (MW) Target",linestyle='--')

    plt.legend(loc='center left',prop={'size': 9})
    plt.grid(True)
        #plt.show()
    plt.savefig(path)

def grouped_MWbarplot(df, cat, subcat, val, path ):
        
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()

    plt.figure()
    plt.style.use('classic')

    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width,label="{}".format(gr))
    plt.xlabel(cat, weight='bold')
    plt.ylabel(val, weight='bold')
    #plt.ylim(0,  300)
    plt.xticks(x, u)
        
    plt.axhline(20,linewidth=3, color='red',label="Exascale Power (MW) Target",linestyle='--')

    plt.legend(loc='center left',prop={'size': 9})
    plt.grid(True)
        #plt.show()
    plt.savefig(path)

def plotPwrConsumption():
    plt.figure()
    plt.style.use('classic')
    df = pd.DataFrame({'DGEMM':cats, 'GFLOPs/s':data})
    ax = sns.barplot(x="DGEMM", y="GFLOPs/s", data=df)
    fig = ax.get_figure()
    pltName = "dgemm_performance_variations.png"
    path = results + pltName

    #attainableGFs = 4800 #P100=4800, V100=7065
    ax.axhline(attainableGFs,linewidth=3, color='black',label="Attainable GFLOPs/s",linestyle='--')
    yX = 8000 #P100=5500
    plt.ylim(0, yX)
    plt.xlabel('DGEMM', weight='bold')  
    plt.ylabel('GFLOPs/s', weight='bold')
    plt.legend(loc='upper center',prop={'size': 9})
    plt.grid(True)
    fig.savefig(path)

pltPath = "C:/rf/lbnl/plots/examples/"
df = pd.DataFrame(columns=['Total Power Per Architecture (MW)', 'cat', 'Power (MW)'])
apps = ['GP100','GV100','GA100']
cats = ['Power (MW)','Power (MW)','Power (MW)']
data = [56.6,38.46,41.24]

for n in range(len(data)):
    df.loc[n] = [apps[n]] + [cats[n]] + [data[n]]
        
cat = "Total Power Per Architecture (MW)"
subcat = "cat"
val = "Power (MW)"
pltName = "exa-pwr.png"
results = pltPath + pltName
grouped_MWbarplot(df, cat, subcat, val, results)

# DEVICES
df1 = pd.DataFrame(columns=['No. of GPU Devices Required Per Architecture', 'cat', 'Devices'])
apps = ['GP100','GV100','GA100']
cats = ['GPU Devices','GPU Devices','GPU Devices']
data = [188680,128205,103093]

for n in range(len(data)):
    df1.loc[n] = [apps[n]] + [cats[n]] + [data[n]]
        
cat = "No. of GPU Devices Required Per Architecture"
subcat = "cat"
val = "GPU Devices"
pltName = "exa-dev.png"
results = pltPath + pltName
grouped_DEVbarplot(df1, cat, subcat, val, results)

def GPUCharacteristic(pltPath):
    fig, axs = plt.subplots(2, 3,sharex=True)

    #DVFS for STREAM on P100
    x = ['GP100','GV100','GA100']
    y0 = [56.6,38.46,41.24]
    axs[0, 0].plot(x, y0,c='red',label="X",linestyle='dashed', marker='^', markersize=2, kind='bar', grid=True)
    axs[0,0].set_ylabel("Exaflop System Power Consumption (MW)",weight='bold',fontsize=10)

    # Power capping for STREAM on P100
    y1 = [5.3, 7.8, 9.7]
    axs[0, 1].plot(x, y1,c='red',label="X",linestyle='dashed', marker='^', markersize=2, kind='bar', grid=True)
    axs[0,1].set_ylabel("Peak FP64 TFLOPS",weight='bold',fontsize=10)

    y2 = [188680,128205,103093]
    axs[0, 2].plot(x, y2,c='red',label="X",linestyle='dashed', marker='^', markersize=2, kind='bar', grid=True)
    axs[1,0].set_ylabel("Total GPU Devices",weight='bold',fontsize=10)


    #DVFS for STREAM on V100
    y0 = [300,300,400]
    axs[1, 0].plot(x, y0,c='red',label="X",linestyle='dashed', marker='^', markersize=2, kind='bar', grid=True)
    axs[1,0].set_ylabel("TDP (W)",weight='bold',fontsize=10)

    # Power capping for STREAM on V100  
    y1 = [56, 80, 108]
    axs[1, 1].plot(x, y1,c='red',label="X",linestyle='dashed', marker='^', markersize=2, kind='bar', grid=True)
    axs[1,1].set_ylabel("Total SM Count",weight='bold',fontsize=10)

    y2 = [15.3, 21.1, 54.2]
    axs[1, 2].plot(x, y2,c='red',label="X",linestyle='dashed', marker='^', markersize=2, kind='bar', grid=True)
    axs[1,1].set_ylabel("Total Transistors Count (Billions)",weight='bold',fontsize=10)    
    # axs[0,0].set_ylim([0, 65])
    # axs[0,1].set_ylim([0, 65])
    # axs[1,0].set_ylim([0, 65])
    # axs[1,1].set_ylim([0, 65])

    plt.suptitle("Power Consumption for Exaflop System Using GPUs",weight='bold',fontsize=12)
    #,bbox_inches='tight'
    fig.savefig(pltPath+'exaflop-data.png',bbox_inches='tight')

pltPath = "C:/rf/lbnl/plots/examples/"
GPUCharacteristic(pltPath)
'''
'''
def GPUCharacteristic(pltPath):
    
    plt.figure(figsize=(13,8))
    plt.subplot(2, 3,1)

    #DVFS for STREAM on P100
    x = ['GP100','GV100','GA100']
    y0 = [56.6,38.46,41.24]
    p1 =plt.bar(x, y0,color='red')
    plt.ylabel("Exaflop Power (MW)",weight='bold',fontsize=8)
    plt.axhline(20,linewidth=3, color='green',label="Exascale Power (MW)",linestyle='--')
    plt.legend(loc=0)
    plt.xticks([],[])

    # Power capping for STREAM on P100
    y1 = [5.3, 7.8, 9.7]
    plt.subplot(2, 3,2)
    p2=plt.bar(x, y1,label="X",color='orange')
    plt.ylabel("Peak FP64 TFLOPS",weight='bold',fontsize=8)
    plt.xticks([],[])

    y2 = [189,128,103]
    plt.subplot(2, 3,3)
    p3=plt.bar(x, y2,label="X",color='brown')
    plt.ylabel("GPU Devices (xK)",weight='bold',fontsize=8)
    plt.xticks([],[])

    #DVFS for STREAM on V100
    y0 = [300,300,400]
    plt.subplot(2, 3,4)
    p4=plt.bar(x, y0,label="X",color='royalblue')
    plt.ylabel("TDP (W)",weight='bold',fontsize=8)
    plt.xticks(x)

    # Power capping for STREAM on V100  
    y1 = [56, 80, 108]
    plt.subplot(2, 3,5)
    p5=plt.bar(x, y1,color='olive')
    plt.ylabel("Total SM Count",weight='bold',fontsize=8)
    plt.xticks(x)

    y2 = [15.3, 21.1, 54.2]
    plt.subplot(2, 3,6)
    p6=plt.bar(x, y2,color='limegreen')
    plt.ylabel("Total Transistors Count (Billions)",weight='bold',fontsize=8)    
    plt.xticks(x)
    # axs[0,0].set_ylim([0, 65])
    # axs[0,1].set_ylim([0, 65])
    # axs[1,0].set_ylim([0, 65])
    # axs[1,1].set_ylim([0, 65])
    #plt.grid(True)
    plt.suptitle("Power Consumption for Exaflop System Using GPUs",weight='bold',fontsize=12)
    #,bbox_inches='tight'
    plt.savefig(pltPath+'exaflop-data.png',bbox_inches='tight')
'''
'''
def GPUCharacteristic1(pltPath):
    plt.figure()
    plt.style.use('classic')
    #DVFS for STREAM on P100
    x = ['GP100','GV100','GA100']
    # y0 = [56.6,38.46,41.24]
    # p1 =plt.bar(x, y0,color='red')
    # plt.ylabel("Exaflop Power (MW)",weight='bold',fontsize=8)
    # plt.axhline(20,linewidth=3, color='green',label="Exascale Power (MW)",linestyle='--')
    # plt.legend(loc=0)
    # plt.xticks([],[])

    # Power capping for STREAM on P100
    #y1 = [5.3, 7.8, 9.7]
    y1 = [5300,7800,9700]
    #plt.subplot(2, 3,2)
    plt.plot(x, y1,label="Peak FP64 (GFLOPS)",color='orange')
    #plt.ylabel("Peak FP64 TFLOPS",weight='bold',fontsize=8)
    #plt.xticks([],[])

    #y2 = [189,128,103]
    #plt.subplot(2, 3,3)
    #plt.plot(x, y2,label="GPU Devices (xK)",color='brown')
    # plt.ylabel("GPU Devices (xK)",weight='bold',fontsize=8)
    # plt.xticks([],[])

    #DVFS for STREAM on V100
    #y0 = [300,300,400]
    # plt.subplot(2, 3,4)
    #p4=plt.plot(x, y0,label="TDP (W)",color='royalblue')
    # plt.ylabel("TDP (W)",weight='bold',fontsize=8)
    # plt.xticks(x)

    # # Power capping for STREAM on V100  
    # y1 = [56, 80, 108]
    # plt.subplot(2, 3,5)
    # p5=plt.bar(x, y1,color='olive')
    # plt.ylabel("Total SM Count",weight='bold',fontsize=8)
    # plt.xticks(x)

    y2 = [15.3, 21.1, 54.2]
    #y2 = [15300000000000,21100000000000,54200000000000]
    #plt.subplot(2, 3,6)
    plt.plot(x, y2,label = 'Transistors Count (Billions)',color='limegreen')
    #plt.ylabel("Transistors Count (Billions)",weight='bold',fontsize=8)    
    plt.xticks(x)
    plt.yscale("log")
    
    # axs[0,0].set_ylim([0, 65])
    # axs[0,1].set_ylim([0, 65])
    # axs[1,0].set_ylim([0, 65])
    # axs[1,1].set_ylim([0, 65])
    #plt.grid(True)
    #plt.suptitle("Power Consumption for Exaflop System Using GPUs",weight='bold',fontsize=12)
    #,bbox_inches='tight'
    plt.legend(loc=0)
    plt.grid(True)
    plt.savefig(pltPath+'exaflop-data.png')
'''

def GPUCharacteristic2(pltPath):
    plt.figure()
    plt.style.use('classic')
    #DVFS for STREAM on P100
    x = ['GP100','GV100','GA100']
    y0 = [56.6,38.46,41.24]
    # p1 =plt.bar(x, y0,color='red')
    plt.plot(x, y0,color='red',label='Exaflop Power (MW)')
    # plt.ylabel("Exaflop Power (MW)",weight='bold',fontsize=8)
    plt.axhline(20,linewidth=3, color='green',label="Exascale Power Limit (MW)",linestyle='--')
    # plt.legend(loc=0)
    # plt.xticks([],[])

    # Power capping for STREAM on P100
    #y1 = [5.3, 7.8, 9.7]
    #y1 = [5300,7800,9700]
    #plt.subplot(2, 3,2)
    #plt.plot(x, y1,label="Peak FP64 (GFLOPS)",color='orange')
    #plt.ylabel("Peak FP64 TFLOPS",weight='bold',fontsize=8)
    #plt.xticks([],[])

    #y2 = [189,128,103]
    #plt.subplot(2, 3,3)
    #plt.plot(x, y2,label="GPU Devices (xK)",color='brown')
    # plt.ylabel("GPU Devices (xK)",weight='bold',fontsize=8)
    # plt.xticks([],[])

    #DVFS for STREAM on V100
    y0 = [300,300,400]
    # plt.subplot(2, 3,4)
    plt.plot(x, y0,label="TDP (W)",color='royalblue')
    # plt.ylabel("TDP (W)",weight='bold',fontsize=8)
    plt.xticks(x)

    # # Power capping for STREAM on V100  
    # y1 = [56, 80, 108]
    # plt.subplot(2, 3,5)
    # p5=plt.bar(x, y1,color='olive')
    # plt.ylabel("Total SM Count",weight='bold',fontsize=8)
    # plt.xticks(x)

    #y2 = [15.3, 21.1, 54.2]
    #y2 = [15300000000000,21100000000000,54200000000000]
    #plt.subplot(2, 3,6)
    #plt.plot(x, y2,label = 'Transistors Count (Billions)',color='limegreen')
    #plt.ylabel("Transistors Count (Billions)",weight='bold',fontsize=8)    
    #plt.xticks(x)
    plt.yscale("log")
    
    # axs[0,0].set_ylim([0, 65])
    # axs[0,1].set_ylim([0, 65])
    # axs[1,0].set_ylim([0, 65])
    # axs[1,1].set_ylim([0, 65])
    #plt.grid(True)
    #plt.suptitle("Power Consumption for Exaflop System Using GPUs",weight='bold',fontsize=12)
    #,bbox_inches='tight'
    plt.legend(loc=0)
    plt.grid(True)
    plt.savefig(pltPath+'exaflop-pwr.png')

pltPath = "C:/rf/lbnl/plots/examples/"
GPUCharacteristic2(pltPath)