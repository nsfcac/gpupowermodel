from matplotlib import style
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from seaborn.palettes import color_palette

ls = 14
ts = 10
ms = 75

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
    plt.savefig(pltPath+'exaflop-pwr-usage.png')

def Plt_models_MAE (pltPath):
    models = []
    algs = []
    vals = []
    # MLR
    MAPE = [4.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 3.0, 4.0, 4.0, 3.0, 5.0, 4.0, 3.0, 5.0, 4.0] 
    models.append('Power Model')
    algs.append('MLR')
    vals.append(100-np.mean(MAPE))

    # RFR
    MAPE = [1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    models.append('Power Model')
    algs.append('RFR')
    vals.append(100-np.mean(MAPE))
    print ('PWR-RFR:',100-np.mean(MAPE))
    # SVR
    MAPE = [4.0, 4.0, 3.0, 3.0, 2.0, 5.0, 3.0, 3.0, 5.0, 4.0, 3.0, 3.0, 6.0, 3.0, 3.0, 4.0, 3.0, 5.0, 3.0]
    models.append('Power Model')
    algs.append('SVR')
    vals.append(100-np.mean(MAPE))

    # XGBoost
    MAPE = [1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0]
    models.append('Power Model')
    algs.append('XGBR')
    vals.append(100-np.mean(MAPE))
    print ('PWR-XGBR:',100-np.mean(MAPE))
    # MLR
    MAPE =  [44.0, 21.0, 39.0, 22.0, 11.0, 36.0, 1.0, 3.0, 44.0, 1.0, 13.0, 20.0, 36.0, 25.0, 36.0, 51.0, 15.0, 39.0, 24.0] 
    models.append('Performance Model')
    algs.append('MLR')
    vals.append(100-np.mean(MAPE))

    # RFR
    MAPE =  [3.0, 6.0, 5.0, 5.0, 2.0, 4.0, 1.0, 2.0, 5.0, 1.0, 2.0, 5.0, 3.0, 4.0, 4.0, 3.0, 2.0, 3.0, 3.0]
    models.append('Performance Model')
    algs.append('RFR')
    print ('PERF-RFR:',100-np.mean(MAPE))
    vals.append(100-np.mean(MAPE))

    # SVR
    MAPE =  [36.0, 22.0, 39.0, 31.0, 14.000000000000002, 34.0, 4.0, 9.0, 41.0, 5.0, 19.0, 16.0, 38.0, 22.0, 27.0, 28.000000000000004, 17.0, 35.0, 28.000000000000004]
    models.append('Performance Model')
    algs.append('SVR')
    vals.append(100-np.mean(MAPE))

    # XGBoost
    MAPE =  [3.0, 4.0, 5.0, 4.0, 1.0, 5.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 4.0, 5.0, 3.0, 3.0, 3.0, 3.0, 5.0]
    models.append('Performance Model')
    algs.append('XGBR')
    vals.append(100-np.mean(MAPE))
    print ('PERF-XGBR:',100-np.mean(MAPE))

    d = {'Model':models,'Accuracy (%)':vals,'Regressor':algs}
    df = pd.DataFrame(data=d)
    
    ax = plt.figure().add_subplot(111)
    # plt.style.use('classic')

    sns.set_context("paper",font_scale=1.5)    
    sns.set_style("whitegrid")

    # df = df.sort_values('Models')
    #  ['firebrick','forestgreen']
    # graph = df.plot(ax=ax, kind='bar', x="GPU Microarchitecture",y="Power (MW)",color=['mediumseagreen','tomato'],legend=False) palette='rocket'
    graph = sns.barplot(x="Regressor",y="Accuracy (%)",hue='Model',data=df, palette='hot')
    
    # graph.axhline(20, label='Exascale Power Threshold',lw=2, color='r',ls='--')

    import itertools 
    hatches = itertools.cycle(['/','+', '-', 'x', '//', '*', 'o', 'O', '.', '\\'])
    num_locations = len(df['Regressor'].unique())
    for i, patch in enumerate(graph.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)

    # for index, row in df.iterrows():
        # graph.text(row.name, row['Importance Score'],round(row['Importance Score'],2), color='black', ha="center",fontsize=10,weight='bold')
    #  
    ax.legend(loc=0,ncol=2,  fontsize= 12) #bbox_to_anchor=(1, 1)
    
    ax.set_xlabel("Regressors",weight='bold',fontsize=ls)
    ax.set_ylabel('Accuracy (%)',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    ax.set_ylim([0, 112])
    plt.xticks(fontsize=12)
    plt.grid(True)
    # change_width(ax, .25)

    # _show_on_single_plot(ax)

    plt.tight_layout()

    plt.savefig(pltPath+'models-accuracy.png', transparent=True,dpi=600)

    # plt.figure() 
    # # plt.style.use('classic')
    # sns.set_style('whitegrid')
    # # sns.set_style("ticks")
    # sns.set_context("paper",font_scale=1.5)
    
    # g = sns.catplot(x="Score/Error Type", y="Score/Error",
    #             col="ML Model",
    #             data=df, kind="box",
    #             order=["MedAE", "EVS",'R2S','MAE','MAPE','RMSE'],
    #             col_wrap=2,
    #             height=4, aspect=.7);
    
    # g.set_xticklabels(rotation=70,weight='bold',fontsize=ts)
    # g.set_xlabels(weight='bold',fontsize=ls)
    # g.set_ylabels(weight='bold',fontsize=ls)
    # g.despine(left=True)

    # for ax in g.axes.flat:
    #     ax.set_title(ax.get_title(), weight='bold',fontsize=ls)
    
    # plt.savefig('C:/rf/models-accuracy.png',transparent=True, bbox_inches='tight')

def Plt_models_error (pltPath):
    mlr = []
    set = []
    se = []
    
    # MLR
    MAE = [2.06, 4.08, 1.87, 2.49, 1.53, 2.01, 0.97, 1.13, 3.59, 0.87, 1.18, 2.63, 1.74, 2.21, 3.89, 2.54, 2.47, 2.7, 2.54]
    MAPE = [4.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 3.0, 4.0, 4.0, 3.0, 5.0, 4.0, 3.0, 5.0, 4.0]
    MSE = [7.81, 30.22, 5.28, 11.14, 4.9, 8.84, 1.54, 2.39, 21.4, 1.11, 1.84, 13.53, 6.14, 7.87, 25.16, 9.75, 14.36, 11.84, 9.32]
    RMSE = [2.79, 5.5, 2.3, 3.34, 2.21, 2.97, 1.24, 1.55, 4.63, 1.05, 1.35, 3.68, 2.48, 2.81, 5.02, 3.12, 3.79, 3.44, 3.05]
    MedAE = [1.47, 2.97, 1.74, 1.93, 0.91, 1.17, 0.77, 0.78, 2.98, 0.8, 1.22, 2.28, 1.36, 1.82, 2.78, 2.35, 1.41, 2.29, 2.19]
    EVS = [0.96, 0.97, 0.98, 0.99, 0.97, 0.93, 0.91, 0.95, 0.97, 0.92, 0.92, 0.95, 0.96, 0.97, 0.97, 0.97, 0.98, 0.96, 0.97]
    R2S = [0.96, 0.97, 0.98, 0.99, 0.97, 0.93, 0.91, 0.95, 0.96, 0.92, 0.92, 0.95, 0.96, 0.97, 0.97, 0.97, 0.97, 0.96, 0.97]
    se = MAPE+MAE+RMSE+MedAE+EVS+R2S
    
    for i in range(19*6):
        mlr.append('MLR')
    for i in range(19):
        set.append('MAPE')
    for i in range(19):
        set.append('MAE')
    for i in range(19):
        set.append('RMSE')
    for i in range(19):
        set.append('MedAE')
    for i in range(19):
        set.append('EVS')
    for i in range(19):
        set.append('R2S')
    
    # RFR
    MAE = [0.75, 1.23, 0.39, 0.73, 0.37, 0.86, 0.18, 0.38, 1.08, 0.16, 0.33, 0.89, 0.43, 0.81, 1.0, 0.39, 0.93, 0.58, 0.67]
    MAPE = [1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    MSE = [1.45, 3.36, 0.26, 2.73, 0.37, 2.35, 0.1, 0.27, 3.24, 0.11, 0.34, 4.49, 0.36, 2.42, 3.14, 0.71, 3.15, 0.65, 0.93]
    RMSE = [1.21, 1.83, 0.51, 1.65, 0.61, 1.53, 0.32, 0.52, 1.8, 0.33, 0.59, 2.12, 0.6, 1.56, 1.77, 0.84, 1.78, 0.8, 0.96]
    MedAE = [0.45, 0.79, 0.34, 0.38, 0.24, 0.32, 0.08, 0.25, 0.64, 0.06, 0.17, 0.5, 0.32, 0.43, 0.3, 0.21, 0.41, 0.43, 0.41]
    EVS = [0.99, 1.0, 1.0, 1.0, 1.0, 0.98, 0.99, 1.0, 0.99, 0.99, 0.98, 0.98, 1.0, 0.99, 1.0, 1.0, 0.99, 1.0, 1.0]
    R2S = [0.99, 1.0, 1.0, 1.0, 1.0, 0.98, 0.99, 1.0, 0.99, 0.99, 0.98, 0.98, 1.0, 0.99, 1.0, 1.0, 0.99, 1.0, 1.0]
    se = se+MAPE+MAE+RMSE+MedAE+EVS+R2S
    
    for i in range(19*6):
        mlr.append('RFR')
    for i in range(19):
        set.append('MAPE')
    for i in range(19):
        set.append('MAE')
    for i in range(19):
        set.append('RMSE')
    for i in range(19):
        set.append('MedAE')
    for i in range(19):
        set.append('EVS')
    for i in range(19):
        set.append('R2S')

    # SVR
    MAE = [2.62, 3.7, 1.78, 2.58, 1.48, 3.04, 1.13, 1.51, 3.38, 1.54, 1.17, 2.32, 2.47, 2.0, 3.43, 2.29, 2.62, 3.3, 2.37]
    MAPE = [4.0, 4.0, 3.0, 3.0, 2.0, 5.0, 3.0, 3.0, 5.0, 4.0, 3.0, 3.0, 6.0, 3.0, 3.0, 4.0, 3.0, 5.0, 3.0]
    MSE = [18.58, 30.4, 9.13, 12.59, 6.07, 13.48, 2.23, 3.81, 22.54, 6.65, 3.54, 18.78, 7.64, 9.11, 25.6, 11.19, 16.6, 25.22, 18.11]
    RMSE = [4.31, 5.51, 3.02, 3.55, 2.46, 3.67, 1.49, 1.95, 4.75, 2.58, 1.88, 4.33, 2.76, 3.02, 5.06, 3.35, 4.07, 5.02, 4.26]
    MedAE = [1.5, 2.14, 0.84, 2.09, 0.79, 2.67, 0.76, 1.21, 2.71, 0.79, 0.77, 0.76, 2.47, 1.42, 2.35, 1.83, 1.4, 1.81, 1.19]
    EVS = [0.93, 0.97, 0.97, 0.98, 0.97, 0.9, 0.88, 0.95, 0.97, 0.65, 0.86, 0.93, 0.96, 0.97, 0.97, 0.97, 0.97, 0.93, 0.94]
    R2S = [0.92, 0.97, 0.97, 0.98, 0.96, 0.9, 0.87, 0.95, 0.97, 0.61, 0.86, 0.93, 0.93, 0.97, 0.97, 0.97, 0.97, 0.92, 0.94]
    se = se+MAPE+MAE+RMSE+MedAE+EVS+R2S
    
    for i in range(19*6):
        mlr.append('SVR')
    for i in range(19):
        set.append('MAPE')
    for i in range(19):
        set.append('MAE')
    for i in range(19):
        set.append('RMSE')
    for i in range(19):
        set.append('MedAE')
    for i in range(19):
        set.append('EVS')
    for i in range(19):
        set.append('R2S')

    # XGBoost
    MAE = [0.83, 2.25, 0.55, 0.84, 0.44, 0.95, 0.26, 0.45, 0.72, 0.23, 0.29, 1.01, 0.54, 0.93, 1.25, 0.49, 1.21, 0.78, 0.82]
    MAPE = [1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0]
    MSE = [2.58, 26.29, 0.59, 1.75, 0.45, 4.44, 0.16, 0.39, 1.23, 0.13, 0.19, 2.42, 0.62, 2.63, 4.94, 0.59, 6.77, 1.24, 1.84]
    RMSE = [1.61, 5.13, 0.77, 1.32, 0.67, 2.11, 0.4, 0.62, 1.11, 0.36, 0.44, 1.56, 0.79, 1.62, 2.22, 0.77, 2.6, 1.12, 1.36]
    MedAE = [0.29, 0.72, 0.38, 0.51, 0.23, 0.3, 0.13, 0.26, 0.4, 0.11, 0.17, 0.59, 0.32, 0.41, 0.56, 0.3, 0.38, 0.61, 0.34]
    EVS = [0.99, 0.97, 1.0, 1.0, 1.0, 0.97, 0.99, 0.99, 1.0, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0, 1.0, 0.99, 1.0, 0.99]
    R2S = [0.99, 0.97, 1.0, 1.0, 1.0, 0.96, 0.99, 0.99, 1.0, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0, 1.0, 0.99, 1.0, 0.99]
    se = se+MAPE+MAE+RMSE+MedAE+EVS+R2S
    
    for i in range(19*6):
        mlr.append('XGBR')
    for i in range(19):
        set.append('MAPE')
    for i in range(19):
        set.append('MAE')
    for i in range(19):
        set.append('RMSE')
    for i in range(19):
        set.append('MedAE')
    for i in range(19):
        set.append('EVS')
    for i in range(19):
        set.append('R2S')

    d = {'ML Model':mlr,'Score/Error Type':set,'Score/Error':se}
    df = pd.DataFrame(data=d)
    
    # data = gpu_data.groupby(['arch','app', 'clocks.current.sm [MHz]']).mean().reset_index()
    # data.rename(columns={'arch': 'GPU Microarchitecture'}, inplace=True)
    # data.replace(to_replace ="P100", value ="NVIDIA GP100", inplace=True) 
    # data.replace(to_replace ="V100", value ="NVIDIA GV100", inplace=True)
    # figsize=(10, 5)
    plt.figure() 
    # plt.style.use('classic')
    sns.set_style('whitegrid')
    # sns.set_style("ticks")
    sns.set_context("paper",font_scale=1.5)

    # sns.set_context('paper', font_scale = 1)
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    # g = sns.FacetGrid(df, col="ML Model", height=2, col_wrap=2)
    # g.map(sns.boxplot,x="Score/Error Type", y="Score/Error")
    # sns.boxplot(hue='ML Models',data=df, orient="v", palette="hls")
    
    g = sns.catplot(x="Score/Error Type", y="Score/Error",
                col="ML Model",
                data=df, kind="box",
                order=["MedAE", "EVS",'R2S','MAE','MAPE','RMSE'],
                col_wrap=2,
                height=4, aspect=.7);
    # Set labels
    # plt.ylabel("Error/Score", weight='bold',fontsize=ls)
    # plt.xlabel("ML Models", weight='bold',fontsize=ls)
    # plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.set_xticks(rotation=70)
    
    g.set_xticklabels(rotation=70,weight='bold',fontsize=ts)
    # g.set_yticklabels(weight='bold',fontsize=ts)
    # g.set_yticks(weight='bold',size=ts)
    g.set_xlabels(weight='bold',fontsize=ls)
    g.set_ylabels(weight='bold',fontsize=ls)
    # g.set(ylim=(0, 6), yticks=[0,1,2,3,4,5,6])
    # calculate numbre of samples:
    g.despine(left=True)

    for ax in g.axes.flat:
        ax.set_title(ax.get_title(), weight='bold',fontsize=ls)
        # This only works for the left ylabels
        # ax.set_ylabel(ax.get_ylabel(), fontsize=ls)
    
    # plt.legend(loc='upper left',ncol=1)
    # plt.grid(True)
    # g.set_style("darkgrid")
    
    plt.savefig('C:/rf/models-accuracy.png',transparent=True, bbox_inches='tight')

    # vals = []
    # for i,v in enumerate(importance):
    #     vals.append(v)
    #     print('Feature: %0d, Score: %.5f' % (i,v))

    # plt.figure()
    # labels = list(p100_data.columns)
    # plotdata = pd.DataFrame({
    #     "Features":vals
    # }, 
    #     index=labels[2:]
    # )
    # colors = 'rgbkymc'  #red, green, blue, black, etc.
    # plotdata.plot(kind="bar",color = colors,rot=60,legend = None )
    # plt.xlabel("Features",weight='bold',fontsize=ls)
    # plt.ylabel('Importance Score',weight='bold',fontsize=ls)
    # # plt.margins(0.05)

    # plt.xticks(fontsize=ts)
    # plt.grid(True)
    # plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.tight_layout()
    # plt.savefig('C:/rf/'+app+'-'+num_features+'featuresofimportance.png')
    # # plt.show()
    # print (labels[2:])
def get_pwr_diffs():
    diff2 = [0.7265144840316253, -2.480373974552272, -2.4067917644012695, -1.1492172393144386, -1.7754046588117873, 0.02137861417333653, 0.7364090514172545, -1.795937838809131, -0.017547403741382084, 1.6458853259368738, -0.02126590979980847, -0.15707650521322591, 0.06148510743746982, -0.30896248088555467, 0.07746918552113158, -0.11357273707071869, -8.656464417737965, -1.6887179574631688, 18.958516177512408, 0.04531980030775884, 0.5995045113277442, -0.12116566067101786, 0.0735360458416281, -0.023522581320051472, 1.685891913262516, 0.5489309754722171, 3.9166351560199644, -0.8007999999512023, 0.7794323410051192, -7.408037209503746, -0.00025595726823013365, -5.684575075840542, -2.372647013169839, 5.985464936057326, 1.8164197736245313, 0.016626441712872975, 2.843139461558785, -0.4354935528098167, -0.7459379342875749, -4.696030951000168, 0.0091646409524202, -0.05718714527773727, -4.464766166462908, -0.9804976809892239, 1.3306331344450797, 1.2949822252606822, -0.21075275269698324, -1.4168615462646272, 0.000603844942425269, -2.4472587216832054, -5.48338367399046, 0.17489898137863946, -0.012350713553473724, -0.841495052620516, -1.416019748459874, -6.078624921044906, -0.06992099219493753, -8.826792812985317, -4.713055266978984, -0.1364085046563872, -0.06459274498032386, 0.021986407450036438, -3.178009715214145, -0.21560363460903176, -2.5709156108224676, -1.4824835664613758, -1.171556592863304, -0.06820575578201726, 1.791603944151845, -0.7125413009783941, 0.011473944427038418, 1.4883314939548455, 1.7570796541574225, -2.163779922077431, 0.08723201630096611, 0.5510289703414912, -0.030087170283565, -0.03609746182754492, -0.12143027998440203, -0.05613659769228008, -0.12269634903867654, 1.7288867762074034, 0.9859351464230173, 1.9406520525850581, -1.3137571211975114, -0.32079618580631575, -0.5487531874192939, 0.41485538022473634, -2.3527866038060097, -1.272299177324328, -0.8207098620269306, -2.0689103501818664, -0.7753182133172629, -1.5850771010929066, -1.1061578588889915, 0.07909740189474235, 0.06634135618183734, -1.7291489468724563, 1.7763316720085172, 0.10007504316722304, -3.1446069163126538, -3.5247258020713446, 6.3525631656007135, -1.098685466250764, -0.4837118977422534, 0.016149860709433028, -0.15919537755372204, 0.031183817508804168, -4.776970796069833, 1.9273854051159844, -2.1897823628308117, 0.6476416489245906, 0.14865248522221464, 2.4158692694597903, 0.034523374410170504, -3.7347585778951498, -7.3280436943041565, -14.433516720087653, 4.634988813942726, -0.04443033923447359, -1.8826981444482271, -4.662279921582709, -1.4434738427647034, -8.409540673482425, 0.06175181018170406, -1.2145897054320258, 3.5243030682453877, -3.2072183755584263, -0.6163407059346753, 0.39702324991183957, 1.8671465593983712, -5.16105850347418, 1.4120990524062549, -2.502036092010357, -20.28448466809047, -1.8057836588911726, -0.006469174186022997, 2.245960265753439, -1.9556407203787813, 0.0092113826846969, 0.08994175877637645, 4.759563572461502, 4.164165053489398, -0.4470591212506889, -15.726025818237915, 0.025200054276851347, 4.8050246198992, -1.4774699411427221, -2.8009203079253524, -0.5220645449374643, -1.5335444049396898, -0.019166496441791026, 3.1622022051836325, -1.7657071813512033, -22.39300230423403, -3.892415488607895, 1.3113303624563173, 0.09139647550381369, 0.25344069329894126, 0.9700381333327925, -0.7931749205984531, 1.73122862891465, -4.978510225969444, -0.3300037398383182, 1.9887295256072122, -0.8121319891096732, 2.837593494905974, 3.649054433305622, -0.4518571497440149, -0.34478449364127073, 1.0064916724362192, -0.4723444048671297, 0.26351344587834546, 0.06940884355559973, -0.412448785031188, 0.6199882413591737, 0.10129649419609166, -0.10752803766766306, 0.8650700577662249, 0.0436837493547344, -0.037495712256450986, -1.4027823065510319, 0.4440555644109594, 1.056406022309055, -0.4916640664356038, 0.6116619047484662, 1.0028097864556145, 0.03842577394559754, -0.295956706204322, -0.052591181483599314, 0.3728571472222413, -1.0687234378754198, -2.1145179235046925, 0.2700310079350885, -0.125677974564411, 1.7995643471475589, -1.4584406459496293, -1.0654088140136793, -0.04546995521302932, 1.571402744197016, -0.3079149401818313, 0.5656813643365126, 0.18368330076978623, -0.08798152273246984, 0.038635841285199035, -0.09086595822272159, -1.344992003314509, -1.9560123256590174, -0.7937333562355064, -0.010273167158437957, 0.9362692415431155, 
-0.06442641239605962, -0.627472514042978, 0.20426595377419687, -0.48418082693922315, -0.28668329480132115, 1.1903908081685444, -0.6524198614983163, -0.16232689858416904, 0.5005614901408393, 0.27667960091405064, -0.2354236560351879, -0.10697509601945399, -0.2409869186457172, 0.03734272111331194, -1.9243080167310467, -0.5518547410065082, -1.45665851498174, 0.1599277854661736, -1.2380041056588666, 1.0431271251440961, -0.014244385947677074, -0.7286950927219991, -0.18153806662370187, -0.938499432694087, 0.23839183022714394, -2.4415400396440106, -0.6209922630657729, -0.9906454452918467, -0.009611777174526992, 0.41824032014154966, -0.933698796373875, 0.03473451493815105, -0.3308898830247955, 0.058165362224961825, 0.14840187767330093, -0.43456681536960673, 0.8188855460497209, -0.005163228723937152, 0.21491231107415842, 0.1527650021109892, -0.4895971480870571, -7.823565910617916, -9.841912329956315, -0.6908615266435021, -0.32520020989619525, -6.284719839750977, 0.8875812159914886, -11.308847297797598, -1.3200668318338984, -2.5257318305968397, -11.790624597067676, 1.0043238434918536, 0.0073138050443688485, 0.10579416801040509, 0.9487729378975303, -0.22645868013486137, -0.024509907037476353, -5.431060828587533, -2.8250879106195725, -0.7543338572644984, -1.1998853841143173, -0.6002660585808712, 0.8080132417768908, 0.023984060318277045, -0.005472509497579381, -6.056485354735628, -1.9929620373082315, -5.711615213588004, 0.07098414710641521, -0.352736121294015, -8.406732454768829, 1.3787650694381313, 0.06504646705143102, -0.8677425092136133, -10.900436541369913, 5.158484779303848, 0.03235360224766026, -12.436676392042571, -14.99649193309591, -3.4555756785188407, -18.52688863805305, 0.5091664839246235, 1.1254337627951116, -6.268898661672424, 3.356610980810302, 2.6547292088512933, 0.8330347955783708, 1.3702658351348873, -11.255780731669219, 0.26788705630173837, -7.636351059657173, -3.7990377105850115, 2.604807389997987, -0.11410164897024799, -2.7773406454935383, -12.843579742618573, 0.02324766687971902, -0.07634602753025632, 3.212164819284098, 8.211335218640485, 0.040962076455421936, -0.4305926369507205, 0.024583664692194418, -11.940389091706606, 1.1359604655742004, 1.1755076013094907, -3.889830572585737, -0.5531567230093088, -0.0010936452952634568, 17.228641984683293, -1.226160542634375, -11.592728264387254, -0.1392062843415829, 2.8765751722314974, -0.1809541607575227, 0.016976247372731734, -0.1382183274984783, 1.3409553720168077, -2.162021643243797, -8.861310060716391, -0.060076357492455656, 0.5366731569278045, -18.284572106143912, 0.013631698962157657, 2.0882775727846337, -2.648234319416531, -12.537301466463617, -5.696691183050902, 0.2592791451510408, -0.7994142235820192, -0.11237150005007379, -7.58238537545445, 0.14946919760288324, -0.5457396890236907, -11.85014960600131, 0.015178826225621833, -0.007970495411036893, 0.16476618640080432, -0.2560684206629418, -0.02581545240325056, -0.04327046281629521, -5.86850651625042, -0.36433758340908184, 8.587528833109218, -0.006163341876380457, -2.2786243093129457, 0.02088493642155953, 0.06185646337210926, -0.03253141585817332, -0.018789896669957784, -0.2079750446091424, -2.165995298049623, -0.01948774448369761, -0.3057236814967723, 2.6549927109183216, -0.025542781133090386, -6.4769224030292065, -8.271737567024473, 2.5302170794323473, 7.469288979844848, 0.021973159596853975, -0.9538240181175297, -5.517740906516337, 
-0.10269548463924139, -4.3669078745402885, 0.2637323142918362, 0.025555016835198785, 3.656715832724501, 0.029572632096403595, 1.369311881167988, -0.02089406607414901, 0.060124724802015805, -10.45534136891861, -0.44475530085395576, -12.294403667145403, -6.21151946380968, 1.9492299250447331, 0.1477997241847646, -0.2802856699478866, 1.9652155387201162, 0.08822053471696734, -0.037942599343232075, -0.17391204034728958, -7.3248003735005796, 0.020899319697015528, 0.029534576561417225, -0.05603506752560605, -4.708851161479245, -0.04074487668169269, -0.3387466903637204, -10.695504394063207, -1.0012877862713196, -0.06285006601852672, 2.3448244355640213, -2.2069236689476384, 0.0013094922589473867, -5.534260011551041, 9.631182554760144, 0.09096313378968546, -0.005659490800056233, 0.13556690724979603, -0.01663625489970144, 0.0525082587618968, -12.669166816094133, -0.07623698859804762, -0.0008563243324601899, -11.944031198941417, -0.7323741874105849, -0.772465444895758, 0.1862817955635805, 0.8893384836526579, 2.9327289164423718, 0.45007411756318305, -0.09753007092811572, -0.10934899283641641, -0.1176960939723557, 2.06260708401485, -0.44948399494486324, 0.9469274908402809, -0.03350117458718671, -0.09965461966144318, 0.020690936853078767, -0.8184757151589395, -0.17265362497074932, -3.7305821570275484, -0.031071954364755072, 1.3354224031302024, -1.640364438358617, 0.17154573834466902, 1.356755038450963, -0.030281385454891563, -5.794673748161877, 0.014464580614500733, 0.17977464902179463, 0.22793880639503783, -0.12760310143316644, 1.4481515623351982, 0.09292979110659161, -0.22464196832009975, -0.5203331955638646, -0.6866003005563499, 0.24029358639729992, 0.07955132132444476, -1.0938889430049556, 0.012600353323882985, -0.2448847386742372, -0.1554921645552909, -0.592444792036261, 0.3421109246327063, -0.14018123528297366, -0.15003711583926105, 0.9985430219868761, 0.1580116309131938, -0.23948598682986244, -0.08203739891948203, -1.4312456237879019, 0.4934942609332822, 0.17225487824678964, -0.21993055900305336, 1.4761915963150258, 0.3109093597596271, -0.02890913290934094, 0.945445457425862, -6.628842319641095, -0.06079677061166677, 0.14030301471863993, -2.4062395697291237, 0.4169587896352738, -0.1466832595744023, 0.06607264914769928, -0.04924693195604135, -0.37682687430812223, -10.152849494242133, 0.9521243140112929, 1.4077466751006753, -0.33668348970005724, -0.058076834985854475, -2.610267742479806, 0.17505708902838535, -0.00907313000713117, 0.04452744098017547, -2.50477977927126, 0.7582658099790365, 0.5182576465156856, -0.48998352140093004, -3.676591603664761, -10.40755248731557, 0.4826274099055041, -0.6882848459574618, 0.17864275933939666, 0.11176515445595214, 1.5321532083875127, 2.0856430726176995, 1.5266630077071248, 0.49602857275314705, -0.3363252234908778, -1.0660831255857026, 1.7609828599906123, -4.755130533271672, 0.914585184487315, 0.2536696671586611, 0.790188579405644, 2.2189461327838202, -0.6982819900421084, -0.4995120552803165, 0.022158635261963155, 0.19766293040091654, -1.3495236729968276, -0.7326274564225841, -0.4775064350242957, -0.5029193465389987, 2.6080897677579244, -0.6963250615811312, 0.8629398634770844, -6.244570885000101, -0.014315734196124197, -0.015771958860685942, 1.3369230460126715, 0.7957926739933434, -1.5936364400502754, -0.24905776557929116, -0.5897040984756217, 0.2773914813348668, -5.7824038958023145, 0.9417409512143706, -0.46610443146314395, -1.8644123152564092, 1.9689718325712136, 0.024748313136413458, 0.7049491471158262, 0.19024065288277114, 0.25464641432451174, 0.3037557846619521, -0.6591176441864235, -0.7384084722462028, 2.897682447491448, -0.2848047580868567, 3.661984797705717, 0.9781705039002837, -1.7888500932702804, -5.475402880052556, -0.7303978227028622, 0.15018092810706207, 1.6061104187444784, 1.3062840786771162, 0.0022853561519120547, -0.4310423985541263, 3.110658135654777, -0.0004999475270217602, 
-0.22040701385017059, -0.3753256203010551, -0.6575328706770946, -0.6619467707255495, -0.559047137912156, 0.0019149026562459426, 0.0794065298819362, -0.5783962588390246, 0.10381123534696357, -0.4641275195260022, 0.3310876827515763, 0.000443959860017884, 1.8163406039270455, -0.5917356652472776, 0.006111744566510424, 0.28503442963197045, 1.5210871897986564, 0.2414439816897911, 0.1515365757649434, 0.1982765573971932, -1.8038023180515594, -0.6095875350550948, 0.2670860943834512, -0.09078192528177809, -0.031155366823156783, -0.07289359807933948, -0.48926498804346963, 0.18888557368585523, -0.9025190547668771, -0.18303767829944206, -0.24180558318079193, 0.14548321197467118, 0.42211449635033915, -0.5156873851654993, -0.9397248830059652, 0.35969388160446414, -0.36191631941178315, -0.6051431773380926, 0.11780568095237953, 0.060737572800654505, -0.05269881798849241, 0.09870626746377553, -1.0270863781851745, 0.07065225915000894, -1.0651879758009741, -0.2592947457209718, 
5.801921225916722, -1.0216290437791429, 0.9253877124469909, 0.2789085918266707, -0.04869024688900225, -0.10269473892817871, -1.1780026101884573, 0.4591187987801675, -1.0572751926904473, 0.1949433266271825, 0.07984868260484035, 2.0369410542199375, -0.0016163635669528276, 0.6803094999523935, -0.09234458662077571, -3.544066212620649, 0.6298120335400341, -0.08356207187046749, -0.1553903960457248, -0.9882393882797444, 0.2857066721727932, 0.14366085638057058, 0.13942335605243272, 0.076218591892804, 1.824234115677406, -0.04857242527415906, 0.23382462341621846, -0.4087980152474273, -1.2046457496621628, 0.18696476745878954, 0.06627012721752124, 0.45810494174595817, -3.9836999215713362, 0.2657535601185259, -0.2730342000507804, -0.24084189188076977, -0.6525453501620717, 0.020696782692205318, -2.7910221280349745, 1.9725472799961281, -6.164033775951857, 0.08894368233045213, -2.683301066200208, 0.07187431096537011, -3.2418167214188145, -3.358300931717274, 0.29992356807171205, -0.2788913500143906, 0.08427757360563959, -0.20118173233417735, 0.6291364263263972, -0.5128017668185478, -0.024569379429117078, -2.4465272168033323, 0.7738964936614963, 0.14120739824507922, -0.07862576148242795, -0.3110739617318714, -2.782932288575239, 0.18068098916695874, 0.11156066029774792, -0.2805586678949794, 0.170038628002537, -0.4189257150019472, 0.02120911629543798, -0.16702576320999185, -3.005485843232819, -1.1285299465254894, 1.3324397389426963, 1.036164038032254, -4.652987245103375, -0.06510600438106451, -1.265726178894056, 4.165430655504707, -0.46339400181766166, 0.2620404177392146, -0.08333705865570806, -0.09637982720092708, 0.034255765179409536, -3.8321428273611673, -0.03830266971137064, -0.043210502286690655, 2.0394227127760445, -0.20339671038543372, 27.07391105674391, -0.011049150848251088, 0.8678455511563072, -0.12845350380051457, 0.024076374192659955, -0.02348199704550069, -3.5425206878023943, -0.7689991859027856, -12.17350937324548, 0.017767062220926277, 0.8174012256330272, 14.461529429716805, 3.677268621627384, -24.65089760005216, -2.5616207913944464, 10.522231207402925, -2.2968369752062614, 
-0.015082190741978252, 4.8720863932373675, -2.7501391765225236, -4.166556588052373, 3.6411830082013665, 0.017251469820571685, -0.08545172239850274, 11.387298633308802, -1.3123771669128672, -5.9451525265563845, 1.7796669824539748, 0.006041876052456985, 3.61481878551767, -0.005615426537815438, 3.2177925447127933, 1.6602829094211415, 2.5478400876803846, 0.03852544481821951, -1.5510114108589192, -4.447222123752155, 0.019376520760104654, 0.2101902419981272, 3.866920784519536, 11.972521281462761, -0.003106520939724078, 0.04557103152890818, -0.04348680936510618, -18.114641510995327, 0.04613782737003902, 1.7764240543262702, -6.933264220833067, 2.9243959572766727, -0.01491265643232964, -0.8575156929573353, 3.1945993593386675, -0.02438671311864482, -0.3104078207651213, -2.576448287771214, 0.774692604424402, 0.028824720252387692, -3.8977931889171202, -0.09941459540177533, -0.01506075981284738, 4.310568156354705, -0.12035608548529808, 0.007047975121778904, -1.8472966671608333, 3.007192190107773, -3.7065417494770116, -0.7582704997168221, -3.121314321221071, -0.1524196714701418, -0.23025778577470746, -1.0608516562123498, -0.03435499454402802, -6.450297137157882, 0.16980735709712746, -0.7105866515576835, -3.856803221902368, -0.005083049209002155, -0.0028153628662934693, 0.0535537500352774, 
0.5069368512564267, -0.04292070355977273, -0.01731999359104819, 1.3197988123922784, -0.3729126306755788, -0.7223936796627157, 0.054831562029335146, -0.38517145646220996, 0.029006488471633673, -0.06912194599312116, 0.030038873799039578, -0.1555513882465931, -0.7307386425665854, 1.8188558818779086, 0.04449402092239296, -0.003004013797500704, -4.688635108586666, 0.04180232931776473, 0.2394282209218659, -2.0639304756875205, 3.2759715778530776, 2.7800332416989164, 0.017299307937378217, 
-0.9966483642352344, 0.5105133773251112, 0.590869749244284, -4.25362581279974, -0.04976425223093628, -0.02963640446040472, 1.7029255094730473, 0.0857657966608869, 0.4888184407689451, 0.0944109898821992, 0.003778418875810985, -1.8847130047305996, -0.008691325540109318, -7.392950015786802, -0.1357532051239403, 0.30776625798738166, -0.009282252308942418, 0.20081402231911127, -0.9770067538551785, 0.05792336499852624, 1.4842255779967672e-07, 1.162909950350084, -0.7602204506714685, -0.011931851912464708, 0.02453851741235269, -0.005038831124007714, -5.60997694948766, -0.009882880194947319, 0.23195995014352988, 0.1446341826333608, -0.6081967465287121, 0.06173145438424754, 0.19135811206278674, -8.161494200107292, 0.012988684906041215, -0.41555758660246767, 2.062545055854244, 0.054851767415776465, -0.04600034525635266, 0.24790367807383973, -0.011306233678446631, -0.011376556014038641, -1.089716228458876, -0.03813679455550556, -0.029854180936990105, -3.0232512701663197, 0.16956856539739817, -0.46550918497883487, -0.0752259419257868, 0.473324224742818, 0.0923484942609889, -0.18901667929258537, -0.07137395002226299, 0.020483063884931596, 0.0904585527556705, 0.006980690477888629, -0.06907219751207805, -0.16497282633103794, 0.04189844539021692, 0.07348888649045193, 0.047405123191801124, -0.6762892422335938, -0.2866182056809521, -0.3913299208896035, -1.683117832986234, -0.26611245854752497, 0.6342987636221693, 0.08624939577885726, 0.0918287086565499, -0.08808114615071361, -0.009886456527546272, -0.02099484148489239, -1.53001260909884, 0.3791830874714819, -0.806755189930513, -0.24476739214945553, -0.15635971472438115, 0.004894843582547992, -0.1125738029686687, 0.41360580151489046, 0.44201088666385857, -0.1064701468814917, 0.5278059561259951, -0.01680243446568852, 0.5227623129547254, -0.5862549110061934, -0.4630356357011678, 0.40205860784246283, 0.013826976332431684, -0.14568050630994378, -0.13959491675915814, -0.21895123962690377, -0.2502256804104874, -0.08861139523163786, -0.06362157371134458, 0.3392269158020227, 0.015795207676589484, -0.36873507544378725, -1.1232648394747429, 0.11372121559663384, 0.03455657567660353, -0.13865617979932665, -0.10343174149018353, 0.02818328154248917, -0.13169985414504737, 0.20332232810566353, -0.8537551138519532, 0.08898358142208451, 0.1340171406063746, 0.036715821598988896, -0.42574781260236705, 0.02318886798367714, -0.34587910195392624, 0.1909393264493957, -1.8682837481343526, 0.014271938786485805, 0.9611623249454055, 0.013275952675719793, -0.0026122715714649303, -0.0008775945295482757, 0.31802761396836843, -0.17512494459710837, 0.12486204072467899, 0.04838144055532467, 0.04774213491542412, 0.07093192231940293, 0.27486090629517435, -0.0009486258149138393, -0.05510039346795992, -0.009732825232838138, -3.0384391227914094, -0.15440282186957432, -0.07587783786713942, -0.1344891764831999, 0.4895521513984278, 0.8572784168429592, 0.20883281343658666, 0.005797748012547288, -0.09743573914190051, 0.1286080435067305, -0.04371934533871524, -0.03129358841384544, 0.03756314595075594, 0.09024749626922102, -0.0688538444000315, -0.366255391787945, -0.04219657372305363, -0.03472637646905241, -0.7790729626065058, -1.1184168762484887, -2.3289242697665316, -0.0018450079949516862, -0.16556850446772842, 0.21170511795583735, -0.02279305680578858, -0.1221231188617864, 1.4763073062326555, -1.4382346794538847, -1.883427239752308, 0.31819665371684636, -0.7393584248708152, -0.9903160460829383, -0.080502835197656, 1.9863337081782362, -0.4562787518164271, -2.918707446794741, -0.4666906094149823, -0.0346090883426271, -0.47103131595784475, -1.4208394091112666, -1.1338851362205986, 1.2371236519964555, 0.14160076962321, -0.015814086441743314, 2.0785882245344, -0.31328601292048575, 0.10966636036403088, 0.20453772478370524, -4.400292881692579, -0.17196998752612558, 0.08976951190060589, -2.699531394859065, -0.8780058808854392, -0.21994101442913916, 0.01752712398722167, -0.5419491978674955, -0.038280316681635895, -0.1915675931416203, -0.20588948425496767, -2.5053105757248204, 0.43263126511916994, -0.011451045993837283, -0.1176861278767305, 0.00010955724845018722, -5.0924080442988355, -0.05382479059758083, -0.540698367899239, -0.12046916665028107, -1.8062718060057747, -0.006623016430587825, -0.08537826407541615, -0.09043514833054189, -21.64761705595264, -1.199781776851239, -1.7310188778878768, 0.3044891533281131, 0.3763345536789444, -0.35828577352943114, -0.07673836412963198, 0.04254445554741437, -0.18937486260968228, -0.04271228944455885, -0.3734702777213599, -3.072106679056958, 0.10681655329373996, 0.14372761209997265, 0.03634590254369385, 1.0534098030195267, 0.7848305273840168, -1.3033517037634255, -0.845302266661129, -0.17639168271029604, -0.11262310016114441, 1.5709824096424825, -0.5062504658214593, 0.6324013178911798, 0.11137128802798912, 0.5041339875357309, -0.5431178564635673, -0.776230801915645, 0.470073580774482, 0.08397690425874771, -1.916062980991967, -0.8118992763187549, 1.7571050207302363, 0.4995224686663988, 0.3473370136556113, -0.01665643444853515, -0.12178724614011571, -0.04446945826149573, -1.7514741952898305, -0.19387365457298955, -0.5395282436745319, 0.1351558147622356, -0.09973044748268478, 2.9737635938942333, -0.03432453940862956, 1.1973158729509947, 0.3921335275265605, -2.476125130056708, -2.4223675445863506, -0.18121377048590404, -0.9436407513884362, 0.8084107719903955, 1.2032829830410279, -3.799305479984028, 0.45475297793316827, -0.026704234020527906, 2.21742703056195, -0.11055168052180875, 1.0925343907086145, 1.5400965612356359, 0.08247764188176632, -0.27383628185384623, -0.1391151355122915, -1.0836538275349028, -0.6175314756336334, 1.5900551626779986, 0.00906345867817393, 0.6138814261628909, 1.9654474396173214, -0.3923884815295331, 0.12903932576725197, 2.342506847556585, 0.6946842795219794, 0.17991084122819956, 0.16387605886906442, -0.04666505906167018, 0.5318579149930542, 0.4565073665078003, 0.8827499571046786, 0.7840121932575386, 1.8092618847294943, 0.09320489201636661, 0.8668932504586238, -0.43262926666080403, 0.008769632095049218, 0.2633876185220174, 1.1139493549539736, -1.336569572040439, -0.03891635789752712, 0.4398839414467517, 
-0.008677832132796937, -0.017159825069263945, -0.7200039296548297, -0.03815500707550257, 0.4847502177188545, -0.5227142854507179, -0.10914859377298569, 0.7530388949114695, -0.44671861693132087, 0.3629037004539697, 0.7271756449116253, 1.119223600170308, -0.03745392651677548, 0.29679192074866734, 0.6684305201685561, 0.6149803230889148, -0.41696104266389966, -0.35135486252811177, -0.6592562826944359, 0.6340949323981704, -0.0010947185210525845, -1.3066866619985404, -0.908304868794211, 
-0.36967603481197386, -0.02357981101438611, -0.8705848817069892, -1.8049706056629162, 0.13065220936481836, 0.8332927629565461, 0.3544922334967211, -0.03300188154324246, 0.001593603979522129, -0.9038079975236712, 0.505319747624867, 1.0119491764719868, -0.17055407754774876, -0.7005099782416693, -0.6434311232194716, -0.9188069602588556, 0.4393739299907935, -0.27043749516654714, -1.4408528514925507, -4.524948466419943, -0.08817697210712794, 0.2006357095482798, -1.2075376074589315, 0.5308939525504002, 0.4080173734949568, 0.13419327112021762, -1.040951207868659, 0.3501577885710532, -0.11330153988662062, -0.4986787340258445, 0.134504302936719, -0.4941967691963427, -0.23452774246472075, 0.04098410581636358, -0.33728964679198725, -0.024061361545165028, -0.3254206961361774, 0.020508097297252448, -0.19627057238626833, -0.7432652107053457, 0.002776969595068124, -0.7747974855876478, -1.0412404424534287, 2.082831953880799, -0.9200327973247653, -4.805296176102495, -0.007771889992156389, 2.329462031138931, -0.2745667192155352, 0.3013707384576563, 0.3890183162662595, 0.1917733243123081, 0.18721166749168106, -2.316575561777128, 0.37029630909397326, -0.05503480576313535, -1.757256069375046, -0.7899629640885024, -2.174145526749541, 0.09436910740163995, -0.14957689214247694, 0.2701063567757416, -1.0684992695296955, -3.1131608721847215, -0.8751831557184673, -0.7322788122533197, -0.3630658295000302, 0.838022401383455, 1.5836212040274944, 0.058594689545643064, -0.7744282304022647, 1.083128049373073, -0.8117068710864146, 2.282285753973241, -0.0809529301376486, -1.0919714017304898, -0.762103674877693, -0.4114328378520611, 0.019739975033687074, -0.13772826502137292, -0.013082068451495843, 0.03403536260245943, -1.2754945765501446, 0.00767615948505096, -0.001430137658864794, 0.9915278945660191, -1.1951551967403162, -2.320265303641804, 0.017042924204993426, 1.2949296150596865, -0.10864005414331501, 0.022403488404464156, 0.03555237129440769, -0.8288130839210766, 0.4595677394585209, -1.12839129119493, 0.454198202471801, -0.404996299990799, -4.2689476882919735, 0.09443199751999742, -1.6057666080466788, -0.9476659358483772, -3.5393289443657068, 2.0522370749071968, -0.02181801940839989, -0.9216127879508065, -0.1825767194925021, -1.0020339420027682, -0.09018758634883284, -0.005380252817026587, 0.08009639871835361, -1.6053266576844578, -0.8493694443759523, -0.08811117371399746, 1.9161853717648967, -0.05767318732701199, -0.955659346401319, -0.1642583506168549, 2.0058445281857473, 1.9222613495210226, -2.216541898118635, 0.003152106669688237, 1.6938954560340562, 0.2892328147711254, 0.028242654819521817, -0.017099541832465093, 2.141251740325714, 0.00412866161502734, -0.14354694461743378, -0.21581531229292494, -0.02747626737509279, 2.8477110891029582, -0.01900594569389824, 1.4340013624809842, -0.35576674172602907, 1.0055086750626288, -0.012402966502044421, 4.608940310044915, -0.014830164429881165, 0.18827587550724445, -0.09517579626454165, -2.063503287977227, 0.8461904262080964, -0.017517331617163734, 0.1710579164062267, -0.11398641386892194, -0.1416131769187956, 
0.04374099863912306, -0.09831225421158507, 0.17685424584443865, -0.6175335574967846, -0.014753256530028125, 1.0097621351805088, 0.7109146915904461, 0.299777834829996, 1.0518487539332284, 0.10977915944310013, -0.957275040199363, 0.015684981295748912, 0.25293273779271885, -0.08094019439309363, 0.20373279527716193, -0.06363804750552049, -0.03707209976874992, -0.02511417794778481, -0.043387119535246654, 0.786695557207203, 0.009538079559725077, 0.019247174333784756, 0.3598207616156799, 
-1.8227558607961, 1.1375655266956102, -0.013727743225018685, 0.3206978752807714, -0.028490828154566827, -0.001431108912356649, -4.101534756563822, -1.779649967300088, 0.38038144464096746, 0.2481365639826265, -0.08458479498986549, -0.28094288294308, 2.571085028653698, 0.007438773651472275, 0.2503764946795002, 0.1264125297755072, 2.5988210263303273, 0.01117388928217622, 0.004457318435356683, -0.19700807804984777, -0.12674549320277606, 1.1684482437737884, 0.004285923146468917, -0.02925021730382582, 0.015016817962838047, -0.33760505220185166, 0.10318309560465622, 0.3003753994589715, -0.5590433788488696, -0.24771940167959627, 0.723985224411166, 
0.0011623864033367681, 0.47695305095597007, -0.5845341228211538, -0.15552879332290104, -0.00752222111920986, 1.0404856780307767, 0.06488279410754672, -0.029238216450742982, 0.02841369447045139, 1.0919851955622875, 0.24061955245437616, -0.02189635581353855, -0.004607759111912912, -0.016312050234155606, -0.9630027734148854, 
-0.016276520954832563, 1.031830635939997, -0.2853028344113824, -0.5572326272861048, -0.004628243304672708, -0.006105502180133726, -0.3337026161568133, -0.05543758690379974, 0.05821618550210417, 0.8680042548514137, 0.2722762768796585, 0.00843567420555047, 0.05905739234056995, 0.015193926068143071, -0.024212319375571667, 0.4781287090057731, -0.09369975600429115, 0.0123887124710933, 0.20459439287819237, 0.00872025713890423, 0.6414839832779364, -0.7740603785471762, -0.5818345567607537, -0.4998384193209233, -1.4892026289017224, -0.007775982743027043, 0.037948223731383735, -0.7598327977941324, -0.3204946962752473, 0.12676952780610407, -0.6762465701675353, -3.2655876307155083, -0.005133809301391068, -0.04078953148945175, -1.3192632266969184, -0.0451285964953172, -3.62962855923179, 0.06675125407996063, -1.8884021020761708, -3.589139671157298, -0.03413298643069851, -0.19458539807904174, -0.044119533589373816, -0.016034443820217348, -0.030220375995725135, -1.2963357010595047, -0.007868676730382163, 1.4970227799612275, 0.0011844174933912655, -0.2613457634569585, 4.225665075327356, -2.610074716192706, -3.5764645395378096, -0.3896976985329985, 0.3672766257941902, 5.05061000252924, -0.02018326158832906, -0.4409875312502436, 0.5788769383348011, 1.069169351056928, -0.16109987914400392, 0.08921990286413006, -0.1832293113876915, 0.5806478782899092, -0.09751082224173047, 0.4238540427353854, 0.10610398024383016, 0.07843005305781503, -0.45044953126603104, -0.09070999030149096, -1.045807567920619, -2.380058902420288, 2.4339699507587085, 0.003973954715192463, 0.39611858136322553, 0.010467953450501, 0.0217625038475191, -0.6129795645275706, -1.4991400454916146, -3.450774308198419, -0.14553611394018873, -0.49747085978739136, 0.011821079945605106, 3.471599007238197, -0.1221325123639545, 0.045960995808400185, -0.20226561125535625, 0.4603685537560267, -0.02561180538133101, -6.8982489851396025, -0.26979405746376983, -0.05696312208016252, -0.2215823726371724, 1.8070441828504045, 0.6672761405868357, 0.10378778536399125, 0.10645792182721436, -3.3954331037552237, 0.009651821392232307, 0.8673606453538696, -0.04629611394442179, -3.5232721439057997, 0.00227356580671767, 0.15227497390974065, -0.2886758851838067, 0.2172527649896523, -0.7155560581369684, 1.1840368419968925, -0.7387909204426109, 1.957158241769669, -0.08052089003592755, 0.33265317884415424, -0.4674378776487913, 2.3899722959971257, -0.6296845029103721, 0.15855008668160053, -0.3661818194253996, 0.23447404266195093, -1.1485654321472083, -0.12543902266081375, -0.5112509840356267, -2.2850965173126383, -1.182107392434098, -0.8973114951588741, 0.11906312135863573, 0.8443121515085963, -0.21754761892384522, -0.09851936564461994, -0.11960046141138747, -1.7718359772900243, -0.3213419586626003, -0.9179890590616537, 0.7611912675479644, -0.2886588687894829, 1.9783356280705107, -0.2636862460625906, -2.0425206246590903, -0.778160042466709, 3.68897748806954, 1.7829125603344238, -0.033249079715623964, 2.31879316335376, -0.4124226856109061, -0.9941126056913845, -11.785350484280784, -0.4530673145172557, -0.1374560331679291, 6.308836845753817, 0.9745810593955895, 0.8443585769189781, 1.4534937617538688, -0.09450928064272546, -0.2809799867464591, -0.017223047718687212, -0.197048578359869, 2.0215976716184656, 1.5039202027118819, 0.30307585552733585, -1.2377464632516038, 0.09292031215563412, -0.6338132244824557, -0.073548730475828, 8.697441648630587, -1.474924515736788, -0.022400283935283483, -0.14119369414406435, -0.6913965044633557, 6.714486527964851, -0.21926079275158372, 0.5179702676202993, 0.019880526037042046, -0.4127252405948525, -0.13394485458863414, 0.08392314726516759, 0.2527423074548807, -0.005579287331499927, 0.46882556681897825, -0.11336595965657636, 0.47880369729440986, -0.33642863008692103, -0.6922675992934373, -0.07825119691253235, -0.11205474693060324, -0.05657030486908354, 0.0008640165715405601, -0.010870403677152751, -1.4108501756577994, -0.5536276588658424, 0.5034628287622098, 0.5615283569210447, -0.6688152173120017, -0.19181040601985444, -0.020110703854385292, -0.09617827230397324, 1.201613714984127, -1.9869638956069906, 0.7894938788330563, -0.5067300217504567, 1.7069009186547675, 0.2605058504678439, 0.125141064881646, -0.5118235975528549, -2.694636057790653, 1.312874146073348, 0.03429988631981473, 0.6735989172257177, -2.0488089917913967, -4.238057730928574, 0.26108586125758393, 0.8639007454972258, 0.043431521634651915, 0.03759731514102782, -0.33922079591310705, 1.4060217312114531, -0.18297356129886566, -0.34525532838014783, -0.983522595134545, 0.25439925119736984, -1.2102497631845068, 1.1600647425095758, -1.2991233547145669, -0.8372341664305338, 2.126561655803499, -1.9854832591958456, -0.6047523883696186, 0.1033924719882009, 0.05382484661862463, -0.9418070477071439, 0.7211858608749537, -0.32091017649970865, 0.09775244171153474, 3.962213512639366, 
-0.8026322611926133, -1.3579950179034483, -1.2311076981219742, 0.021963792001145066, -0.16981659434458152, -0.3826514577147009, 0.1294110092539853, 2.584748685108181, -2.011074458985277, -0.38884582382010535, -1.0802758428917443, -1.1880427901952402, -0.10405513872291294, -0.13387431628022028, 0.23711871421583908, 0.7877520009341339, -0.18896285441421412, -0.0234202049960075, -0.6377159978891669, -1.4981838322539005, -0.00659016250880029, 0.8292562538787891, 0.3491246418373919, -1.6501945674341414, 0.2633295407210454, 2.2918337729154814, 0.16175305210433066, -0.03170732845374857, 0.8705181054013451, -3.6876817387312286, 0.6173881165374411, 
-0.26116428740128583, -0.036594929893567496, 0.12928967909424927, 0.9121032885524443, 0.4330820110250784, 0.051862294085275096, 0.9183732257006056, 0.43467580011680695, -0.8042672640400426, -0.2175231465137415]
    diff3 = [0.2181657041425069, -0.20583241100981553, -0.40123779011178584, -0.24045054522169806, -0.8825931872586139, 0.008994871724617326, 0.41735396285683635, -0.0853869463990975, -0.012581631336786359, 0.38407641125541403, 0.15565944406252186, -0.09220875759631753, -0.20325656695551686, -0.7174242164345728, 0.002171057327522874, 
-0.1005640377911945, -0.9694735777887757, -0.2932913231311076, 5.652431899207585, 0.7583227219216724, 0.5963992691046442, 0.15342832659251116, 0.05540956903671912, -0.02283123692160416, 0.855221708290145, 0.5389716054372968, 1.0560657748559237, -0.482356188492453, 0.6310001614660479, -1.9946250229945974, -0.009703399021880443, 0.3740862898056747, -1.6942293798420707, 0.6668734549010367, -1.71842903838116, 0.020844365521284658, 0.9218910252102859, -0.2987142783974335, -0.6582005365886232, -0.38773503183493574, -0.1829694714932657, -0.010250900753440817, -0.3652693804111351, -0.04399483795322112, -0.14058643469269327, 0.005902767909780948, -0.009696044735804321, -0.6659011837964215, 0.10443889268707096, -1.4175236079187528, -0.6744200543015495, -0.3179601337026341, -0.016862499134099096, -0.5854920288664545, -0.8705311746199129, -3.5548272105279537, -0.08229556560524998, -1.7728009333664403, -1.1135650026022716, -0.08645913337494449, 0.008240380067810804, 0.023681395333134958, 0.6968879469637841, -0.12126796377184945, -1.3317802142532358, -0.4288863020120317, -0.7656263218182531, -0.06690970551822772, 1.7079419595891636, -0.2893417948635886, 0.031028693793800244, -0.062390784794303045, -0.8136863897902344, -1.404424069452645, -0.014019939366910705, 0.03715764586774384, -0.018992314786139275, -0.015643281392712538, -0.30950890972640366, -0.05377831407625422, 0.06712890233534097, 0.4006869822584349, 0.5101703744617367, 0.492398970120135, 
0.4384343624315221, -0.03700767754484957, -0.24492273322582037, 0.4930338357843027, 0.19199584760013977, -1.615820781720572, -1.0847066026855856, -1.2924504185021561, -0.9736230426141503, -1.3038878865096137, -0.7883832828327826, 0.14271066976478153, 0.07561726487975307, -2.768806818871596, 1.5993111519718468, 0.25928463011219094, 0.7005956848894215, -3.192352676332291, 1.433932486736012, -0.7377179361964039, -0.7516933965360124, 0.2417624738622095, -0.16797200456100114, 0.03596874155223162, -0.22878981946388421, 0.21809838768774625, -0.3829614995878359, 0.5275491444304095, 0.2413796862887665, -0.3011237171359653, 0.14282063221257602, 0.6479080644555779, 0.06188261002240836, -4.360943055135408, 0.6771935670071798, -0.02626218907985134, 0.36341079031922163, 0.6783663982610904, -0.9288292613572651, 0.2209666341935872, -0.3780967314726311, -1.2889795713651324, 1.136887045941819, -2.622719449824274, -0.17267825409987836, 0.9233599730183641, 1.3723562233461095, -0.5322199307712054, 1.4963826115337682, 0.4359101056841723, 0.11299232733033193, 0.12683855682590206, 0.01239100896107459, 2.55994693911785, -0.20672777322943148, 0.0037874014982293147, -0.08518788842454228, -3.4728957280578356, -2.2031969596272347, -1.0279626945345726, -14.430309539908691, 0.0054963647670476234, 0.8905752608300332, -1.813582880656277, -3.6360746875871826, -0.6657577354604598, -1.4116672642951755, -0.004640669074980508, 3.4589270804445107, -0.35095242340420896, -20.399571277601993, 0.5647304349332387, -1.5472100450077164, 0.1254422621947242, -0.1697840111255573, 0.30392424120749695, -1.0392161459159084, 1.434101953178903, -0.45691056760878723, -0.5261833845631898, 1.9016470898976934, -0.38524168990441865, 1.5023396777832971, 1.899204159878252, -0.37685255905770987, 0.07364478805713759, 1.0125321423766493, -0.35661424189238033, -0.032678859284374084, -0.03742259367899692, -0.2990488747615956, 0.4859519732569595, -0.415554134098727, -0.00116557911944426, 1.3433306389220192, 0.056576528019000705, -0.32820704366888265, -0.9545982623735014, 0.3563885692918234, 1.2869045780120203, -0.00025363115349819054, 
-0.6645353582518396, 0.693952192464323, 0.692540423345875, -0.20983513925123987, 0.2478459394025947, 0.1957373764882533, -1.3044632294141039, -0.9536651064351247, -1.0699582105103502, -0.5343288508072987, 0.28494134106882996, -1.0301396296074188, -0.46875326497378467, -0.08218483410441024, 0.7434310236167221, -0.34651029965220914, 0.6933575559839795, -0.36702143019815026, -0.13535089913566622, 0.526245279947716, -0.11296383377850816, -1.0876178642033594, -1.0461856527468285, -0.5472653642587275, -0.07425605400177915, 0.2800527566821245, -0.06661299209500271, -0.6166317171051077, 0.0804751809031643, -0.4776702670899695, -0.20372338754387442, 1.5865435005416018, -0.7272622720142436, 0.582616411702432, 0.3767707172698067, 0.22964740400725958, -0.13739663187300266, -0.053074201445440394, -0.12351663700709281, -0.026746881301868086, -1.1603416842485217, -0.7835967899095237, -1.1987693458104403, 0.20224727660414032, -1.4642134271946006, 0.5341134154890739, -0.11588639941729184, -0.33391461225949826, -0.17283858197053803, -2.134804727103635, 0.2776546832803888, -1.2571357664637048, -0.5163135235565193, -1.2312715075248022, 
-0.11686878793092603, 0.3217375151110531, -0.43505414877932935, 0.061770866053862505, -0.3594499191737839, -0.01972506930512452, 0.19816110030696166, 0.12017018378902833, 0.9917607576811207, 0.17181001982838495, 0.01652337095913481, -0.11145993598522352, -0.6564388443167957, -0.5484199593896335, 0.2115804975309601, -0.28587470647408963, -0.18405981229560098, -0.21295133535609523, 1.1863672274247747, 0.19262143485843808, -1.216660231578004, -0.3556074248132006, 0.3593010494301012, 1.195920745240194, 0.021984971776873863, 0.0028159067023239004, -6.796829191911343, -0.02772553594989091, 0.19643566375441424, -0.20632840068276437, 0.07437902354561743, -1.3507748903799381, -1.0425710615952397, -0.40450329431766363, 1.0970084718621962, 0.03319046713480844, -0.008658317113138025, -0.3677280991438465, -1.2964530499645264, -0.5136925248761344, 0.21507287744790915, -0.9021793800196605, 0.015377644163478976, 1.6815934808702124, 0.5356021164733136, -0.033091478042749145, -1.401402760750912, -1.795875456090954, 1.197450281987841, -0.6419004038232146, 0.14284550551157338, -3.4961913880620443, 0.5327491838777263, 0.14052854123669078, 1.736589711105374, -0.5692579703303693, -7.300756626346725, 1.2455632798200043, 2.0953033369371923, 1.375504118692504, 0.17085128954676065, 0.7684216901027199, 
1.050579627023538, 0.10533684170701463, 2.504040750901538, -0.13972165571759376, 0.26991230953417755, -0.8279430382193169, 0.061448499151190106, 0.18694240301781662, -0.06965479737888813, -1.8704359475953538, -0.2958315904543056, -0.40414146556097563, 0.3781863461322885, 0.9141252330878302, 1.4387888770062816, 0.3129289237217989, 0.15169525291072716, -0.43595552484396194, -0.0008354369253851246, 1.063323280131982, -0.1234832658689129, -11.749500216270015, -0.1250012233455493, -0.2645962265386288, -0.6263277410789527, -0.9395726622167544, -0.02815494882898406, 1.6437837834488889, -3.89346957230093, 0.1078347480197408, 0.15378490930369537, 0.6065501340410506, -0.12600096143148676, 0.24946733036856017, -7.522655291717115, -1.2066678617716775, 0.23263810728184353, -0.696685115281241, 0.23451810440613485, -0.3046780597258305, -0.0874734268826387, -0.5479526682054541, 0.33465817682392185, 0.27790968962716533, -0.29840514854363676, 0.06300817233372413, 0.0034345101318393745, 0.1661620461679405, -0.7104788056506095, 0.006657107878673685, -0.03949939846636141, -0.8464492652947797, -0.265547484004955, -0.019621203890736183, 0.20285344758529078, 0.17246219164847787, 0.04378088454510021, 0.08116255486882551, -0.03143573376301134, 0.12067898261948073, -0.0514473574967127, -0.08327595505426189, -0.20048711996453505, -0.29779422771707686, 0.42180848942176397, -0.05161697822209277, 0.5059691486903262, -0.39629173721091604, -1.171148122968006, 0.10652845948656875, 0.011303084453587076, -0.45106440856723395, -0.12377861972676385, -0.3823224399739331, 0.4138032769867408, 0.26391446145497355, 0.028575500990847047, -0.24345081246667633, 0.0986871553732982, 0.6328152947972399, 0.48172405698792886, 0.09215958232572774, 0.2195320723254497, -0.7828639596604319, -0.12945650222850702, 0.7290618098202231, 1.7962503715296236, 0.1701428860322025, 0.4458466285335163, 1.224143275611027, 0.1080634678347181, -0.022221475182924166, -0.04280296588814281, 0.16663753838595596, 0.03975441248005751, 0.010546748698622821, -0.05450621674735601, 0.7016553223016047, -0.05226485660346469, -0.31091158685414655, -0.07032224251616981, -1.010158159331553, -0.04050690417108882, 0.4269710632725605, 0.5085447983691154, 0.00901921843652076, -0.20266556117572065, 0.44006513490548116, 0.30242851555155426, -0.0013317231125284934, 0.09452344444848393, -0.006802931194300754, 0.05215550265985058, 0.34237673897095533, -0.4079017364559263, 0.024253387491178557, -1.3162584091691514, -0.29785507298970515, -0.07846644157439187, 0.13309037525881706, -0.14116486336678236, 0.8161717173723488, -0.16674913205963549, -0.15054695013677133, 0.04672078309150152, -0.15382879957633833, 0.5507526960560725, -0.38849465944637984, 0.12271208058130867, 0.1396135288556124, 0.09924328860096665, -0.17086928250515143, -0.5627202033572161, -0.08157471482441991, -0.5749414045044077, -0.061623542231430406, 0.4360368243609756, -1.228183802289422, 1.74851339795962, 1.1637465286766897, 0.1404905502804752, -2.354363397190916, 0.017621426239962545, 0.0784967571360724, 0.19524080201679794, 0.14099692556743548, 1.84088712884909, -0.07838697242409154, -0.5282383334222374, -0.36426341963594666, -0.2647094175581657, 0.34094432266487473, -0.3505443706021083, -1.2439808581256955, 0.017823421111991422, -0.17626556011254024, -0.04921288836970916, -0.4373517613307669, 0.28918382819533406, -0.07010483824948466, -0.31699785390887314, 0.45250772724139665, 0.06907447901156161, -0.2045125993496768, -0.10050134974746072, -1.4840869946526922, 0.5397617241931485, 0.3301799189426262, -0.06848732174088923, 1.2420813054756223, 0.27336941883263677, -0.03344938170824463, 0.49195930587038106, -5.226755774691192, -0.04650654312661828, 0.0744033194904219, -1.7075523622708033, -0.5111053591399468, 0.24077639505348714, 0.47187699604708655, -0.037130058175328884, 0.3815921881469819, -0.4570221652424138, 0.6425697516047748, -0.0025388950910709696, -0.18159045899455606, -0.06520252564509121, -0.5483966387955093, 0.006559361688552201, -0.010713428457137297, 0.05248329313435818, -1.316172243483237, 0.5180484242825685, 0.15057646001398695, -0.23306566784838623, -0.5209508511416203, -0.1720776462481055, 0.5154099612926686, -0.6529717452582844, 0.16259502623653788, 0.11620667687078878, 0.015696934880473634, 0.33808190657345705, 0.036921816615425485, 0.08122851300535672, -0.4281960895708252, -0.11910232774908636, 0.282496478460331, -0.017128367786579446, 0.13499162430940004, -0.17512336625524583, 0.03913651372396032, 0.2712226722895963, 0.007397262114242409, -0.03120035594961479, -0.005329161058011067, -0.23780294028747306, -0.011229741185843523, -0.025431389578184138, 0.10265632737036157, -0.042219408879031306, -0.23137707452261225, 0.02722845194095669, 0.2019952522561148, -0.028190746568895975, -0.01883272906678002, -0.012188573751750198, 0.11151811984523619, 0.25397743473732604, -0.04968235978236635, 0.004210725562781903, 0.11427577690331603, 0.19616271750813752, -0.020738213506952263, -0.184748295171687, -0.15785560059316595, -0.3363081227242759, -0.452175390436949, 0.01974794754508835, -0.16243978912036994, -0.053796399580946286, -0.009840994509520584, 0.26060504665072415, 0.012568378846175676, -0.013968231696168232, 0.08472153360839485, -0.10258965063402314, -0.09148709926802212, 0.119471486764688, 0.02497067420706145, 0.0037471495833898416, 0.015046871595906453, 0.15122116656188211, 0.026006658639076363, 0.3732516396900323, 0.0008790137525096497, 0.026505827929451442, -0.060728545630901465, -0.0008844777150400773, 0.00524857444686333, -0.26369769612880845, -0.08738514672027975, 0.03531850654702495, -0.019483325644600313, -0.0038990045701119413, 0.2699118723265883, 0.004593374990651711, 0.14219078225009696, -0.05532114515204967, 0.06938272592908845, 0.0036145090631514165, -0.030945680064306202, 0.062299963573330785, 0.0009562115895960233, -0.01172302606239839, -0.24908172979643695, -0.028124532827295212, 0.05666265270643578, -0.018301476240615955, -0.01832891423692473, 0.0011572873630676384, 0.10114827629405454, -0.017334208761841552, 0.012119825802074047, -0.12381941101133975, -0.3786941915397257, -0.17033243060078362, -0.3910718206068253, -0.14775976221837794, 0.1212059306374016, -0.01897021288826295, -0.7919165215428166, 0.09368249162058362, 0.26821554666483394, 0.41230210286441604, -0.06662709682208146, -0.15367184554156665, 0.00986487715108808, 0.06024138956848191, -0.09523623652698632, 0.1334906860703029, -0.27354467232503055, 0.17430814218295865, -0.46427378505420336, -0.4942777755190093, -1.259047833622617, 0.07233565005952869, 0.7767801346257883, 0.13985147982832302, 0.03705716307764817, 0.034736049954823045, -0.047705584141404245, -0.04473106846006658, 0.42506575855249196, 0.26922347199248264, -0.3141792792970861, -0.9921456447413917, 0.15435114918702908, -0.1836408321796057, -0.19422113531273055, -0.24851784620744866, -0.8121612238671645, -0.038224073672068926, 0.17681701186184284, -0.02435107997347785, 0.338631954072909, -0.014173331527409516, -0.17740976029275402, 0.14146946896119772, 0.22515572613107082, -0.03951040440142606, 0.24481941977384736, -0.5649861353328873, -0.5824823178467184, 0.36210984425435555, 0.35670090367155183, -0.15844607204603278, 0.6513256831268492, 0.3133340131142788, -0.09698064903599857, 0.30940210172456517, -0.24372250021207975, -0.023244120912501387, -0.35190327217080153, -0.8579549904470198, -0.6827062161587065, 0.132127248945622, -0.5925071869305327, 0.048844701048018635, 1.2141805082508483, -1.4042569123994895, 0.7609810879057335, -0.20040588479516686, 0.23309035416362178, -0.06754620302074699, 0.3536452121664979, -0.04511327524804898, 0.030822110493815558, 0.029996997273514125, 0.08285482748669892, -0.16943693030813023, -0.4784330471916647, -0.1882851542489803, -0.2894602773427408, 0.12277391514137292, 0.5847400325627348, 0.1738770696934111, 0.006998550978323692, -0.08060630500582988, -0.21494282992865976, -0.10203260828273386, -2.148762061237491, 0.25475642090989936, -0.11664411203838654, 0.26707128857374585, -0.595368575640947, -0.046126776151602655, 0.049788388966462094, 0.7625500255784488, 0.24077039425314695, 0.42590578979138627, 0.310761238601998, -0.07604783140433824, -0.6850513747312306, -2.0803060048002067, 0.07292443160820028, -0.02025871995299866, -0.6290657625821154, 0.965280197829884, 4.716431024096195, 1.7749253623972265, 0.7723839638767629, 0.2676180290282417, 0.016512161705669826, -0.01881566218791164, -1.3877788488089209, 1.8311331127561914, -2.3786612501065747, -0.09894468482019647, 0.8338966353612349, 1.269045494058858, 1.5479661922706995, -5.411565584719327, -0.20677573519904513, 
-0.09885423306806729, -2.1117007112541586, -0.009780798952697012, 2.3363867983880766, -0.31421433799975773, -2.7308052496114215, 1.0257179728209707, -0.008965883148270848, -0.0513077607241641, 3.12081327331299, 0.10367999907188619, -0.680485258714782, 0.2828202521556733, -0.0017369212189706218, 1.1848634532814515, 0.3054198897123186, 0.5251335414116625, -0.7695644460275588, 0.7265288520547557, 0.04177090708672182, -0.37041585006942057, -0.1275370050792759, 0.024571213091405752, 0.2028449696429675, 1.529647207403329, 0.6703520109442138, -0.008292331989736113, 0.05571673355727569, -0.03508342314839297, -1.2058523294998622, 0.05134888426660211, 0.9317138747444034, -3.2915658697505705, 0.9263472882228143, -0.015191524433035397, 1.2860982666428669, -0.1625164376612389, -0.01867165810985938, 0.4610225373639736, -1.2553473498176686, 0.9208856580831224, 0.022879062467545452, -2.2242432572991646, -0.06307419842767104, 0.0035557519294187045, 0.88874137648817, -0.12424129405356865, 0.3087696208976425, -0.3962572413382759, 0.5868540330228456, -2.0696071186637823, 0.01575561515172552, 0.11652477852008758, -0.09530173995079849, -0.1849685287880618, -0.2152706573486185, -0.03359457263938381, 0.0946229533357652, 0.07153650904167108, 0.05231293758301092, 0.07071362920930113, 0.0063230947932133574, -0.0014050714739717307, 0.018898877566527972, 0.14533014860892735, -0.04528953958240578, -0.009557196421138059, 0.11151091100275323, -0.198674311094706, -0.5107566894831947, 0.13981746724685706, -0.21324453363196483, 0.012578105521040328, -0.05792193746368568, 0.0736266032338726, -0.16992042434220167, -0.14557034243148337, -0.016669127876042467, 0.11593597948307277, -0.03803861009917853, -0.1844881634397879, 0.04956512648767486, -0.2226159558507419, -0.18747624692923637, -0.0028571792796725504, -0.46382923303967516, -0.01041009653611269, -0.12726090397045908, -0.09224686201407195, 0.17326783809480872, 0.15124437311389016, -0.04373151418025145, -0.0304877683132041, 0.1114983672737111, -0.07815701767775352, 0.0993954393102996, 0.16786654055457717, 0.004261362332606211, 0.09220873288582965, -0.0218194904686122, -0.012616797193679474, 0.10715252514705043, 0.18139459905248145, -0.011019118825718266, -0.019457504261556124, -0.14683228463163545, 0.04921505951021388, -9.970929527369776e-05, -0.17265203119407602, -0.5454011313087435, -0.00927114865473655, 0.0316581533601088, 0.017230479328574688, 0.16089464054905278, -0.00145054137509959, 0.1123263105213077, 0.12404611868494442, -0.2592645027147569, 0.05734024587792064, 0.26325181692716626, 0.002711433971740007, -0.01367035883153278, -0.13463046428252312, -0.422153487821852, -0.10531844527076473, -0.04051862395849426, -0.03289349203606662, -0.011700315153365892, -0.006462669261363629, 0.07283374044170188, -0.03205498817046504, -0.022334371112876283, -0.14312098424042574, 0.15627276894952047, -0.08359769866343925, -0.06967344399545539, 0.2781883836362766, 0.06231929349412013, -0.14716846449123722, -0.10871707099043704, -0.0028648358399649965, 0.010139831184837078, 0.12285732898911306, 0.16820241114013612, 
-0.0027144709746949047, -0.005866404055474561, 0.00344668579863594, -0.06220063502067319, -0.5901057050714371, -0.21455858369359504, -0.33163799171824593, -0.7045245114100283, -0.24110567099696567, -0.1004628283150737, 0.13761061562291133, -0.1957913995192797, -0.04892290968082591, -0.07274321102750747, -0.04004269508578062, -1.0426465141432644, 0.82681903208357, -0.4329232806937142, -0.2668811197874632, -0.12334715898548865, -0.0659193322460041, -0.06993200281225853, 0.2022951069310821, 0.2523915497747353, -0.21765505834677867, -0.11296607490859145, -0.02377538310566507, 0.3534165095217503, -0.23211520794272644, -0.47083134655662917, 0.4205760520554094, -0.0237403261749094, -0.12045591249822962, 0.04347161785968012, 0.26473737155006205, -0.1961085872271724, 0.006860637332501085, -0.14164844139730093, 0.2517184217183086, -0.021915832903118826, -0.12689477528248716, -0.3825573133453588, 0.13204089685103781, -0.04546155515986783, -0.26410366484626024, -0.093907073141267, 0.023850760851921393, -0.11960020409608774, 0.18347297852425015, -0.6613735768281472, 0.10118922695266974, -0.3711794063312013, 0.03746702126484536, -0.16534832264492394, -0.0071806057185455074, 0.030819164286931766, 0.04125140753676959, -1.0128804796267232, -0.05492108723083078, 0.6147392353603465, -0.005709535947886479, 0.005176966451827525, -0.13254407884232222, -0.07241460858958959, -0.18375648690659574, -0.9109538762420826, 0.016275623561057273, 0.04686621522744616, 0.03513626440212647, 0.2328300735198212, -0.009867842850816544, -0.087144433575439, -0.06972424120964149, -1.2535374892491973, -0.12015296783098961, -0.21094615644565806, 0.040817997992036226, 0.49270856510172223, 0.9044892297534304, 0.2018118136140572, -0.04079719271924631, 0.022206306122754427, 0.1592711357503731, -0.19705844846376408, 0.29644509786885465, 0.18906070827841148, 0.10836747270792557, -0.3197201558227576, 0.0674027743667196, -0.12806594877761057, -0.08132131720084601, -0.3918436651478743, 0.04046161402858672, -2.427838626295056, 0.7763434934983948, -0.15545191168149586, 0.3696711504797676, -0.18555300238877948, -0.11999027783448923, 0.616593331106202, -1.341988611681174, -0.7836682504993036, -0.2559444976406553, -0.7562618187093051, -1.1705434587225056, 0.08606905852206381, 2.1774777152095197, -0.277215396449094, -2.4204273522201873, -2.745620801251107, -0.01551519161015591, -0.5742130812423625, -0.578208518973014, -0.962924840758987, 0.8596336234794393, 0.15582635224217967, -0.0765355288116325, 2.005643236680271, -0.06488594043716489, -0.08473434962016313, 0.5386008034945604, -3.6963506851386896, 0.21591933154066112, 0.17441163228959056, -1.2580860519572639, 1.1386172980491835, -0.1677738778285942, -0.038566626176013585, -1.3394464910184496, -0.3614343969876188, -0.18617496383570398, -0.20673113296674472, -2.5354598595667994, -1.359755190370052, -0.012414433118827617, -0.015273895489286815, 0.010115026132254457, -0.9956971123632314, -0.11457190505611692, -0.9596058421359217, -0.1168860630233155, -1.3565472954659938, -0.055422289548182846, 1.021607118736668, -0.07798293094488429, -19.554777200643205, -0.43470037507435677, -1.7274616905312001, 0.2058355509678762, 0.44632862078752567, -0.2781160630213293, -0.009198151785078323, 0.0770822124424484, -0.13139786005928045, 0.0012064917979159873, -0.571918143114992, -1.8556346729584163, 0.1039822020836283, -2.098225691557168, 0.10484193814490084, 0.16057192819215516, 0.28331127304110026, -0.8221668369605339, -0.5118653265696764, -0.017994459640299, 0.011771425234414323, 0.7904378035188131, -0.2103219282526112, 0.39862403217909304, 0.07508534477535278, 0.10316280275475975, -0.448695830208127, -0.6554220869172553, 0.10277855927126467, 0.022135854688613676, -0.7178942660067946, -0.4116645716179619, 0.43221151751612297, 0.4918497452932371, -0.0905126594331307, 0.022863997145414316, -0.12898817737391255, 0.010830631844118699, -0.6942008243312472, -0.06543605281810017, -0.25259360963803346, -0.13408144375733855, -0.11175797364154505, 1.3328598195215307, -0.03700474674338494, 0.7790965439810762, 0.20662008948784205, -0.7813218779974846, -0.19179344316090408, -0.09869310598894998, -0.11119339864526978, -0.30707715701588967, -0.10007392209883648, -0.8831130826314038, 0.057316645150336853, -0.02214335445317772, 0.41788708300834543, -0.2866676939941968, 0.48651304409099083, 0.9722921370530742, 0.10248240003070208, 
-0.1222956433536666, 0.01774610496548945, 0.4403132177827018, -0.2428098628654709, 0.7558997148255315, 0.06059711925874822, 0.5327226597384751, 1.250246218869826, -0.05586218767411566, 0.02620663670436585, 0.7893128647496539, -0.22388031596419466, 0.15765473924825102, 0.04350626533177859, -0.0558252682439111, 0.8855977290339396, 0.0880457887247772, 0.9656248159499796, 0.1567914768340799, 0.46817416017840685, 0.06837926096016034, 1.4451748045483654, -0.13742204210031161, 0.017661101669144585, -0.34075040561116765, 0.014047485112968161, -0.21266238832279782, -0.0905352291864503, -0.06808320022825143, 0.010970013300536152, -0.009817074815487103, -0.1436152649350717, -0.04417043323974923, 0.1470084720499969, -0.1281460785981423, -0.0352127875243653, 0.42444258405544844, -0.36680258882439887, 0.18633081820502184, -0.3851715432120386, 0.5737325212589681, -0.5456774981190478, 0.3285520715741299, 0.3598731333918579, -0.1878314272059214, -0.38560963376832547, -0.28194293385757874, -0.5937176301658482, 0.35195195749763286, -0.1548862058458127, -2.72160653465302, -0.898490121794417, -0.3282759847016621, -0.2101548489043381, 0.18096410121216167, -1.3661619702215688, 0.4554010720176507, 0.7571522601954825, 0.5064184205652822, -0.08945779787843122, 0.0029235260634550286, -0.9282982998935694, 0.3863697765210219, 0.9245655450355201, -0.18843573654400814, -0.5832142326874106, -0.10215620553282179, -0.8425388206557898, -0.12677520114219476, -0.2634180524204055, -1.044280043816741, -3.343817557211736, -0.09526847149103901, 0.7283489730157982, -0.735364738262092, -0.04287724694638939, 0.24563617905313606, -0.09742677012459211, -0.9769882400595407, 0.22886766903361888, 0.1581162048379241, -0.37720978572401975, 0.1710823712505487, -0.5727282299645893, -0.21063932841119026, 
0.08789691426322577, -0.3052061188671331, -0.20247908164081707, -0.25648932147953474, 0.02880839001986857, 1.522244058563743, -0.6562169722558053, 0.01451284442763523, -0.789315045658249, -0.41712806262623303, 0.7982427995465429, -0.7331018412788595, -5.911419333084304, -0.013156067928598247, 1.8737086074925173, -0.5520734314504239, -0.1448468183929421, 0.39812437859279726, -3.115471771948286, 0.18821750548465843, -0.8921572420279062, 0.4324381735140008, -0.06542183983040673, -1.2724402374157506, -0.09850153766772962, 0.1588453111594248, -0.053104545702140626, -0.11227295264080794, 0.4184031618855357, -1.3747082261994095, -3.1078915276035843, -0.8344630849918246, -0.5833278664937183, -0.3357204692637339, 0.1902408445498054, 1.6752697732927118, 0.14219428446223503, 0.4334767507617272, 1.3405894464766135, -0.7179477188011845, 1.7882872536435173, -0.13114470878653606, -1.1607031236456464, -0.14863301569148746, -0.5564368473888663, -0.8097447982039228, -0.060506912940695656, -0.021323978907702212, -0.8359027256694489, -1.7706586903313877, 0.005847226000824435, 0.03027046602637995, 1.3520098511847038, -0.37464031184491375, -1.8787147103710424, 0.7030360522786268, 1.1826620577305533, 0.02558240108791665, -0.09637648383539954, 0.026431112120434364, -0.7443411027308571, 0.9497067735973843, -0.06576272049669285, 0.301645180647796, 0.14277934886256105, -1.6115815038619417, 0.05433363976084138, 0.12353762183124672, -1.1627185581818935, -1.8712108292413347, -1.16994784791828, -0.00817263810816371, -0.533135124384998, -0.00022055079168126213, -1.8615885286914704, 0.5515179136501445, -0.011902473224694177, 
0.03560802912196692, 0.19535645864431217, -0.7062386529009217, -0.6985538354723246, 1.937276492851339, -0.08682938134050033, -1.212643509093951, -0.09312089877344931, 1.7138691524761924, -0.6390804973127757, -1.2184873030044514, -0.0005629517970149323, 2.000869898620195, -0.36670322251106313, 0.028940793546297527, 0.006915054763069861, -0.09054028200941389, -2.5413296812537283, -0.16643144815375877, -0.24209598024607004, -0.030331720523854244, 1.85751041047601, -0.09483807675317024, 0.795265611936685, -0.5077974756793253, 0.3067354779976199, -0.015453125625782604, 2.1611882963019866, 0.006005794908290341, 0.20534709885671276, -0.3665726779726981, -1.7272455177769643, 0.7855974791402502, -0.03012120474090807, 0.24351541199204974, -0.1614636905724467, -0.18585495454122736, 0.08979160165399946, -0.12672506361484182, 0.28142336845097304, -0.5587872413719879, 1.0316919422621282, 0.8939983778920606, 1.0093789802606068, 0.056697922662124256, 0.5090470027268381, 0.10859468980095954, -1.016429137340097, -0.001345474236956079, 0.37881541666327223, 0.09127422154060127, 0.28226345312369006, 0.4630714828881253, 0.12706566353285353, -0.024461707129347587, -0.29655201214711013, -0.3263831837944977, 0.16459024780272813, 0.02381007020649406, -0.6033623092709917, -1.7694180305738882, 0.08141810342179667, 0.5540192608434822, 0.3009555876709413, 0.1326144580033315, -0.06912843092207765, -3.2098028943219887, -0.6284529751593197, 0.22655500492847835, 0.5925876777250778, -0.4084362437066389, -0.4666309071717194, 1.0186448106339725, 0.007699077754629968, 0.36988852820202567, -0.28088798090493583, 0.17194153079864805, -0.4524440639042524, 0.001985674896651801, 0.40978837896577147, -0.2085489520958106, -0.3052063205792521, -0.06272418384926937, -0.042838139430962485, 0.03431301274663667, -0.34082597209547316, -0.17626744667197158, 0.31698936726803595, 0.6004612289652158, -0.16103153593076058, 0.3249959377601499, 0.14395920242958482, 0.0456711939113319, 0.5544403784960537, 0.7326693256849524, -0.005860282550806062, 0.6836278652882228, 0.11331788776257667, -0.02722059097467877, 0.02275943841143402, -0.10826953631102754, 0.007423985434783731, -0.02278027013874606, -0.005728897122530441, -0.012571899273368103, 0.016753238546542093, -0.015835428609435098, 0.9579273316112307, -0.24789721623683647, -0.5259606412454048, -0.008337255264166288, 0.20311937012006354, -0.2357789240517718, -0.07490504440891499, 0.007447140592660162, 0.8738267129015185, 0.11114062294580407, -0.022046092940904316, 0.08526729322367288, 0.010894638268744927, -0.025645471351111837, 0.1375674189380618, -0.03544720333143658, 0.16744088071409635, 0.43183968313574894, -0.7001255635745878, -0.13615225692675637, -0.8688289004026899, -0.35097447073535193, 0.3323886551014539, -0.8576621333913153, 0.4215575466018464, 0.0246222926955042, -0.846773575640384, -0.2235692523615711, 0.1702746036364431, -0.3389786833859034, -3.550101749440678, -0.05173933190896207, -0.5198055356674374, -1.9336851643929123, -0.010693377554552796, -4.528920518379181, 0.13997049015672758, -0.487741708327718, -0.9589740214064335, 1.1611282522073196, 0.4743985156630117, 0.2313612063817132, -0.191070913190714, 0.004233937468974602, -1.0734289579953042, 0.7655615192547316, -0.08941888465140835, -0.5445721338929275, -0.2802654147265997, -1.2062310943148589, -2.820035031952365, -0.8407691238166564, 0.14104551351840655, -0.6783318035360821, -1.03646531183729, -0.023808363245734654, 0.5062342303743037, 0.3143382708172453, 0.7873667339576969, 0.3205213384254364, 0.034761938881693766, -0.1434297052075877, -1.1563125168800497, -0.35485420479623997, 2.024899099106335, 2.04190923295954, 0.09573699588147377, 0.037864905385006864, 0.19246985142910233, -0.34340986913845484, -1.4783186231364311, 2.9416141986501287, 0.023224582956011375, 0.8464919019511683, 1.6370185234994068, 0.01546866754708276, -0.5883025811093958, -0.7315465362143954, -1.3176930415838086, -0.19004631281771367, -0.5574592705593773, 0.004392980161824767, 0.9854261345444115, -0.0984792569132864, -0.11255417194932704, -0.2122438525075836, 0.3286295778378019, 0.017095593749608895, -0.4757142666532843, 0.06455360621677642, -0.06279463462747259, -0.057696810088458506, -0.7989390871846211, 0.3614499822404724, 0.029018074192492804, 0.28605850792470733, -4.085627263498857, 0.0005147959543663205, 0.8282248647971358, -0.0644936964093219, -4.017992212713601, 0.508313930635353, -0.38814095538110394, -0.39266760552641244, 0.252610351909496, -0.29000381977567713, 0.9136005396885665, -0.7840813446326749, 0.16924502660548768, -0.010271192226795733, 0.3181829840988044, -0.3637459342603435, 1.2047881571948622, -0.4705234185942757, 0.37154350393632285, -0.3931374061844508, -0.004506693387362759, -0.7043921768505044, 0.04244306073773174, -0.484182453388037, -0.03147788394358031, -0.897805005529932, 0.048537345983504565, 1.0834467512057984, 0.23876291929101257, -0.05063597495804828, -0.161338444848802, -0.08042511707359523, -0.34856882454465676, -0.45817305928642327, 0.4111024666835732, 0.5556513073589926, -0.41213328136716854, -0.5930593614825028, -0.2035943896049872, -0.4312650445368007, -0.2566734728830795, -0.42612674839860176, 0.15137781435073805, 
0.11939594978526813, 0.5256459046049784, 0.0451162808067096, -0.9270790961918323, -0.6888404173890592, -0.437458718564244, -0.00984355719254637, 0.05303241895516919, 0.7246071797305049, 0.07203982367634865, 0.5848445299845579, -0.09726662439214095, 0.13031821744492333, 0.20467513539236393, 0.31619464096600325, 0.9284552982774699, 0.6320370006665712, 0.19727724233074184, -0.3719733102523364, -0.5139346432689109, -0.5188172707698193, 0.04014620259784607, 0.4752278722307466, -1.0077448186143272, 0.014098368467656996, -0.1242970735761375, -0.5589573027338091, 2.6471905948535976, -0.22307050102896397, 0.7117040491295796, -0.1279326590369152, -0.2983239280526888, -0.12216071396837691, 0.1765415999565363, 0.1619118281740839, 0.13932935572801597, 0.19193989711736492, 0.19163973428126724, 0.3951568746464247, -0.2687529178168546, 0.13653049662196537, -0.010710435110780736, -0.10039934088059965, 0.3757949790452386, 0.13195480519281944, 0.2154825464665322, -0.7765492222405186, -0.5865756664232578, 0.5970749362911434, 0.01300931466053612, 0.09094731800530553, 0.036392316050644524, 0.01769038907561793, -0.25579037232108703, 0.9581418964156683, -0.6865349194115424, 0.7486189108566492, 0.45892946287250425, 0.4666739989912685, 0.3970524993295754, 0.14719536724265936, -0.8296098757318688, -2.211076198709712, 1.0382810277313368, 0.03757455886600525, 0.46940905973978886, -1.0395808647292881, -2.416014100038751, 1.2974037974723487, 0.9051552036873005, 0.088053334512594, 0.06096355473894022, -0.01584866858032541, 0.2279567260573856, -0.07171785724905533, -0.14080534704883974, -0.9308656499919366, 0.09181636360637668, -1.3175095077885572, 1.0031831105082105, 0.2626769616527014, -1.1312390736498656, 0.77984311482669, -2.197682104196957, -0.3747667013899374, 0.18919117376799477, 0.1545872899523033, -0.7154361683044357, 0.39188685012967994, -0.3033394331335373, 0.07252424220587983, 2.871236676139617, -0.1644958072378131, -1.121951329506814, -0.6172233414041131, -0.057281393815650006, -0.053103482055590234, -0.2553287710389469, -0.13656497612926444, 1.3731507009889583, -1.0344894940180893, -0.1755942655372138, -1.0476999709706263, -0.5859351032034965, -0.123567085480218, -0.20675417627737858, -0.37756089113287317, -0.233226543729117, -0.21670568673462753, -0.0024072999062099143, -0.4266878841701853, 0.36477435084631793, -0.0824265162033484, 0.6680623639688363, 0.3712525642312414, -1.4711110501120288, 0.2376969503208599, 1.8218767846639423, 0.19405167332284634, 0.004290038886566094, 0.2470550088573873, -2.3152019306174623, 0.47936562342827926, -0.2232872403861066, -0.059222178728632, 0.13599823711373915, 0.8471617406090388, 0.6076740411058665, -0.07921155846048578, 0.9726013504432274, 0.14525376840596493, -0.5398976010311927, -0.09172080414684558]

    diff4 = [0.6212004329272531, -0.9885594099188921, -0.7770016160280733, -0.1577208037600144, -0.42171807502881364, -0.011456787334559237, 0.4385081917560001, -0.2852739887135627, -0.051153259292867403, 0.25836398248566184, 0.003936504878595315, -0.06456532073612209, -2.1135992919679296, 0.3949083655700534, 0.007177748104233217, -0.11107126685142532, -1.2449305437048679, -0.3971413009733169, 5.796349319484349, 0.5459844256199773, 0.6093381340133988, -0.014661983682799473, -0.07876771772539826, -0.007293538382874942, 0.919819376640902, 0.4226061952918059, 0.7285617099302328, -0.38859459436812926, 0.3600032052570157, -2.5687681221206162, -0.0330913587761259, 0.14735274760070638, -1.6839162034123518, 0.05637690307250409, -1.48231922858038, 0.024724242348987957, 0.5202096684608577, -0.32288649681513704, 0.6303620109867012, -0.332071682631252, -0.11352878251075538, -0.05713353065848281, -0.9119888121846742, 0.07103181595635277, -0.387006645775422, -0.2655399629718289, -0.06824890427447627, -0.5331623994499637, 0.0020730573894098825, -1.255043366232968, -1.744359628270658, -0.8260310143381275, 0.015930996589005986, 0.34321320990824944, -0.5392372040490159, -2.4851928390692493, -0.1439198734517504, -2.497587254688682, -1.5461489444666938, -0.040365409289492504, -0.012496972087127745, 0.013505706780904347, 0.41162002919769236, -0.09140731596207274, -0.2612208207750655, -0.26734019589488867, -0.8496655351971967, -0.03992404563953755, 1.4549904931825495, -0.48819295721276035, 0.0006681082431754248, 0.0915760436761417, -0.6481031499614431, -0.4008921178488123, 0.048457132998514396, 0.2097525372252349, -0.07088915666925288, -0.034979767061663836, -0.29926355500226265, -0.09823814203140557, -0.014017787273282067, 0.48821345705933084, 0.34822240475742205, 0.032059082200639466, 0.4073469531465861, 0.10071589478907583, 0.2915197336908051, 0.6990520270186664, 0.9555610726589521, -1.3312937020945057, -1.0160759353370992, -1.5130452798201333, -0.7685466870177748, -1.2281589109785358, -0.6547283770542691, 0.3706624052396421, 0.05964270112668402, -2.6337842941360634, 1.5625753115026626, 0.5409910485842033, 1.2637709576019063, 1.1219969313611813, 2.1915112483669645, -0.6175793705838899, -0.5235241539732698, 0.18157452671503194, -0.1731165213393453, 0.024400645889144812, 0.14358288659479967, 0.46445992938704705, -0.008710564519788022, 0.5238750178061622, 1.8497254960535656, 0.6030916859944, 0.4197789558320437, 0.8233986813384888, 0.05912539979868825, -3.006537663722952, 1.085930232667124, -0.016192188678012087, 0.181884180591652, 0.5683646845726429, -0.7200409115306883, -0.018908127393402197, -0.6985790682193169, -0.9959533282080955, 0.7825864296628708, -2.8111997275554472, 0.7953550638362685, 1.1021402872836887, 1.3236136598589638, -0.6389985138830525, 1.4685049418024008, -0.05767985777497131, -0.22528632598955767, 0.884451502914672, 0.008159005043410161, 2.14533157604302, 0.4748964917126415, 0.008668823193005437, -0.1868397619992308, -1.6447213885242036, -1.8888208053889457, -1.1224637488237832, -13.600119613485894, 0.01180827432332876, 0.5007218098036361, -1.769862418900253, -3.473969232609548, -0.7565398877689233, -1.5129589256969354, 0.004051601739867294, 3.4603776814388425, -0.6611710672771522, -19.27880904056829, 0.6731156803891594, -0.02773673635320506, 0.33227805213057593, -0.3686658203155062, 0.2018975533427323, -1.1321422127606553, 1.4788808394004178, -0.5848575551720501, -4.989638250022303, 1.7762084546692662, -0.34013499322043117, 1.2237063828615078, 1.8171088380409373, -0.5583050010628625, 0.11934163826134636, 1.234729173243494, -0.24388625337076064, -0.24799881416196, 0.10476460461084258, 0.42334841990071936, 0.582099668003373, 0.03178064196669084, -0.07643115459130456, 0.5608659081061802, 0.1033190460435236, -0.682663165960399, -1.2790134968558888, 0.2157220775500477, 0.8757318932845024, 0.027317082775340396, -2.1127663617952948, 0.5521252312801295, 0.2061303446707754, -0.3884805282429795, -0.03121814232035547, -0.05437805574990051, -1.1357910836958638, -0.9276682703572448, -0.9482484244718776, 0.2035399689608397, -0.29348113892316263, -0.8778195872470036, -0.16446947972035275, -0.20361546406427777, 0.8505632592526098, -0.198368033308725, 0.3099065246257027, -0.20528889728754507, -0.35573356649435794, 0.04132307372356081, -0.10894473868025045, -1.2367388341155063, -1.0066306734837838, -1.152975868919377, 0.0025590943047930637, 0.6708760662179287, -0.2945165844107862, -0.3476341917417898, 0.37048081995726534, -0.6197122462268254, -0.1993729922280778, 0.9865363763890187, -0.6103527248191369, 0.7806546283280937, 0.7919763136793989, 0.1972874971247407, -0.17200234074893928, 0.19082612276145028, -0.06302076285898295, 0.050068267004704126, -0.784417113204654, -0.5033914946074844, -1.3621819938520403, 0.27630839579470745, -1.306063197614037, 0.8502042948002924, -0.06086759908616557, -0.4675699308585024, -0.1576958306705194, -2.0504248711753377, 0.18823287011993983, -1.2289809459050787, 0.5794812960340465, -1.0677389356218399, -0.3439300882416916, 0.2339348107575745, -0.7929116766883837, -0.00984818736379367, -0.3248101111713737, 0.07573285207580227, 0.23861492809429308, -0.11226381292186716, 0.5464399625937304, 0.031400948812823515, 0.31200180841942426, -0.06978066751172207, -0.3473152532700041, -0.8330333856207659, 0.23331816591782228, -0.31490728488658704, -0.37635204099522923, -0.44114935184977355, 0.8648155876615817, 0.33134902393398136, -1.0918595938083087, -0.37635178985622986, 0.38562786820109807, 1.0752979589510758, 0.04850261219495877, 0.43267781061162225, -1.911514686011074, -0.597451818517456, 0.007873229142404625, 0.4923418563507056, -0.630532118051903, -1.1710637431466182, -1.4378003297492583, -0.031122442476657852, 0.8612244259033517, 0.03300108601034424, 0.01109931520844043, 0.9420995585778655, -0.9199638413649751, -0.7425971543779895, 0.6277462070110573, -0.7587003829830934, -0.07071002475164789, 1.3600418411070265, 0.4242203155048543, -0.13458352537081453, -0.8361096456113586, -1.891630017830721, 1.1926641409662082, 0.34661586044207127, 0.19557091249063774, -0.9593411475286189, 0.6040467573794501, 0.45706474186610535, 1.4577772825948259, -0.03971128684479197, -6.215338044588378, 0.08427865335288232, 1.936523637197638, 1.3601852058450064, 0.2336691937557447, 0.67606785568708, 1.1229779390540955, -0.35297729237321107, 2.1438127184516844, -0.1322094580775115, 1.39847085240838, -1.0039417812531468, 0.10088055408882468, -0.1326351088349753, 0.23442436927555832, -1.8299060715323776, -0.44867515551634085, -0.4518965011941205, 0.34521611456921164, 0.607905686282308, 1.160594185366108, 0.37722094016380936, 0.19117964556023992, -0.5234445199337756, 0.015834773781065792, 0.5945646071837416, 0.018855359930057602, -12.415884004910481, -0.06173353270754944, -0.7955134879933041, -0.4908456700628676, -0.43833465880160105, 0.1369030973323646, 1.4431696490473627, -3.793841994950654, 0.18747944724148624, -0.17078806727442242, 0.7728898219108089, -0.06253742620634739, -0.1277013953843209, -6.528408669357134, -1.144901280236752, 0.3204959206687761, -0.5469532992470221, -0.05771178403669808, -0.4713400257467981, -0.11706097867303811, -0.6963145335965777, 0.07012333843506724, 0.1049719599824499, -0.40635491142909075, 0.035757411977904496, -0.0008043726714532795, 0.12828205305425655, -0.07902992129181996, 0.0020954058123123787, -0.04821703086457063, -0.6102036008102374, -0.044271000040254194, 0.09230636732648634, 0.20561895159014654, 0.07237253654372466, 0.02366059414174515, 0.11893310292741432, -0.0382168466740751, -0.22728826415922754, 0.029978334602063228, 0.2550513763198978, -0.34750860485173973, -0.39157071736593707, 0.7340400276985122, -0.06003735706299551, 0.4414681635574311, -0.34660720338335693, -0.28697071470118374, 0.008390631472849464, -0.018148585437998577, -0.5433413520619723, -0.0939931335236679, 0.2145562090758233, 0.36773423316151366, 0.2445442666791564, 0.029936971175807514, 0.08404193684303607, 0.03611532639820325, 0.49043150173683614, -0.0004282380513558337, 0.0825538518810518, 0.20909479159170985, -0.4617347586604481, -0.0572882003122146, 0.6407716891079787, 1.4951279479046065, 0.1790445954294242, 0.7370696744208232, 0.7824807112191934, 0.12127614615626925, -0.04736580241932131, 0.2208546118535395, 0.6246791350396563, 0.04066940660650431, 0.021392725696657067, -0.045461929162939896, 0.6228952949957289, -0.06717867316898207, -0.2922522104615837, 0.16252131880031584, -0.822665138942952, -0.02672622987083173, 0.6494440126102177, 0.5627741886812032, 0.022236435169439517, -0.2946128604639, 0.31563479900077596, 0.19890035782051996, -0.021397249168344956, -0.009494925691868161, -0.001686101685706376, 0.04597972487339774, 0.3599913404576114, -0.07927505298377469, 0.010198345392872454, -1.2611882582671825, -0.7947380238744444, -0.527894728687258, 0.24910997420681724, -0.19649798338779334, 1.0283906781705312, 0.2689386912652836, -0.037969053334954594, -0.6407761395616589, -0.11550506374483405, 0.7647646252956974, -0.23171853962405464, 0.08025556355781305, 0.23774547801510693, -0.13192495755122025, -0.3341258856942204, 
-0.6969528566949066, -0.11419123360283834, -0.4283015403693824, 0.04464190793865441, 0.3227016494478079, -1.0985261730313454, 1.0937355337065924, 1.0664180128625773, 0.2978111380567299, -1.8151930984074198, 0.021611880310146603, 0.1332461160862337, -0.1545520355921326, 0.1552503283408413, 2.548014916156127, 0.24148505144044918, 0.0698226616633093, -0.8889229679166704, -0.14664857849963653, 0.4891298188348969, -0.2659260773839094, -0.9048909222795913, 0.015760791055299705, -0.016705994108427547, 0.024285297861595723, -0.42916195441976157, 0.26691354029822634, -0.04466869464710754, -1.0033812151244348, 0.7649702562913347, 0.031236640485786893, -0.19073073238367044, -0.2767411859309945, -2.6537903508468403, 0.5171911256351223, 0.3841380090242268, 0.01216314857596501, 1.1219935778909047, 0.044629296504403726, -0.02809941028191787, 0.21529847918656486, -3.551657043701155, -0.041666766782185505, -0.39112413200660257, -1.1356767702313988, -0.25616717579299575, 0.08009820056078354, -0.24018145734691387, -0.03293163318879522, 0.6294804516136026, -2.2566647874661285, 0.3653765219769127, -0.04456892269399049, -0.1734006520835507, -0.06074454869933277, -1.0129327170765379, -0.001571479333797754, -0.01406026520707826, 0.00823952080141055, -0.7290566165170844, -0.08190929300101857, 0.13745920614166351, -0.3500251018418794, -1.6641903056663097, -0.38074497517308714, 0.3637078863647858, -1.2184389932147113, -0.2577455132556494, 0.009075729783717179, 
0.22370354480505483, 0.2896790703717471, 0.0706810357310701, 0.06977711721246038, -0.4142787341653964, -0.13902163075696095, 0.28015260400345454, -0.025181211476255072, 0.12133137749644618, -0.1549230317173631, 0.06465114349408907, 0.2863768243532121, 0.01208255168855743, -0.03426751278852436, 0.006541366861078757, -0.23501696974211583, -0.01049556833241283, -0.027580692613248914, -0.11657134503274591, -0.06255412073548783, -0.29995331172824535, 0.026853605796713254, 0.15671487024145137, -0.02035185211764201, -0.011272433654692549, -0.013511925546371373, 0.17619631000758318, 0.2252267908124317, -0.04953748195989505, -0.04281931429733987, 0.07354864850671561, 0.5910538591932237, -0.022168196545248264, -0.2442195086842105, -0.2125059100539488, 0.06202995125660493, -0.527857954783066, 0.020342528423125117, -0.16642950678485846, -0.051239283682939174, -0.012366931248571689, 0.2869363264233016, 0.026088403621074008, -0.018366821760182006, 0.09834280081791036, -0.04981711772818542, -0.18886162299214249, 0.15309154035423234, 0.029822372175225098, 0.03830981401424083, 0.022322298460341727, 0.1614079837780409, 0.14481375674104413, 0.3461745033002117, -0.00027716172602865186, 0.09512040302477942, -0.15360402937837847, -0.0011879445484517248, 0.0003674400161663982, -0.226061159926509, -0.07776687414794026, 0.037313907780205824, -0.027351739790454133, -0.0014687948943077345, 0.32242228257819505, 0.003456306223210248, 0.16069260282565523, -0.032924175699356795, 0.055074917875224116, 0.004295522091961601, -0.21435187006778023, 0.0868873041519933, 0.0022520115629731663, 0.016749126347242793, -0.42868701463786607, 0.051039326369132, 0.04089746838636188, -0.011757605834411322, -0.02037314175596805, 0.00030378727885249646, 0.09284618456304372, -0.02329081959567958, 0.019290063387337852, -0.1373747598741275, -0.33665572103402397, -0.16936510319160902, -0.4034492835945045, -0.14432879966391, -0.14711081261910408, -0.13620556330273814, -0.8558247146970714, -0.15461144631724721, -0.005919076227634434, 0.1875292693742665, -0.04095025661610663, -0.24964327399067798, 0.14564664433336816, 0.10231149765663616, -0.20121167104232285, 0.5141623528296932, 0.5427959914973499, 0.10699008051418701, -0.8562852531462255, -0.2955357749376475, -1.3207271588734102, 0.5331238851129072, 0.29915683680487604, 0.3007414214205042, 0.08385294782731734, 0.16051763228056615, 0.13306365745776105, -0.37133091180409394, -0.41642336242006195, 0.16834739180011837, -0.3599389112177249, -0.89985427789955, 0.20735625233477606, 0.19895700998331733, -0.0491969939823349, -0.2878932583168705, -0.46624696436225577, 0.004733240519911419, 0.2036215557280059, 0.05218272078367647, 0.4818288500801984, -0.5803195375407029, -0.08220219924273664, 0.21677869459775678, -1.089590696154822, -0.11847157651608597, 0.6195061707543772, -0.24249585565041798, -0.6638469169340766, -0.07333373481432659, 0.14787054484779105, -0.5923220271723153, 0.4615745270558804, 0.28799733553329787, -0.19979228380295666, 0.8173293492785447, 0.06799919447839642, -0.24516137873554555, -0.7960975627110187, -0.8481225608506833, -1.225287021267988, 0.09150657091658587, -0.4884418778637638, 0.07592771139243126, 1.2761048245293907, -1.4381454643466682, 1.045446475290717, -0.2373914197908178, 0.34747333419537085, -0.11675638839101765, 0.3278168124922871, -0.046899145832938416, 0.06968061757770982, 0.20207159432528954, -0.11588131116482714, -0.031856904898312166, -0.30963465217049446, -0.30226062356664585, -0.6071036897535649, 0.14890681232956382, 0.5116699296782485, 0.3558229479754047, 0.33006332062031873, 0.08422704443452034, -0.30065230833319134, 0.13056508660847044, -0.8485485653556566, 0.2837263829033816, 0.1344342807610559, -0.22621042076120546, -0.43096240523532003, -0.08224899480679682, 0.03734522105425242, 0.3178002820085908, 0.1043801353331304, 0.4407737491931414, 0.3002336151180458, -0.0642405055374553, -0.09917971880712884, -0.3399102508122098, -0.026860765176557777, -0.05884677294208984, -0.009724975439098671, 0.2799283183544219, 4.705269600860987, 1.0619067720702091, 1.3363769338299676, 0.2562134119571695, 0.0059088404412932505, -0.018636917678755083, -0.45125962913064654, 1.6409515315712042, -1.634826769434028, -0.09475883931018814, 0.20221420366559073, 0.6866708557344623, 1.7435374730991668, -4.494107199838211, -0.3183068303593899, -0.3632209395917272, -2.9338689019929234, -0.015054489312412045, 2.4491596783790897, -0.172557578077857, -0.7942341927430618, 0.5259711511120031, 0.00012162658185843611, -0.0510097780417027, 3.1444245468955785, 
-0.08694628710144059, -0.3910820788167513, 0.18057745441107897, 0.004997603113366722, 0.9799972074211922, 0.39636279134024477, 0.47972819741244166, -0.5350151910243994, 0.5948369289398414, 0.03265961984209298, 0.8750728347112897, -0.11934386469374658, -0.011588950661526098, 0.20553219741604067, 0.8287280874733227, 0.16753399321132179, -0.03725860626194333, 0.05524138408611634, -0.02673167589459524, -0.5800516799688467, 0.05493611387511521, 1.372078886403017, -3.255180551385479, 1.378991243478211, -0.052619079011627434, 2.186068646554915, -0.2383212208572445, -0.009167384522982047, 0.37412827882735655, -0.8774830088060241, 0.82914926048403, 
0.012016217283154162, -1.882960907778127, -0.05749026061411655, 0.012095633004285844, 0.8506906130554484, -0.12758001073707703, 0.4050667661796581, -0.18451849441206036, 0.22084437897595421, -0.4154974319702376, 0.034688238857285114, 0.0708954881332744, -0.10130200048691762, -0.2645834330433843, -0.16910169478619963, -0.03418388328575617, 0.044322732500702955, 0.08388414248909015, 0.003722000481161558, 0.03667951322463381, 0.041611150091803495, -0.0028325398719886152, 0.0069529743986471715, 0.16779019144529528, -0.0291693043345731, -0.007040114484990312, -0.7477382259841718, -0.09628967118649712, -0.4164879187880146, 0.1508633593399047, -0.20705730310435655, 0.03518779790162441, -0.05665381405320602, 0.07700573435210245, -0.2021329245234469, -0.1773211256384073, -0.14861792655531758, 0.113971361987943, -0.05021429998285143, -0.1909774948601779, 0.05439770538226085, -0.18390045503455354, -0.18315436348893144, -0.02941390477424477, -0.5373588931741935, -0.03619908554166784, -0.10954814905776544, -0.10521171393711626, 0.21517711216564805, 0.1584064859753127, -0.04126303308231627, -0.03313453782088516, -0.161577284788315, -0.06998843114902087, 0.15778069444509413, 0.12977309732605136, 0.0022795748553186, 0.07896510827581693, -0.008670157755222618, -0.04128026987448408, -0.39781643869110894, 0.1657688792014227, -0.008142569487755935, -0.03138538914516431, -0.05364206840637564, 0.0017576335741935623, -0.001767986295597268, -0.1806399120002311, -0.594823513994946, -0.011817703678079994, 0.032430427065431644, -0.022481852318186668, 0.13535005525692156, 0.001127407983769757, 0.07342382240278766, 0.17203710086911883, -0.2854425213605438, 0.006488536529008826, 0.47834191251453007, -0.011418459492482214, -0.04164667670134392, -0.15461248780102466, -0.05293061371856567, -0.13363755846307512, -0.042829539537770245, -0.0214904780777303, -0.011655251950070777, -0.003462398537646294, 0.1039261329279384, -0.03155699407135515, 0.00750387194213431, -0.12632216983435995, 0.18091672523615898, -0.1037991193421206, -0.03978060696794472, 0.19106557605980612, 0.065756616506782, -0.09151796940000168, -0.0816095747115071, -0.0011572990595496435, 0.044891220046885394, 0.03104749769642723, 0.03724756784026795, -0.10880190844037685, 0.0686039445849218, 0.014524977054307442, 0.027663634984222085, -0.45875869485090703, -0.21820585665322056, -0.35607853015011415, -0.46140546135375615, -0.10433238540181122, -0.16360842944177278, 0.2250143130824469, -0.16380667090938772, 0.029762690100177736, -0.016835505146048035, -0.03422676125070723, -0.7302117448145466, 0.7462403301357909, -0.3046269427219457, -0.24299661331919253, -0.18749927997847493, -0.14152696906588602, -0.07948294970020697, -0.23531788304203616, -0.16834007215672386, -0.2491989629849911, -0.2608495210213775, -0.015758330075641425, 0.3316933583641912, -0.2228846864688947, -0.28367726045031105, 0.3225200521892475, 0.011325918347267816, -0.10928861500730847, -0.0658959951877236, 0.2954760643670369, -0.18684316310022098, 0.011864002349383895, -0.2903528070560455, 0.1474580889295467, 0.03784272687464352, -0.11266292538545031, -0.37324348184222345, 0.14735511423061354, 0.00968522345334577, -0.21315989841969696, -0.08011694692977045, 0.02535527806532656, -0.1251924687383692, 0.12207785285404782, -0.2448737718620606, 0.12542211689468274, -0.5979838822874797, 0.04610925622665718, -0.3687431914456667, -0.029477522057412386, 0.08222807017183698, 0.047817169557617945, -1.0416258888518044, 0.006733011160910962, 0.7283127525006847, -0.010145150604280673, 0.0014771300239573293, -0.09359083221588804, 0.2617813708293184, -0.13871169997715782, -0.7665830431597058, 0.0395949200334087, 0.12451705225579701, 0.06709731022384346, 0.19767618350035576, -0.00014283185550567623, -0.00011077587668495426, -0.11761599642921539, -1.5491646830276764, -0.1321436814976451, -0.09766996951462659, -0.06407675308682315, 0.5146528523895029, 1.0551409140439318, -0.2588656145185837, 0.008310126984120814, 0.03999749300706412, 0.2618279336497551, -0.31660474953804396, 0.1830291847666814, 0.13283626004193394, 0.15245078987473448, -0.21941348906837987, 0.0952718715884231, 0.07152806031377423, -0.056714987276755835, -0.3040602669435657, -1.2357265181374828, -2.252419548588648, 0.7422540120524559, -0.05511762148522337, 0.3092196687461879, -0.14860576885822496, -0.12078668475483312, 0.849228743714491, -1.2654668187207534, 0.22375508457400883, -0.16302939848697662, -0.6883249303556269, -1.138255501284533, 0.04703955104586299, 1.8481049953970654, -0.35921853497498546, -2.1674613861089256, -2.686525471137074, -0.020255108005713396, -0.7558293018733764, -0.48638578750826866, -0.2244735522113075, 1.0276404079980779, 0.21962229797480148, 0.014438751171510944, 1.174907660615503, -0.13239176977349132, 0.2727073146698302, 0.5047223498823854, -3.6805819937197057, 0.20626669591254654, 0.20899581580862048, -1.041952389297208, 1.005022678356113, 0.3143018272920557, 0.01057136827122207, -1.3772765893197558, -0.05301537958995084, -0.21101527746684923, -0.08843931348589962, -2.564492550320921, -0.8863941676324032, 0.13502058959090135, 0.08099463955463193, 0.005558376920468788, -1.3927017865045457, -0.020710941688328433, -0.9836856437013992, -0.2121873950545421, -0.801007123252468, 0.04784571966929008, 0.5769541712570998, -0.4742458009995403, -22.77996577494967, -0.5768251594488873, -2.0466516606877434, 0.27544295287500375, 0.43002112212128907, -0.47493052881675624, 0.06055671254680561, 0.10628718131818005, -0.11936355568745682, 0.03438166415124755, -0.5624480683231212, -1.7751424798697713, 0.30782204658648027, -2.6957277758696634, 0.23734891219340426, 0.11175052322002443, 0.4276087905011039, -0.9124582599549385, -1.0920829988599507, -0.010884827522524176, 0.06573058431913381, 0.5471313772320769, -0.2506356510834138, 0.3872258664267534, 0.045381425470772285, 0.21766615860389038, -0.537054501270223, -0.690902780878119, 0.22513368393809685, 0.022416412746473213, -0.3136756755816279, -0.4767024322041351, 0.3976063457789536, 0.5560410002537424, 0.11570247231190933, -0.008257807157441732, -0.1296269318656833, -0.024759638775233128, -0.07298237146999043, 0.0392016685053278, -0.014285908490705879, -0.3733495700898217, -0.105782388278989, 1.3615085417586883, -0.03860581068700242, 0.18465644663092462, -0.17033602914191448, -0.08198985770111022, -0.048484039764815634, -0.1803054759221503, 0.2842347835484773, -0.11233139155500993, -0.1463838462593543, -0.6644291059469651, 0.17583716796282545, -0.021981348042857007, -0.005384517358571372, -0.44988176958948145, 0.2474296189743015, 0.9576497676679665, 0.10301119818394255, -0.35269099111578583, -0.009351899120353835, -0.09712728274487148, 0.1769341021841413, 0.6634370093584323, 0.026298129613955723, 0.6451341797940842, 0.31993681221234027, -0.2121419884766773, 0.0883589060001384, 0.8521545018626, -0.3946961287731554, 0.15962562729414032, 0.09475134993394363, -0.11492699712629317, -0.01014479160546955, 0.20947198457430005, 0.88042348104468, 0.054555747585219194, 0.39470479125137814, 0.031834678308630515, 1.2710044360098536, -0.07457416480214363, -0.07340385726070409, -0.40305928457212303, 0.11961364076306324, -0.18544716781107695, -0.09246589437721298, -0.14323240533543213, 0.01062720128631156, -0.019035097185188476, -0.24091443393212586, -0.041839786180268845, 0.23843285664659675, -0.2988270280569765, -0.170606357590664, 0.5336724502475434, -0.45667152935992306, 0.2324453316738726, -0.7779740722739348, -2.4940858724739527, -0.09834791244968244, 0.29426025054892335, 0.350355807686185, 1.8849463431830742, -0.39130006906704295, -0.2367606396059756, -0.43740660490043126, 0.3367918245260384, 0.012123795443379493, -2.5537898425593824, -0.994870575726118, -0.37391948718373413, -0.1035164655744154, 0.23685330785711756, -1.5818489529130773, 
0.4534088801602323, 0.7930713086940102, 0.8756952744324096, -0.03848285646117944, -0.0015687392945977763, -0.6111235430287394, 0.3782965040892208, 0.8638395031251918, 0.14797389325610766, -0.7455894630056008, -0.5675174802554466, -0.8923805195669345, 0.12456603514206677, -0.2912590384361238, -1.0996833348315391, -1.6297118051278972, -0.09463776483533337, 0.739185929432864, -0.6593971825043639, 0.11770306911837736, 0.37414795063156703, -0.18755843265659422, -1.0459692785118122, 0.1482754171207148, 0.5972316241921618, -0.19771054640690977, 0.23544561797208985, -0.5382693800780842, -0.30234757829417447, 0.19019139773207172, -0.30030754722309894, 0.14183340621376317, -0.1053136139162234, 0.027541440764963454, 1.1482255751332247, -0.42549660091550834, 0.010626952921604982, -0.937463499706908, 0.1725578790034774, 0.8516469337987331, -0.7328205884158905, -7.000827137407278, -0.01767701450180681, 1.7416885622195508, -0.41818296978506453, -0.7108813401952858, 0.40430467799133396, -2.1287004007705477, 0.1831085691262686, -0.9817189456909006, 0.43543347599701576, -0.06902819357974721, -1.5169051240161764, 0.17696437428132583, 0.8210816897353226, -0.15191612072320027, -0.16682724969707863, 0.6616708265440678, -1.294201684768332, -4.98657401420553, -1.1387655753481027, -0.5504840037331675, -0.24528210601569356, 1.45813113936849, 1.4118965390183433, 0.05465095407015497, 0.5375618467671757, 1.3077576624627625, -0.5049399267594339, 1.7038064298377265, -0.09705914786017189, -1.7672923931374953, -0.5154375610454593, -0.708440208371627, -0.7371980001055789, -0.1467493641938873, -0.003782828286738038, -0.3649778724137249, -1.3552371414244249, 0.009076160683711976, 0.034770747964572024, 0.49996173925961784, -1.0310077663147865, -2.195025098876357, 0.8192701339435544, 0.9459976461534865, -0.11334087036660634, 0.028507962290561295, 0.015800994248394318, -1.2050644384625144, 0.8956464646374798, 0.09355997699482543, -0.2113991376717479, 0.47960079654686183, 0.2085583705239742, -0.0841600012374073, 0.09637717804349677, -1.7660304433943992, -1.7262732912757315, -0.6376995368629821, -0.008566397383937385, -0.050246748463095514, -0.28095686466087955, -1.6037662020152652, 0.5115437917725956, -0.0048979235441350966, 0.04232663894539712, -0.16838081109759173, 
-0.8851346652923979, -0.16588344648953068, 1.7933606219550313, -0.07347214939951385, -1.3045239532267487, -0.22193268922725906, 1.788252183125266, -1.3765743190751465, -1.1406003578893973, -0.0038960031565480335, 2.067370873025922, 0.20838988680011994, 0.026633382104790826, 0.026848804217927125, 0.1425854341937054, -1.3295716429708477, -0.12950962273627198, -0.23676258385486193, -0.0290416994403202, 0.4461491869823817, -0.1946129522773461, 0.8578557752072555, -0.5684542826204932, 1.7452131458549758, -0.021041488692489452, 1.7315305659811884, -0.01995194539378531, 0.1886213608432783, -0.29084011635789864, -1.6987846025403996, 0.6363968253140513, -0.03699622709175543, 0.2606204187663792, -0.1558521702561606, -0.16898637899691948, 0.18417232852297616, -0.10448260005598797, 0.14133301051243308, -0.36524350960873164, 0.8137381125395535, 0.8699391038703368, 0.7916625475835559, 0.13110032854605436, 0.4662772732678775, -0.04633443724636521, -1.2834962745397718, 0.0012801175711700807, 0.2568070713628998, 0.22877090728547955, 0.11337483667205106, 0.11417694822982583, 0.19666807246689189, -0.015702456270467735, -0.21362106509548084, -0.3354476679091576, 0.025626484169777086, 0.0058275646593983765, -0.10506552909319566, -0.9476139888409492, -0.6668516699203337, 0.6272951191030813, 0.4876631058224987, 0.2011461109371595, -0.07110818285337928, -2.23341943006168, -0.03153290003896814, 0.1895663977003892, 0.7142192741156634, -0.3442939996955232, -0.7084689338281223, 0.3837695482219772, -0.016107591160910317, 0.6906419247265063, -0.5929286874266637, -0.1307039293947554, -0.5688742645097165, 0.002169309164791855, 0.05750600983739673, 0.45515388441326365, -0.3356975304433263, -0.009941896116671955, -0.02978380258700497, 0.013887461427628978, -0.36886774861363847, -0.12773334111666657, 1.0307001610181317, 1.358511888312897, -0.1746534337515655, -0.12304358894201073, 0.21354392317730486, -0.5051115578616105, 0.6569120071331582, 1.1276533801533617, -0.006988570283887441, 0.11130101850392293, 0.22873578365097558, -0.025134814772137304, 0.018160657963122162, -0.041426995371466546, -0.57846362450951, -0.020973501085350676, -0.0030178644830627377, -0.013460039125313017, -0.09035434439184087, -0.02577990955749243, 0.11776092101209201, -0.340237307153302, 
-0.24967170432822883, -0.0076033946916567174, 0.37713779939669223, -0.17836903782030333, -0.07523925388495911, 0.04666461350645079, 0.7778041227400649, 0.057026109396133506, -0.01189711489041656, 0.09691087768361228, 0.012852241755062721, -0.024743517857466202, 0.10355615690625086, -0.046223267195784956, 0.2321186886703046, 0.562113152577794, -0.3823228480731018, 0.20513616750075414, -0.8543188936666155, -0.3334753323883035, 0.4380352771914886, -0.851012320177432, 0.21954213537290457, -8.738196385138508e-06, -0.7679844098394142, -0.06845679805924476, 0.2513205284541442, -0.3301302554768881, -3.781880525494401, 0.008379924500282243, -0.327657689777908, -2.2997530191700264, 0.1694605630137076, -4.886792187697132, 0.3464597885430578, -1.3050957637063476, -2.178300348824891, 0.9320366590025486, 0.5287088176595631, 0.12924167649126161, -0.001282249394769508, -0.008944707782141847, -0.6530273583071704, 0.557317817601529, 0.18559915588109277, -0.6576758802072504, -0.3774199827776812, -0.5442461734522368, -3.3048673340803134, -0.9259105987271141, 0.26158152125344714, -0.5604371311219012, -0.7794958788179969, -0.018212036138756105, 0.618002730901722, 0.347627785340066, 0.29652556951720044, 0.43663044792086225, -0.06725699523321538, -0.055602058803941645, -0.10010376767121443, -0.2041839932024061, 1.9410439595803126, 1.8905901528619466, 0.1326326737461514, 0.04861290118694228, 0.10770622397239293, -0.26146454831491894, -0.9572850089207066, 2.8720516426835445, -0.003659191840988285, 0.4228499449612855, 1.4167405676386835, -0.0007768069161429025, -1.0950153278256138, -0.663982052629791, -1.5449432835430486, -0.18358327598924973, -1.042304946418831, 0.007662716272683667, 0.5403541223884503, -0.10323974916629908, -0.06508924210477574, -0.17977166929109956, -0.16172707884460635, -0.005469816673304706, -1.0303879418647597, 0.08294259856047859, -0.046884471458881194, 0.1462284533398588, -1.3479647272892095, 0.4678964240080319, 
-0.048402889684005856, 0.2422489636634566, -4.091249449610345, -0.010486229212332887, 0.773320784080326, -0.061359847881227836, -4.620982391895929, 0.23167185479171337, -0.3505329828765582, -0.06839416782300134, 0.4199056625252666, -0.2183691136154735, 0.7241289902777126, -0.7198158589266583, 0.4565325201848722, 0.008972258276500611, 0.3129373359706378, -0.609337018445224, 1.1180974968632569, -0.23098259808794097, 0.31440799770476957, -0.30692436972750414, -0.3749272494095024, -0.354993958077209, -0.007768712257984589, -0.22271125643607803, -0.5407680392944911, -0.3264484994380865, -0.021688491100263718, 0.9554994171294524, 0.13699054899758067, -0.048928110167011596, -0.27222380201500584, -0.17194729472496562, -0.8594378621926211, -0.35264465183031746, 0.7188366719315553, 0.6305066149675298, -0.4269991931485322, -0.6198970253784779, -0.08778726181549246, -0.5733718319545176, -0.08169477992605323, 0.007820314570253117, 0.11168666547975192, 0.0517932294922403, 0.8829590746635958, 0.04044000999303421, -1.1666043263289012, -0.4151720925935365, -0.3407902833507066, 0.0007863278125128659, -0.4551043930219265, 0.48192278747018236, 0.20466865870258744, 0.8230347232904336, -0.08090250829578594, -0.08060611608296142, 0.16026523062694054, 0.3879162026083378, 0.6540346855044703, 0.6407665961381639, -0.04431385449744596, -0.03873262997760918, -0.03657959498006136, -0.2078831132362957, 0.03812517981682362, 0.917849335599854, -0.4106206061452866, -0.042817873979466015, -0.09598355944014259, -0.26692759283495704, 0.9233215949887494, -0.1086845838653474, 0.8577552866702121, -0.16603074471386492, -0.48817987228094495, -0.10457908495502011, -0.006278682947822745, -0.022036538939318007, -0.04546018124769802, 0.028611265192139967, -0.1030437631843597, 0.30294246420683635, 
-0.2651500678614269, -0.27009611191589755, 0.007499490246573259, -0.09150383898172265, 0.2905087546884886, 0.07591161635350119, 0.17047378390874712, -0.7417520672654803, -0.44911284517392147, 0.7311215485463691, -0.2655958462170105, 0.13596807410739586, 0.19268889857880112, 0.15992315087200382, -0.4195302987219378, 0.8952859460141553, -0.8019786362998644, 0.73655207631586, 0.8809733435928422, 0.21905023029832194, 0.2968767008294648, -0.010519282809127617, -0.6817256337522153, -2.1207290532071355, 0.9786070140241634, -0.11390127156427354, 0.5754175068300071, -1.1633774912009187, -2.4833750631099605, 1.127354513185864, 0.9792343041293989, 0.04845327108356656, 0.04205996904745035, -0.3637146006754719, 0.5096173289640973, 0.03398210256867884, -0.07742400707074637, -0.8241591695394774, 0.5243945256631903, -1.6108919244115185, 0.9350212724750406, 0.6606108637821393, -0.8383331050130209, 0.7781511758041688, -2.1201107085250186, -0.6412775143874612, 0.2918498363111155, 0.0867586841376351, -1.046337269637874, 0.3012267424194164, -0.2848876070099706, -0.11497346311877266, 2.345970385369867, -0.3087687194096844, -0.8703658005588153, 0.2769561079316105, -0.07593690681871834, 0.040902198391549405, -0.29897953673142297, 0.032742785060463575, 0.6357237511401337, -0.3446714768937227, -0.3571482136636206, -0.5261020147082718, 0.08526573513083235, -0.17731826243080917, -0.30616686921843694, -0.554589198462466, 0.3267943064479226, -0.2288397783148568, -0.07330517843801942, -0.667994207820108, 0.6116311647667914, -0.08149956414342086, 0.5850819242738083, 0.010035187573947724, -1.6455012925349237, 0.12649889743395448, 1.132988788037892, 0.026286369633339746, -0.07730015741805119, 0.4271726233708222, -1.9542346070074075, 0.47710457711325915, -0.23844461298236297, 0.04557165486903614, 0.0031241956321181874, 0.7475141966228946, 0.41504176529760173, -0.28442106513785603, 0.8701825011084807, 0.2640109149661214, -0.5676359092759711, -0.0861246943156857]

    return diff2,diff3,diff4

def get_perf_diffs():
    diff2 = [0.0, -0.9200000000000017, -10.340000000000003, -1.0899999999999999, -14.0, -1.8599999999999994, -0.2400000000000002, 0.9900000000000002, -1.0, 0.6300000000000008, 10.549999999999997, 0.019999999999999574, 0.0, 4.859999999999999, -0.7100000000000009, 0.08999999999999986, 0.75, -14.310000000000002, 0.0, -1.0, 0.0, -10.309999999999999, -0.009999999999999787, 0.9000000000000004, 0.0, -0.5700000000000003, -0.9499999999999993, 10.799999999999997, 0.8100000000000005, 0.0, -0.2599999999999998, -16.47, -2.9800000000000004, -0.009999999999999787, 0.0, -0.33999999999999986, -0.1899999999999995, 0.0, -4.159999999999997, 0.009999999999999787, 0.0, 0.0, -0.009999999999999787, 0.34999999999999964, 0.0, 0.7599999999999998, -16.009999999999998, -0.6699999999999999, -2.5100000000000016, 10.990000000000002, -1.0, -1.4600000000000009, 0.7599999999999998, 0.9900000000000002, 0.8499999999999996, 10.990000000000002, 9.090000000000003, -0.22000000000000064, 20.25, -0.009999999999999787, -2.530000000000001, 10.840000000000003, -16.43395193591455, -2.1599999999999966, 0.0, -1.8599999999999994, -0.7300000000000004, -0.10999999999999943, -0.35999999999999943, 0.1899999999999995, 0.6899999999999995, -0.8099999999999987, -0.9199999999999999, 0.7599999999999998, 0.28999999999999915, -0.3099999999999987, 0.9900000000000002, -0.9900000000000002, 4.590000000000003, 10.530000000000001, 0.0, -7.25, -0.009999999999999787, -0.8599999999999994, 0.19751216941129712, -7.203206620928162, 18.455418097222584, 0.9256504847405331, -0.5796300738403346, -0.007970390872840127, -0.40657002702823064, -0.010026994850449, 1.1935476159494502, 18.455418097222584, 0.044942230486382684, 16.141221595481966, 0.2912317828409776, -2.1657917866106473, -2.3466035947287303, -0.9112666090800552, -0.9375791411732877, 0.18064373241723075, -2.533824470117798, 1.1908762781135014, 12.737230722085613, -0.17185048891863453, -0.29424027556638244, -0.044800601960794495, 0.1569507865605484, -1.7640983059357254, 0.6559949047337774, -2.84698765006182, -0.042027689670634416, -1.2327344242642972, -1.8846231669708828, 0.7829863995858304, -0.49782840060788125, -12.071176299494857, -1.4999331359046124, 3.553530064895865, -0.03185614989449803, 12.548861187773145, 0.003545472430211305, -0.9660421041362497, -8.749082103867845, -0.005294139780666285, 0.7621197524742698, 0.696686395466255, 0.11083171359617161, 2.616882837562329, 0.015733625942441876, -0.3284506367354929, 
-0.38370875787459724, 1.0580251751273018, -14.345113991903903, 1.6447589477813445, 0.7207485462369849, -0.9353133947420442, 8.353615829656363, -4.758350716001065, -0.2340852620654328, 0.6674830928028772, -0.167063933736447, 0.6264216176669422, 3.9617252255325397, -0.9660421041362497, 0.9823265213723591, -2.4710442010595024, -7.893130957538531, 2.616882837562329, 8.803302796340171, -0.2504970130488111, -0.8913786791441112, 0.078367266089721, -0.048321916072271875, -1.99326176103947, -1.9420605333119738, -0.06977976284148113, -0.8870193296837989, 0.0007240127155991871, -0.19476884019631413, 8.239200232237605, -0.012776048779709726, 1.0317751751273008, 3.6316732567240138, -0.3977532196848763, 0.12462154246524726, 21.038571485099553, -0.37588276863660397, 0.217435275205748, 0.3485607479323498, -0.005770716427731415, -1.1963061652800455, 8.577777777777776, 0.09468514249676918, -0.8999999999999986, 0.0, -0.24548830883876072, -0.2644157536337133, 0.2887050488745402, 1.7700000000000031, -0.007407407407406197, 2.135907963390647, 0.7261670081598908, -3.090740740740756, -0.5296049367258426, 0.094337968679298, 2.1739623473949496, 0.6121849801556456, -1.2000000000000028, -1.6000000000000156, 0.2785938172283462, 1.5632457202509968, -0.8900000000000006, 0.031263027164689206, -0.15514066496163714, 2.3999999999999986, -0.008117875089424409, 2.9529724710217735, 0.8852144899904619, -0.3995510565736087, 0.19971862565181198, 3.501041195169517, -0.07656340989970367, 0.07000000000000028, 0.022633451957299044, 0.044683570397866745, -0.5396667824403032, -1.0716310621787244, -0.0018518518518533256, 0.03741332967355415, -0.8221420858345851, 0.0902681813660271, -0.06324087423589297, 2.58580627216638, 0.370579846053241, 0.29961803989988045, -0.7718016535289056, -1.2000000000000028, 5.7486221030025675, -0.9290583304214479, 0.06351096675393286, 0.008148148148148238, -0.0018518518518533256, -0.10080731147151312, 0.04367715747331147, -0.27383299184010923, 0.033950617283949214, -0.24136256276161738, 0.5876372639127219, 0.3843846503258508, -0.614293933017926, -0.6717833590466373, 1.8299999999999983, 2.490000000000002, -0.005555555555559977, -0.2767348541753716, -0.24593005444689275, -0.08803339517624487, 0.13000000000000256, 0.06620479089181153, -4.537037037037045, -0.056360522985801964, -0.6347180480535748, 2.0749910687695667, -0.3368315811172895, 3.049999999999997, -0.4760235003092088, -0.5396667824403032, -0.662495326074998, 0.009905236081422686, 0.30755454783638836, 8.957591240875907, 0.39000000000000057, -0.025315878113621793, -0.11066473373088215, 0.17492478857663762, -4.949999999999999, 0.005070532520530691, -0.025315878113621793, 22.0, 0.07524928207970127, 6.120000000000001, 0.0024702138955383646, -0.9752162025999009, -0.17907934644500578, -0.09498522912654472, -0.0354065139990567, -9.510000000000002, -0.9352629273689992, -0.6219143524589903, -1.4757885907827024, 6.120000000000001, 0.04493423191626178, 1.8799173013465627, 28.12, 5.392454854847603, -1.7547842920885088, 22.09, 0.006621533920776912, 17.3, 3.996007131498393, -7.083479318734799, -1.5545063923299782, -0.004521629994666654, -0.003898103261224861, -4.935255474452575, -0.0026876135132347656, -13.486423357664236, -6.09, -4.73, 3.6400000000000006, -0.0044750263381523325, -0.002397619901245207, -0.5100000000000016, 2.004310134310135, 22.0, -14.65954787996839, 7.8986618004866145, 15.279464720194646, -0.011899915182361553, -0.0898306254517216, -0.1389407647602514, 0.1853655867627939, -0.020466187970260563, 1.400822734720185, 22.0, 1.3900000000000006, 1.2899999999999991, -0.6683602375225828, -8.909635036496361, 0.17492478857663762, -3.76213923346633, -0.8998716230211716, 8.864017602813323, -0.14567559136833985, -7.527761557177623, 0.039165032339482764, -0.0044750263381523325, 2.8360282167342277, -0.00161473151593583, 9.46, 4.32, -0.351569708152379, -0.6657739140815924, -4.973381995133835, 3.0700000000000003, 15.721417681455632, 7.59, -2.581100244498778, -4.460000000000001, -8.440000000000001, 23.29, -2.49, 22.0, 0.22847813559306118, -11.04, 0.004211314364560792, -0.5680937126662009, 1.0832835375248848, -17.479999999999997, -10.119999999999997, -10.58491237987564, -1.3724586874910933, -8.68, -17.03, -0.12810845894808054, -1.6515370690291924, 20.240000000000002, -7.82, -3.5618356940509983, -10.119999999999997, 2.3547613762485895, -0.7829562241705013, -0.19625341334597124, -14.548357063200505, 0.9089912280701746, -0.3734985832071942, -0.043406371967172674, -0.2741001607315745, -0.010000000000001563, -12.89, -0.010000000000001563, 0.5986681465038792, -18.25, 0.014653903977333016, -2.990000000000002, -0.025960379808040557, -0.0498156628451909, 1.4952631884748087, -5.979999999999997, 20.240000000000002, 0.018110494284623968, 0.5552476899552339, -4.813128169046138, 4.508783810935192, -0.009956140350876552, -0.0028847306561026187, -1.0039193486312143, 
-0.37687526403814786, -1.4380121770596226, -19.32, -0.10395160323166763, -0.0012305333372140126, -0.07055538317638366, -0.09100877192982537, -8.600000000000001, -0.009956140350876552, 1.011982451401554, -15.850067988668563, 2.3547613762485895, -0.010000000000001563, 16.79, 12.880000000000003, -0.19297074757220756, -13.11, 
-0.12749712597619123, -0.4912161890648079, 17.71, -5.91067988668555, -1.1001878698561214, -0.010000000000001563, -5.072521788724387, -0.009956140350876552, 23.91795782463928, -0.010000000000001563, -0.3734985832071942, -0.20315430415129399, 0.07525804374967393, -0.23999999999999844, 0.9089912280701746, -3.849285243752714, 
-0.009956140350876552, -0.19625341334597124, 0.06829487498409392, 0.6129977992782685, -17.479999999999997, 0.8037465866540288, -1.0065847051810017, 0.07295245921410043, -0.16214999999999957, 1.4447188438499206, -0.0006351183063522114, 0.0, 1.2551521657376252, 0.0, -0.8500000000000014, 0.6564166366448383, 0.8186777838233237, 0.30756062732550404, -0.7152565760436076, -2.2235971623187734, 0.32520868679728476, -0.22725180660962607, 1.0200000000000031, -2.316488169364881, 0.03655967531771864, 0.17457751356904794, 0.01376851434846671, 0.3791959736039727, 0.5582117747552431, -0.0049875391018439075, 0.0, 0.12385319733214306, -0.22644077660151574, -1.3384727043701767, -0.09674572346676591, -0.019387531002216463, -0.8500000000000014, -2.157719969336318, -0.06822006789735013, 0.0, -0.0006351183063522114, 0.0, 
-0.0012702366127008702, 0.09844773531515472, 2.836240988671472, 9.135808065126206, -1.4274233465655923, 0.45578419071517917, -0.6643821051858518, -0.1404851909865492, -0.06099425717852469, 0.013343408379033583, 0.0, 0.024631511044363563, -0.18173827074659954, -0.04249750312109679, -0.0006351183063522114, 0.020875936329588285, 0.24193548387096797, -0.0013115649867367907, 3.6194931291785215, -0.012309041303519841, 0.0, -12.920000000000002, -0.10591323889663684, 0.002110921789769016, 
4.788220912702521, -0.11910855975205958, -0.005716064757162798, -0.07107499999999956, -0.02712423681420617, 1.0200000000000031, -0.5954851906687777, -0.06817278401996951, 0.3194235628303428, 0.0, -0.0015079196665404027, 0.49454545454545595, 0.3194235628303428, -0.0006351183063486587, -0.04464634831460135, -0.10456891211362684, 0.7568958014988407, 0.7640154044021017, -0.012309041303519841, 0.11929736511919575, 0.05002795588723963, 2.793533850782553, 0.3898751560549316, 0.12375009859085218, -0.91974861885548, 0.0006352757306284929, -0.40441674608651823, 0.48808492446087115, -3.021288393721548, 0.6538977420499847, 0.06535549714070044, -0.22165031986096295, -0.19073155981554635, -0.90058316294121, 0.10887633475170588, -0.41227591043723777, 0.7681714261641943, 3.1237517350883834, 0.8787457661703684, 0.35929073066682093, -1.9055488504492644, -0.3225698837827835, -0.3958959857852449, -0.3643031130064287, -0.17054617267458383, 1.419895265283273, 0.3211025410625865, 
0.8695944973354415, -0.01250916459566298, 0.1661474676361081, -0.8645810237434617, -1.7788735345847968, -2.7627224491203677, 0.3783247959012144, -0.7179366243832845, 0.2679112228046989, -0.24381342599170353, -0.7937871221497304, -0.1331428982677778, 0.8792748373350179, -0.38339311743271764, -0.03952503797635387, -0.16772086624902016, -1.4170718532354627, 0.529170112884465, 1.1851984768373853, 0.13717577338778852, 0.2128815360586671, 1.3286836355949347, -0.4336702790355744, -2.0250109879900933, 0.9447950161859495, -0.710405747493347, -0.6478262173157958, 0.9614578407150773, -0.49398495903410833, 0.285735671424618, 1.9278004326729103, -0.40933137278803855, -0.38339311743271764, 0.019652769733013997, -0.6867880865841229, 0.06065207731123223, -1.6335533010825074, -0.4888399179989449, 0.12097589030381783, -0.32873468272968154, 0.676785656113374, -0.24094213373513185, -1.8386338507554, -0.2834315380559289, 0.622990857429123, 0.07098283308047826, -0.8035221390299938, -0.28430597180626194, 1.555060234719953, 0.4055316913507312, 2.1759396587963593, 0.15179630661921806, 0.06965818774148147, -0.11276753567412356, -1.1895311818400458, 0.8356381964292012, -0.2954361732382509, 1.658587187563917, -1.0005029293369319, -0.4120580373271565, -1.3102656311407088, 0.11924315016313614, 1.4755556146289166, -0.2678703691033739, 4.054170650386556, -0.05405298214581222, -0.5300000000000011, 0.32767608733933784, 1.6135449462085276, 0.38976109215016663, -0.02557418481775997, 0.11984691866628339, 0.7600000000000016, -0.008580784227113725, 1.374365032870056, -0.007961589180673911, 0.011743765629645964, 0.3096482932947282, 
-0.5799999999999983, 0.757925035388439, 2.4172541577481432, 0.00573169680870933, 0.0799999999999983, -0.4893709043250354, -0.22598209515825474, 1.1900000000000013, 0.07607964394346212, 1.775131550909368, 0.8115375462032688, -0.2402720094758486, -0.006876132056969908, 0.059520132032666595, -0.11693677343228259, 0.7800000000000011, 0.8163021796958851, -0.0891166808459225, -0.08865741389092818, 0.33575537475752526, 0.10999036935228546, -0.9615640966001813, 0.46999999999999886, 0.33239045898481834, 1.7069785558849517, -0.38362395420519846, 0.11658350538550266, 0.44748224842774675, 0.1201446979081986, 0.6576310395953335, -0.008580784227113725, -1.2587109053683783, 0.6146975339976777, 0.2588112272977483, -0.00014918625678106423, -0.35999999999999943, 0.22729999522572086, -2.000788500264761, 1.5287991752827566, 0.30517294890476876, 0.6039725318984672, 0.030000000000001137, 0.9200000000000017, -0.9181365430840884, 1.0619812029475817, 2.963350240300759, 0.06144731671533066, -0.629999999999999, -0.7300513405590436, 1.526302179695886, -0.5100000000000016, 0.11714409428697437, 0.5708152801976674, -1.1480106985048195, -0.14943826110346237, -0.17616759023533746, -0.3295987315174358, 0.020000000000003126, -0.03023890784983152, 0.015083181688037683, 1.0440340231672458, -1.0, -1.0486240464249335, 0.05695141054870234, -0.3732825027258784, -0.17573220366498532, -0.3500000000000014, 0.0, 0.009555729913998334, -0.29000000000000004, 0.04999999999999982, -0.9900000000000002, 6.289999999999999, -0.3000000000000007, 0.25, -0.9199999999999999, -0.1200000000000001, 0.8799999999999999, 0.21000000000000085, 0.0, 0.0, 0.8300000000000001, 0.0, 0.0, 0.0, 1.3200000000000003, 0.0, 0.8300000000000001, 0.79, -0.2400000000000002, 6.289999999999999, -0.8399999999999999, 0.4800000000000004, 0.29000000000000004, 0.0, 0.0, -0.9900000000000002, 0.71, 0.0, 0.0, -0.009999999999999787, -0.03000000000000025, 20.79, 0.21000000000000085, -0.17999999999999972, -0.6500000000000004, 0.0, -20.79, 0.0, 0.0, 0.009999999999999787, 0.04999999999999982, 0.0, 0.0, -18.9, 0.0, 1.0099999999999998, -0.05999999999999961, 0.0, 0.9199999999999999, 0.8399999999999999, 0.9699999999999989, 0.15000000000000036, 0.0, -0.2599999999999998, -0.040000000000000036, 0.48999999999999844, 1.0799999999999983, 0.3700000000000001, 0.71, -0.46999999999999975, 0.009999999999999787, 0.21000000000000085, 0.34999999999999964, 0.0, 0.7599999999999998, -0.7300000000000004, 11.29, -0.6300000000000008, -0.009999999999999787, 0.04999999999999982, -0.7200000000000006, 0.0, 0.4800000000000004, -0.8300000000000001, 10.42, -0.17999999999999972, 0.009999999999999787, 0.0, 0.5599999999999996, 0.009999999999999787, 0.6500000000000004, 0.9499999999999993, 0.240000000000002, 3.009999999999998, -0.18999999999999773, -0.07000000000000028, 0.6799999999999997, -0.00999999999999801, 0.490000000000002, 0.4299999999999997, 0.9500000000000028, -0.030000000000001137, -0.7100000000000009, 1.3100000000000023, 0.04999999999999716, -0.46999999999999886, -0.21000000000000085, 0.22999999999999687, -0.21999999999999886, -2.7900000000000063, 3.6899999999999977, 2.0, -0.00999999999999801, 0.11999999999999744, -0.5900000000000034, 0.3100000000000023, 0.7000000000000028, 0.09000000000000341, -1.1199999999999974, 0.030000000000001137, -0.04999999999999716, -1.0499999999999972, 0.21000000000000085, 0.5200000000000031, 1.6799999999999997, -0.28999999999999915, -0.990000000000002, -0.9399999999999977, 0.7700000000000031, -1.0300000000000011, -0.25, 0.6199999999999974, 0.09000000000000341, 0.5200000000000031, -0.3200000000000003, 0.259999999999998, 0.39000000000000057, -0.5200000000000031, -3.019999999999996, 0.6499999999999986, -1.3299999999999983, 0.1599999999999966, -0.7700000000000031, 0.4799999999999969, 0.259999999999998, 0.18999999999999773, -2.8700000000000045, 0.8400000000000034, 0.9599999999999937, -0.00999999999999801, 0.6000000000000014, -0.3200000000000003, 0.28999999999999915, -1.2100000000000009, 0.6099999999999994, -0.14000000000000057, -0.03999999999999915, -1.8499999999999943, -0.6599999999999966, -0.21999999999999886, 0.10999999999999943, -0.39000000000000057, 3.5900000000000034, 0.36999999999999744, 0.46000000000000085, -0.7800000000000011, -0.5, -0.5399999999999991, -0.3200000000000003, 2.3299999999999983, -0.740000000000002, 0.4200000000000017, 2.229999999999997, -0.1599999999999966, -0.6099999999999994, -0.5600000000000023, 0.0, -0.0034159545127838697, -0.07000000000000028, 0.0, -0.009858480979220374, 2.8130866025563517, -0.02281462218171093, -0.16271863242680595, 0.0014934670810617945, -0.008419051996417792, -0.029816209451578146, 0.0, -0.005586516901887606, -4.9875311720626314e-05, 0.0, -8.784773059922202e-05, -2.4937655860313157e-05, 0.004216435320310907, 0.005098129836284215, -0.0021548794492218803, -0.006266118843262447, -1.1946520170590862, -0.05207202160170077, 
-0.008594312138063387, -0.9657531114029005, 1.6799999999999997, -0.00495359803152251, -2.4937655860313157e-05, -0.010849356172093039, -2.536333118740412, -0.0005856515373281468, 0.13572334993191237, -0.0050561447765815615, -0.17128325670385003, 0.1521697407101854, 0.0, -0.010249360613810765, 0.014937485054478117, -4.9875311720626314e-05, -0.006002521715643638, -0.0005856515373281468, -0.009654731457803933, -0.033702046035811506, -0.009130434782610664, -0.07000000000000028, 1.7853199942760707, -0.0028573924122179406, -4.9875311720626314e-05, 0.0, -0.022227523512198033, -0.004786460224714517, -0.32763164820127066, -0.6566789538877673, 0.00023781401268063718, -0.046804521171609714, -0.004907873529127471, 0.01968523101141262, -0.00611184533579312, -0.16128325670384847, -0.00030746705709816524, -0.0006496970418963244, 0.016518603863930892, 0.05600934033662064, 0.048676003020849734, -2.59, 0.0003665860774955121, -0.01687896798230426, 0.0, -0.9780801554280583, 0.0, 0.018881074168794143, -0.9985800312109543, 0.0146809446970817, -0.027180306905375318, -0.0055517704808210055, -0.00030746705709816524, -0.005911252943636924, 0.0, 0.07317361661751409, -0.001296571771066013, -0.0039411533307722735, 0.05912632374858795, 0.0012771392081738497, 2.1471801510068076, 0.023442363716347003, 1.4199125331783264, -0.2618825376580389, 0.992423824348748, 2.3523801931939055, -1.2202564102564395, 0.0, -0.6227278396188218, -0.3537208631923612, 0.2867524838640687, 
0.8654871794871752, 3.0475203061988445, -2.518780143989485, 8.76239972807614, 0.01510827677427784, -1.002887221717014, -0.8967884226858551, 0.07625166054551258, -1.4102564102564372, 0.0, -0.0009259259259266628, -0.25821859517635204, 0.2737246955462389, -0.15161879478986862, -1.1129789902378686, -0.04116798410729672, 0.34725120666150744, 0.12842603878105407, 0.0, -0.03425925925926521, 0.23725765499868956, -0.0018518518518533256, 0.22888893064185112, 2.0774809131543304, -0.03703729014214474, 0.37501114549942827, 1.5399999999999991, 0.022321401805891483, -1.647070201036886, 0.2520787447166448, -0.38263870500912844, -0.35623250614522384, -1.14333700623056, -0.6425442469662048, 0.03016758652465512, -0.6066877462474292, -0.42060133461681914, -2.261229060197202, -1.8114763204676336, -0.044384971277027674, 
-0.0819913021430505, 0.07469418125626248, 0.0, -0.17583174622517106, 1.008257035873985, -0.7855685432543194, -0.5419517343823941, 0.0, -0.3953976439217488, 0.0, 0.12697936005964827, -0.01574074074074261, -2.0953753017961034, -0.5264159816293148, 0.15927112946113908, -0.9964603948778201, -1.6689965910133928, 0.45084534146956834, 0.8731515380562218, 0.02722922728166033, 0.030896269660225073, -0.044810503191921924, 0.05780880068585503, -1.8444529580867126, -0.9182998775496429, 0.0, -0.23780558077390168, -0.3063852565520673, 0.0, -0.027284541341316526, -0.07334499645715908, -1.6398003145786255, -0.41451597576310384, 0.02912696836306239, 0.6500000000000057, -1.9100000000000001, 2.1099999999999994, 6.640000000000001, 0.0, 1.1499999999999986, 0.46999999999999886, 1.1099999999999994, 0.010000000000001563, -1.3500000000000014, -1.9800000000000004, -0.4499999999999993, 0.7100000000000009, 0.3500000000000014, 0.6400000000000006, 5.640000000000001, 3.960000000000001, 0.0, 14.11, 0.0, -1.4899999999999984, 0.629999999999999, 1.8999999999999986, 0.5200000000000031, 18.32, 4.07, -0.9200000000000017, 0.9499999999999993, -1.0300000000000011, -0.870000000000001, -36.629999999999995, 2.3299999999999983, 0.5, -0.05000000000000071, -0.5700000000000003, -7.310000000000002, 1.0100000000000016, 1.0100000000000016, 0.879999999999999, -0.019999999999999574, 1.480000000000004, 12.409999999999997, -0.19000000000000128, 0.9899999999999984, 0.5200000000000031, -0.879999999999999, 16.049999999999997, 1.3000000000000007, -0.03999999999999915, -0.030000000000001137, 0.240000000000002, -1.5199999999999996, 17.32, -0.5200000000000031, 0.0, -1.1400000000000006, -2.460000000000001, 0.620000000000001, -0.46000000000000085, 0.38000000000000256, -0.28999999999999915, 2.969999999999999, 18.32, 0.28999999999999915, 1.5200000000000031, -0.9899999999999984, -0.7100000000000009, -0.5200000000000031, -22.08, -1.0100000000000016, 0.8200000000000003, -2.1199999999999974, -0.8299999999999983, 0.9400000000000013, 7.890000000000001, 0.0, -0.6600000000000001, 0.129999999999999, 0.0, 0.0, 0.38000000000000256, -0.46999999999999886, -0.05000000000000071, -0.030000000000001137, -0.6663625765868613, -0.7740917979300832, -0.5929289954690553, -1.9347514357266977, 0.006975821968934426, -0.6328744529885384, 0.40306145142818117, 0.0, 1.7902100976284174, -1.7802464645874814, -0.04053293314060369, 0.0190124070323332, 0.0, -0.0005313769059842599, 0.9503227365816826, 0.037977733776322964, -0.5206168655487371, 0.0, 0.4200000000000017, -0.45675570707542157, -0.0010437514366703482, -0.48358515673686675, -0.579023923746032, 0.0, 0.3740981511799717, 0.08249749707010601, -0.6364008543557222, 0.3991928363480426, -0.00022151898734179554, -0.0008860759493671821, 0.5212625838760694, -0.6004955680548711, -0.015021220402109847, 0.0680963449140215, 0.3677633840774206, 0.30136329514534843, -0.15096500271845592, -0.0393719972546247, 1.9429541883487111, -16.590000000000003, -0.37369716481417115, -0.10783457519574569, 0.10999999999999943, 4.409999999999997, 0.0, -0.41083241485949706, -0.0005313769059842599, -0.86451318760672, -0.4200000000000017, 0.058104575691711347, -1.0322447788234879, -0.04840252304946979, 0.5060126582278563, -0.00022151898734179554, 0.0, 0.5463411180276978, -0.30945190918935594, -0.0005693816565788268, 4.49940218113729, -18.94608695652174, 0.3663599099049968, 0.0, 0.0, -0.4647516110496124, 2.4071105631964294, -0.6616037135957438, 0.3668977116496208, 0.0, -0.0030284308438677243, 0.8553525518255221, -0.0008860759493671821, -0.363827817165955, -0.00022151898734179554, -0.07294554876176207, 0.3998299260350091, -0.5424595045919567, 0.7715351348237789, 0.0, -0.00022151898734179554, -1.0500000000000007, -0.00022151898734179554, 0.44971853477583856, 0.0, -0.00022151898734179554, 0.16999999999999993, -1.4800000000000004, 9.219999999999999, 0.0, -0.7300000000000004, 0.0, -1.1000000000000014, 0.00999999999999801, 11.349999999999994, -0.3299999999999983, -0.13000000000000078, -0.2400000000000002, 0.7800000000000011, -0.5799999999999983, 0.0, 0.75, 
0.48999999999999844, 0.2699999999999996, 0.0, 0.7800000000000011, 0.33000000000000007, -0.6999999999999993, 0.5600000000000005, 7.0, 0.0, 0.05000000000000071, -1.0, -0.07000000000000028, 0.0, 0.0, 0.6099999999999994, -0.05000000000000071, -1.1999999999999993, 0.7899999999999991, 0.4499999999999993, 0.0, -0.8400000000000034, 6.519999999999996, -0.4900000000000002, -0.9800000000000004, 0.0, -4.02, -0.2699999999999996, 0.0600000000000005, 13.079999999999998, -0.0799999999999983, -0.07000000000000028, 0.6500000000000004, -0.3000000000000007, 10.829999999999998, -0.5399999999999991, 0.0, 0.0, -12.04, -0.6099999999999994, -0.14999999999999858, -0.6500000000000004, 0.0, -0.3699999999999992, -0.4900000000000002, 0.0, 0.0, 10.11, 5.640000000000001, 0.0, -0.21000000000000085, -0.21000000000000085, 0.03999999999999915, 0.46000000000000085, -1.1600000000000001, -1.4800000000000004, 0.00999999999999801, -0.4299999999999997, 7.149999999999999, 0.05000000000000071, 5.640000000000001, 0.0, -0.7899999999999991, -5.939999999999998, 30.53, 0.6500000000000004, -0.03999999999999915, -0.10000000000000142, 0.0, 0.009999999999999787, -0.10999999999999943, -0.6500000000000004, 0.26000000000000156, 0.33999999999999986, 0.0, 0.9299999999999997, -0.08999999999999986, -0.3200000000000003, -0.5899999999999999, 0.39000000000000057, -0.2699999999999996, 0.7800000000000011, 0.0600000000000005, 0.2400000000000002, -0.8399999999999999, -1.9399999999999995, -0.3800000000000008, 1.0, -0.009999999999999787, 0.009999999999999787, 0.0, -0.07000000000000028, -0.7300000000000004, -0.6900000000000004, 0.0, 0.0, 0.0, -0.6799999999999997, -26.0, -0.17999999999999972, -0.009999999999999787, 0.0, 0.0, -0.2599999999999998, 0.9500000000000002, -0.11999999999999922, -6.34, 0.7800000000000011, 0.7599999999999998, 0.0, 0.22000000000000064, 0.0, 0.0, -0.009999999999999787, 0.02999999999999936, 0.0600000000000005, 0.2699999999999996, -0.03000000000000025, 1.6900000000000013, 0.6499999999999986, 0.0, -0.6500000000000004, 0.9500000000000002, 0.0, -0.3200000000000003, -0.04999999999999982, -0.04999999999999982, -0.07000000000000028, -0.04999999999999982, 0.02999999999999936, -0.7599999999999998, -0.3800000000000008, -0.75, -0.009999999999999787, 0.6900000000000004, 0.2400000000000002, 0.09999999999999964, -0.2999999999999998, 0.0, -0.7300000000000004, 0.0, -0.08999999999999986, -0.33000000000000007, 0.0, 1.0, -0.5899999999999999, 0.02999999999999936, 0.0, 0.0, 0.3200000000000003, 0.6799999999999997, -0.3099999999999996, -0.3099999999999996, -4.685503810330225, -0.005105016127798123, 2.6000000000000014, -0.7504560716004569, -0.3067742886239202, -0.35798966025093115, 0.1128595502775589, 12.479999999999997, 0.0, 0.13004786640244248, -0.02274115070204452, -0.0038449612403113065, 0.0, 0.0, 1.7266061444031315, 16.9, 0.0, -0.43555319125568204, -0.00770533446231525, 15.86, 20.277809604043803, -0.7936640157715704, -0.019885158373261547, 0.0, -1.4611007620660459, 0.3346597205460835, -0.21523171997280066, 0.0, 0.16879956921742512, 3.6400000000000006, -5.200000000000003, 0.04907476528548216, 0.034926979900221866, -0.0011007620660450357, -0.15292784208293675, 2.6000000000000014, 7.539999999999999, -0.6414643129483064, -0.5528627773153545, -0.06531262488070944, 0.0, 0.9939444998660942, -6.0417595125789845, -0.6657574145738554, 0.0, -0.02753612261941818, -0.5906176000684571, 0.013437891194154616, -1.1862705089513703, 0.01168622872222258, 0.0, -0.8522463760282761, 3.6400000000000006, -0.06164267569853621, 2.8599999999999994, 0.0, -0.7933045423242753, -3.020000000000003, 
-0.3063037364877168, -0.6977902809324021, -0.6578469375576859, 0.0, -0.8278409988163418, -0.2258853012407407, -6.722968474259108, 0.0003204306975135296, -0.6419449331033427, -1.0507374213428875, -0.8294608247018367, 0.08985713269331086, 3.1199999999999974, 0.6814215370907668, -0.15961035120673728, -0.20025835820655757, 0.0, 0.0, 0.3118827157802997, 0.2111344609951189, -0.5337542311746013, -11.30330228619814, 0.0, -0.5275914579477856, -1.5600000000000023, -0.259999999999998, 0.0, 10.799999999999997, 1.5100000000000016, 0.3200000000000003, 0.0, 0.0, 0.0, -0.1999999999999993, -0.4900000000000002, 0.0, -0.7399999999999984, 0.7800000000000011, 0.0, 0.1899999999999995, -0.3100000000000005, 0.11999999999999922, 0.0, 0.0, 0.0, 0.9400000000000013, -0.019999999999999574, -0.05999999999999872, 47.0, 1.0599999999999987, -1.0, 0.0, 1.0399999999999991, -0.019999999999999574, 2.66, 0.3800000000000008, -0.25, 0.8200000000000003, 0.8999999999999986, -0.1899999999999995, -0.14000000000000057, 0.0, 4.689999999999998, 22.31, 0.0, 0.0, 0.370000000000001, 0.5399999999999991, 1.4800000000000004, -0.6300000000000008, 0.11999999999999922, 0.10999999999999943, 0.0, 0.05000000000000071, -0.17999999999999972, 1.8700000000000045, 1.1899999999999995, 0.0, -0.009999999999999787, 0.0, -1.7800000000000011, -0.7300000000000004, 0.16000000000000014, -0.1700000000000017, 0.3699999999999992, -0.17999999999999972, 0.4399999999999977, -0.5099999999999998, 5.159999999999997, 0.08000000000000007, -0.09999999999999964, -0.5100000000000016, 0.0, 0.10999999999999943, -0.05999999999999872, 0.10999999999999943, -1.6500000000000004, 0.0, 0.14000000000000057, 0.0, 0.2699999999999996, 1.0700000000000003, 0.09999999999999964, 4.680000000000007, -0.7699999999999996, 0.9000000000000004, 0.0, -0.01999999999999602, 0.0, -0.14000000000000057, 20.28, -0.5639554794520514, 0.0, 0.20528056514949178, -0.013879348656631763, 0.07176921668487246, 0.09973251384591109, -0.8094441964283252, -5.200000000000003, 0.5200000000000031, -0.8840589925400852, -0.574484617240465, 0.0, -0.03100761000325747, -0.014623765530082267, 17.159999999999997, 0.09831369097880938, -0.060470503738836, -0.036940327413441665, -0.9457865527179727, -0.037429091416072424, 9.591581455168402, 0.562257312844423, 0.658657232469599, 0.0, -0.10110268350691243, -0.2897749471233393, 0.0, -0.025865185395007373, 1.2167676537290752, -0.04836014145529788, 5.91092336267787, 0.06390023007742052, 19.5, 0.0, 4.578882749068594, 0.0, 0.0, 0.5200000000000031, 0.23364689843217512, 16.9, -0.14072727272727192, 0.5200000000000031, -0.0119651338391904, 0.5200000000000031, -2.1401370602404572, 0.42551538275953504, 0.01728296006958807, -0.05110462950342942, 20.28, 0.011734387924267509, 0.8267392898581107, -0.0932962328767104, 0.0, 0.0, -0.0062247917571713884, 18.979999999999997, -0.02638839926347991, 1.275747406953128, 1.2046933480564288, 0.0, 7.799999999999997, 19.5, -0.028129632637121205, -5.979999999999997, 0.34838785284839346, -18.200000000000003, -0.011251612611934192, -0.060242424242424875, -0.010210436923200561, 0.356422137896125, -0.47778085011717764, 0.6868401286146586, -1.1015255879711994, 0.09973251384591109, 0.09455070724782466, -0.10707253852265275, 0.6658822591175575, -1.1357576208900717, -0.20502038901456032, 0.4459148936170205, -0.11314710558671948, 0.6983463780544028, 0.0]

    diff3 = [0.0, 0.13000000000000078, 0.0, -0.07000000000000028, 0.07000000000000028, 25.109639519359142, -0.019999999999999574, 0.0, 0.0, -0.015861148197618746, -0.7799999999999994, -0.2599999999999998, 0.0, 0.9000000000000004, 8.478558077436581, -0.27999999999999936, 0.0, 2.1000000000000014, 0.0, 0.009999999999999787, 0.0, 2.4499999999999993, 0.02999999999999936, 3.5089185580774327, 3.509999999999998, 0.0, 0.0, -0.28999999999999915, 0.0, 5.579999999999998, -0.08000000000000007, 0.019999999999999574, 0.0, -0.0600000000000005, -0.14000000000000057, 0.14000000000000057, 0.0600000000000005, -1.0915353805073664, 3.5096395193591405, 0.0, 0.14000000000000057, -0.00036048064085747455, 0.0, 0.0, 0.21000000000000085, -0.019999999999999574, 0.0, 21.86747663551401, 0.0, 3.50927903871829, 0.0, -0.05000000000000071, 3.5096395193591405, 0.009999999999999787, -0.02999999999999936, 0.0, -0.6500000000000004, 2.4299999999999997, 2.4292790387182848, 3.509999999999998, 0.0, 3.50927903871829, 4.32, -1.9400000000000013, 0.0, 0.0, 0.0, 0.0, -0.019999999999999574, 5.559999999999999, -0.02999999999999936, 0.0, 0.0, 0.0, 0.2599999999999998, -0.0021628838451306365, 0.08999999999999986, 0.0, 0.0, 0.0, 3.509999999999998, -0.7200000000000006, -1.3200000000000003, -0.28999999999999915, 0.4174729241877344, -1.5605044452746455, -0.1315013568358374, 0.0447039576994257, -0.7768157517459002, 0.6911216002736627, 0.28930559288831326, 0.0, -0.36368783297106866, 0.8148935677033009, 5.015326177119817, 1.1421853502285302, -0.49923665203582956, -0.4189517996945238, -0.8653665093535352, 0.0, 0.7159858221498236, 0.27596537596414983, -0.22065153479727684, 0.4163837638376364, -0.34840156540065514, 0.04608120546284056, -0.21544318510593996, 0.10384723254998107, -2.9012701279615474, -1.2285246839344275, -0.009115895810525743, -0.09620156257169654, 0.021631334986917494, -0.15120251062374912, 0.004310098272371832, -0.5352904856858824, -0.17457474746731627, -5.014661740414976, -1.359255962445074, -0.7344564806094755, 0.8995913133085853, -1.0689832103126253, 0.4461616421551824, -1.069197773119253, -0.6210037564778723, -6.301689730064529, 0.07228749679784485, -2.1854055846086666, -0.00043213933710717356, 0.15017887580158806, 0.0, -0.19776612050818443, -0.01567736240246731, 0.4163837638376364, -0.9804271058965561, -1.2400667129053549, -0.19663174410072237, -0.598074577100757, -0.1116293992091748, 0.994693252212393, 0.5371681137595754, -0.3947507385893587, 0.4281725270415677, -1.0189516577954976, 0.023498054397242285, 0.09038912741570115, -0.7577546940803277, -0.48780856737496947, -0.7928275850652007, -0.5785300517424474, -0.8166550650368247, -0.5004369990949424, 0.4163837638376364, 0.02707233461662284, -1.5302707940975608, 0.06308134603952453, -0.00852704282043426, -0.006177448497656002, -0.26501083041828366, 0.030359648403283757, 0.04367957231663233, -2.165103751646928, -0.6392612408760723, 0.5982093127493151, -2.324754494635542, 0.9274260101070482, 0.413570983380561, -0.606296281115295, 0.6599999999999966, -0.8459933822619785, 0.4593686846758036, -0.007407407407406197, -0.22263753253887408, 0.016015185287226785, 0.11248699194384137, -0.005555555555559977, 0.5353512706886896, 0.014979612871172776, 0.3066330212037709, -0.6867627987325289, -0.09097292431468951, 0.010000000000001563, -0.9170714594250278, 0.0, -1.3177531294674276, 0.604444444444443, 0.004444444444445139, 0.05000000000000071, -0.31325503396952215, 0.0, 3.1370412259939258, -0.28700250057103815, 0.22850039141426848, 0.29999999999999716, 0.624088132261539, -0.1357126685729142, 0.5566948202802262, -0.04629629629630472, 0.04295554692352965, 0.1818352106552652, 4.794444444444444, 0.786296296296296, 0.3537285274764663, -0.004709350004125312, -0.09995374340139307, -0.6973768926669521, 0.0, -0.41868509603003545, -0.6300000000000026, 2.3566200690106083, 0.1243448861261065, 0.35741533187262853, 0.11623982106375408, 0.29500134517083865, 0.0, 1.5888525618255542, 0.0039940485220189, -0.009259259259259522, 0.08999999999999986, 0.8532211454383649, -0.003311189719331864, 0.015181310789567526, 
-0.5, 0.7180407999395779, -0.11477548338290156, 0.6167394650862086, 0.028395061728396342, -0.6606576057786029, -0.6545611253129895, 0.0, 0.45174654433308703, -0.017392708447712124, -1.2830208275300894, 0.17803086150597558, -0.5504984912271329, -0.43811108761547146, 0.0, -0.6650109653839467, 0.05317385302691591, 0.012290165279795318, -0.0031054834531074604, 0.008670527391497984, 1.2875068897428452, 3.047037037037036, -0.119946728513467, -0.026711457485685486, -0.24139777880015956, -0.14677885456163509, 0.024781280147641027, -0.0009901319055032332, -0.6722315150507079, 2.861061254123822, -0.8888274039573858, -1.737072712655836, 0.0, 0.0, -2.861343101343099, 0.17175167566937155, 0.3190591390021389, 0.15506541112376482, -0.2892502372972725, 5.280000000000001, -0.0925656708146434, -0.005372405372408906, 
0.0, 0.003425188479750929, -0.008306640642413043, 0.34167547064537196, -1.0347578992517832, -3.4168580813988285, 0.28933181534478436, -2.5839057823175526, 0.01699469373919804, -0.31336564699397584, -0.14797789107667114, -3.7470136126539586, 0.13091216531358008, 0.25774899252655636, -0.11618612781563975, -0.08480336540883116, 1.5823151622239013, -0.004296943931979769, -0.014237546499256126, -0.48899388956667167, -0.001343101343099562, 0.0013726941752629784, -8.14, 0.013229804502385534, -0.9730857678172917, 10.910026547951666, -0.5339703666499247, -2.6426862026861997, -0.28954218976952717, -0.7206086956521727, 0.23042056444196923, -0.027825613432771235, -0.006958637469598017, -0.001343101343099562, 0.00032353888697755906, 0.22830802567233732, -0.09861490466085421, -0.443484220564514, 0.2699999999999996, -1.3268884341038714, 0.0, 0.4679654120040677, -1.249396602256521, 0.02443478260869547, -0.02176746879423419, -0.032486187845304215, -4.593819990469795, 0.010960000000000747, -0.029093482162522832, 0.0, 0.018812858543922673, 0.015611201686130727, 0.3847659921771278, -0.016935235168999085, 0.0, -0.025518925518927205, -0.0024087591240906647, -5.955703380689661, -5.934368932038836, -5.840746222788928, -2.5722206261701217, 0.06125036769631009, -0.8813431013430986, -2.590137391910293, -0.02234561343277086, -0.004980710642078279, -7.699999999999999, 0.5676510686258105, -4.543429609676256, 0.14841878503273875, -0.8922570803505803, -0.13000000000000078, 0.2895187486631947, -0.00033734939759000326, -0.9200000000000017, -0.014353280719998196, -0.012159152188850442, 0.19442008389421161, -0.0729353059678175, 
0.379999999999999, -0.7607205087048818, -0.6267231999439886, 0.09981823623650499, -0.6379325511595759, -0.3373162242628922, 1.22057204640711, -0.23850613483520178, 0.0, 0.7691334552562452, -0.015292116923049548, 0.99966265060241, 0.030000000000001137, -0.00033734939759000326, -0.30108497581934657, -0.18036130250399296, 0.17131011877126312, 0.8676051948569707, 0.010000000000001563, -0.8724054929925842, -0.06422709809449145, 0.0, -0.00033734939759000326, -0.16375121145667393, 0.0, 0.2564133406233111, 1.8400000000000034, -0.07354221229844526, -2.640432612312811, -0.06863832400206604, -0.00033734939759000326, -0.9311645999349771, 0.053883665692939786, -0.00033734939759000326, 0.5599999999999987, -0.05024733017285854, -0.11182443917699558, 0.0, 0.0, 0.07503354272184026, -0.12179393773751457, 0.08632839157671768, 20.47, 0.0, -0.12926849937436558, -0.2757402477664286, 1.7228539317480305, -0.9537278194747287, -0.10392472974610811, 0.5143281150253642, -0.00033734939759000326, 0.0, -0.07114845513650891, 0.0, 0.030000000000001137, -0.16462018371654707, -0.3050454469194115, -1.2790561657306903, -0.1200898819436027, -0.5833223184764869, -0.22140252587687925, 0.0, 0.053518131733035545, 14.490000000000002, -0.00033734939759000326, -0.00033734939759000326, 0.9200000000000017, 0.0, -0.21401320512935484, 0.9894154811218456, -0.5741547277936938, 0.0, -0.05507866378676951, 0.05000000000000071, -0.022963108970245116, 0.0, 0.1811869107043833, 0.0, -1.0099999999999998, -0.09799013083079444, -0.03277668280741608, 0.0, -0.12482513573938014, -0.037766980752506285, 5.004434296534736, -0.8706709409930173, -0.06372337297951791, -0.004945454545453032, -0.01012296235225918, -0.10734278143976894, 0.0, 0.06351183063511812, -0.45683636363636637, -0.06666666666666643, -0.016482116104865163, -0.02338755970133466, -0.005032534588418791, -0.014264343024222192, -1.292173947951479, 1.5832679874301503, -0.7564757709977163, -0.0030385977449380874, 1.2833962369589802, -0.0028860274449886703, -0.05311591414661798, -0.001213568209044169, -0.004945454545453032, 3.6460065304892675, -0.016188655513909467, -0.08521656923405185, 0.0, 7.294637168844573, -1.4862699875466987, -0.015492732878437998, 0.049801980198017404, 0.0, -14.96, 0.0, -0.3420989750632515, 0.009241359481704237, 
0.0, -0.7573428014607169, -0.25669155372249985, -0.01854856429462881, -0.00515782734682535, 0.0, -0.10223320615830644, -0.47157359355042505, -0.004945454545453032, -0.07975087831363759, -0.5032537455024935, 0.0, -0.11289999999999978, -0.21822091875183514, 1.0076279728656914, 0.009999999999999787, -0.009999999999999787, 0.0, -1.2870545454545486, 0.26064516129032356, 0.0, -0.01303726247537007, 0.0, -0.005444132631551746, 0.008217821782174006, 0.031881188118812354, -0.5512335376807309, -0.1673315870712253, -0.0029115750298096543, 0.0012781954887213232, -0.09045618591153559, -0.014679066447387257, 0.12741719536255403, -0.05750101211599734, -0.023154307861095624, -0.02758349978717245, -0.03598127088310665, -0.26034493771957745, -0.17616457680288722, -0.45683636363636637, -0.24786054321271145, -0.011905743559928794, -0.4298915506491525, -0.45049920415402767, -0.8444880161394863, -0.31905234975933894, -0.05082766473856282, -0.9565796512858853, 0.04336817647452307, 
-0.24631866440326178, 0.0032765009276261026, -0.1487830168576849, 0.28876423914815774, 0.052521076855754245, 0.2875321494472516, -0.3700425671471379, 0.8618401045664754, -0.0032342769022548623, -0.6568722443775883, -0.29560157865612524, 0.8744913712600138, -0.12515051909941377, 0.08903525157609948, 0.42799771972318723, -0.12124011415759384, -0.5183545424011555, 0.8133561588364557, 0.7919492327474202, 2.6783987282875685, -1.022517687497107, 0.6885310226291708, -0.5203643875866888, -0.9964279204804285, -0.2612789423653723, 0.2902697366767484, -0.33013807859327926, -0.05670334652577935, -0.17855152309626732, -0.11194115922732806, -0.005871298957615068, -0.2810237872456156, -0.3974067919280557, 0.00020713401580252366, 0.07012741126133903, 0.8006957316891814, 0.6762675512904508, 0.9497496542264514, -0.2242439223923256, 0.019015981501773638, -0.019094945111710615, 1.1212817865802052, -0.02945086337388858, -0.34449984638686004, -0.0032072753711247515, -0.14609376825085718, -0.013071602427316975, -0.10838634949016068, 1.3171066531099598, -0.09106259735104061, 0.3089755169534598, -0.3488894539182894, -0.0021086981663245297, -2.70392091710319, 0.09561306090868982, -0.15191287415285615, 0.05643468421762776, 0.017009564459876003, 2.9421393465388306, -0.02256502796891624, 0.8053991515186496, -0.7365498682045271, -0.3926281744072426, -0.26443663220874214, -0.0027353025124767782, -0.3962264878135784, 0.3667442946233095, -0.07734239272900112, -0.002114583333337805, 0.770280122505504, -0.8671006181251286, 0.27995934045073767, 0.752293251958065, -0.022257219626297342, -1.8371573148149025, -0.7568411831563182, 0.010000000000001563, 0.4152204654370273, -0.8144039734111033, -0.0598007027852816, -0.04393505868437586, -0.5520314547837479, 0.26065950407726035, 0.31143775623332814, -0.20625550303604712, 0.023112208163933445, -0.2926482048527035, -0.37950761215772744, -0.0799999999999983, -0.13077091595447143, 0.1155755119782036, -0.6573853543274808, 0.46000000000000085, -0.5506569488918842, 0.09302500323958185, 0.24969195664574428, 0.08280849831257342, 0.03080242259587962, 0.17419884037096978, 
-0.09362779108863961, 0.5536592392602948, 0.023841772867822186, 1.4185411512091832, -0.019999999999999574, 0.3000000000000007, -0.1389719199791415, 0.7524441354408253, -0.08826264479316137, 0.019074760955273717, -0.5500000000000007, 0.27977596557105855, -0.05000000000000071, -0.004893642241498242, 0.9100000000000001, 0.018455565908833904, 0.35868655414246575, -0.17859493893052658, 0.8230392269061646, 0.9546563641879118, -0.24513364846064079, -0.02833237140261957, 0.08230054809865095, 0.7655197669944869, 0.023890784982935287, 0.4732235444174684, -0.10502086823005641, 0.3490161535537055, 0.9200000000000017, 0.05983632724804977, 0.058717574642464854, 1.4383871295320567, -0.8620314547837467, 0.5, -0.02214991803468891, 0.7274244488439798, 0.44666285297060426, -1.2202716687359398, -1.8796974239763493, 0.9800000000000004, -0.4099999999999966, -0.873030318241117, -1.7609411764705882, 0.05558058633446805, -0.014965260170171746, 0.44666285297060426, 0.15989731888191727, -0.5986039787743671, 0.211190083529587, 0.24425444534806218, -0.28085384972709093, 0.011978021978020337, -1.1442475750224084, 0.2481775108217832, -0.2586504898176969, -0.7933371470293942, -0.2627289142711646, -0.4112658578243362, 0.2993604241468475, 0.6300000000000026, -0.015335276967928024, 0.07000000000000028, 0.0, -0.21000000000000085, 0.0, 1.8099999999999987, 0.0, -0.71, 0.0, -1.2600000000000016, 0.0, 0.20999999999999996, 0.07000000000000028, 0.0, 1.8099999999999987, 0.1299999999999999, -0.08000000000000007, 0.0, -0.03000000000000025, -0.02999999999999936, 0.0, 0.0, 4.200000000000003, -0.019999999999999574, -0.05999999999999961, -0.03000000000000025, -0.03000000000000025, 0.0, -0.019999999999999574, 0.22000000000000064, 0.0, 0.0, -0.4200000000000017, 0.07000000000000028, 8.02, 5.219999999999999, -0.28000000000000025, 0.3700000000000001, 0.14000000000000057, 13.02, 0.0, 0.0, 0.0, -0.04999999999999982, -0.019999999999999574, 0.0, 4.619999999999997, 0.0, -0.019999999999999574, 0.03000000000000025, 2.1000000000000014, -0.0600000000000005, 0.0, 0.0, -0.05999999999999961, -0.4800000000000004, 0.0, 0.0, 0.0, 0.10999999999999943, -0.1200000000000001, -4.699999999999999, -0.29000000000000004, 0.0, -0.5800000000000001, -7.77, 0.0, 0.0, 0.0, 0.009999999999999787, -0.019999999999999574, 0.9100000000000001, 0.0, 0.08000000000000007, -0.019999999999999574, 8.92, 0.20999999999999996, -0.5299999999999994, -0.21000000000000085, 0.4299999999999997, -0.5099999999999998, 0.0, -0.3700000000000001, -0.5300000000000002, -0.04999999999999982, -0.9799999999999969, 0.7099999999999937, 0.7099999999999937, 0.8900000000000006, 1.0600000000000023, 0.6400000000000006, -0.7700000000000031, 1.0200000000000031, 1.6800000000000068, -0.5399999999999991, -0.10999999999999943, 0.09000000000000341, 0.5600000000000023, -1.3900000000000006, 0.1599999999999966, 1.0700000000000003, 0.8599999999999994, -0.4799999999999969, 0.1700000000000017, 0.1599999999999966, -0.6099999999999994, 0.5900000000000034, 0.4299999999999997, 1.5900000000000034, -1.3299999999999983, 0.060000000000002274, 0.259999999999998, 0.6300000000000026, 0.020000000000003126, 0.5900000000000034, 0.5200000000000031, -0.10000000000000142, -0.17999999999999972, 0.21000000000000085, 0.21000000000000085, 0.22999999999999687, 0.8999999999999986, 0.259999999999998, -3.1099999999999994, 0.14000000000000057, -0.9299999999999997, -0.990000000000002, 0.5700000000000003, -0.990000000000002, -0.39000000000000057, 1.9699999999999989, 0.04999999999999716, -3.019999999999996, -0.9699999999999989, 0.9299999999999997, 0.36999999999999744, 0.6400000000000006, 0.3299999999999983, 0.020000000000003126, 0.8900000000000006, 0.4299999999999997, 1.0300000000000011, 0.7899999999999991, -0.490000000000002, -0.7299999999999969, 0.6599999999999966, -0.9699999999999989, 0.37999999999999545, 1.4699999999999989, 1.0399999999999991, -0.03999999999999915, -1.8499999999999943, 0.6300000000000026, 0.5799999999999983, -0.00999999999999801, 0.03999999999999915, 0.75, 0.11999999999999744, 0.7899999999999991, 0.13000000000000256, 
0.240000000000002, -0.39000000000000057, 1.0900000000000034, -0.7299999999999969, 0.3100000000000023, 0.5900000000000034, 0.030000000000001137, -0.38000000000000256, 0.09000000000000341, 1.271460164087145, -0.5038965143702221, 0.20863344242527493, 0.0054087179464188395, -0.014846547314578018, -0.0032131128313404034, 0.0, 0.0, 0.0014641288433381305, 0.017054577131261794, -0.027104674604674805, -0.011822176423383013, -0.007065980611732137, -0.008691501862523765, -0.004552429667519675, 0.0, 0.0, -0.01696592185668422, -0.004434058067785074, 0.0, -0.39647003329633, -0.007401321108748249, 0.05098116822091825, 0.0, 0.03545260982106413, 0.10713170551239948, -0.005385841953356341, -0.007391304347828864, -0.007391304347828864, 0.04209105393860657, -0.0007431725980486803, -0.006799595034834738, -0.023449076367537458, 0.0026644189044944255, 0.45189943464970916, 0.003774600424520713, 0.005312511996519831, 0.0, 1.2884834896893747, -4.969999999999999, 0.004907648935365216, 0.0, 0.02050359779256361, 1.3023780336000756, 0.06105508401046311, -0.00316147932527322, 0.0, -0.011748145930690512, 0.2816176470588214, -0.4556590602698556, 0.0, -0.5915865456583838, -0.8937630327035357, 0.0, 0.0, -1.027406766627136, -0.016915673454117375, 0.0, -0.00945012787724231, -0.00022187179111199384, 0.005146923852137775, -0.004621683051486691, -0.0024936061381080066, -0.11381603398525009, 0.0, 0.0, 0.0, 0.0, 0.04410542371409676, 0.0, -0.009141519477554816, -0.0017391304347817993, 0.0, -0.014271755034253708, -0.00022187179111199384, -0.01528132992327258, -0.018212816704663126, 0.14000000000000057, 0.0, 0.0, -0.007812746265154047, -0.0039060734011489018, 0.0, 1.962756105045199, -0.09408092916916289, -0.07770831542796586, -0.6504871045333225, -0.24340159423553231, -0.21869839972930905, 0.4953897320763634, -0.022045414283320852, -5.224030571518975, -1.5962233457357549, -2.4722254614190735, 0.02990975881231961, -0.2671072631555358, -0.06957264957264897, 0.4677946936306725, -0.3299698841645178, -0.9014511017268791, 0.2577895594243387, -0.36242709177704135, -0.15410336335447639, 1.1000000000000014, 0.4399999999999977, -0.09277244744666469, -0.3497779780889001, 0.6599999999999966, 2.3277750901287213, 0.09865848202792726, -0.104730534879101, -0.2077949601050051, 7.123650348463443, -2.2502020085922823, 0.21973878445636075, -3.0026395567130812, 0.530259891177181, -0.4674500316601602, -0.9777650769034114, -0.010376040478384141, 0.21013406863049866, 0.4399999999999977, -0.2374047397030754, 0.4399999999999977, 0.8800000000000026, -0.6532045670709614, 0.7800000000000011, -0.4257654404272113, -0.3223039024997707, -1.8782165390817127, 0.20079658605974515, 0.7446415643882549, 0.18087891540879042, -0.2885560900798936, 0.7014824355398872, -1.1001553179938277, 0.0, -0.09110676235596671, 0.0, -0.09266547214430076, 0.4399999999999977, 0.4449186975784549, 0.09259259259259167, -0.21362488757294074, -0.18283194723274399, -0.4400000000000013, 0.0, 0.5055668500895756, 0.11092436974789877, -0.7006249879304534, -0.0979919018757549, 0.4399999999999977, -0.35698059853363695, -0.4279413362284039, -0.04791225168399471, 0.06360778562740421, 0.6599999999999966, -1.0209557383820105, 0.17428797277662156, -1.1530554495558718, 0.028019957521554062, 0.5414420633085086, -0.2108265561570537, 0.2946711074104904, 1.4041345283404425, 0.0, -9.68, -0.22292856791683135, 0.3500000000000014, 1.0300000000000011, 0.6499999999999986, -0.5499999999999972, 0.129999999999999, -20.939999999999998, -0.23000000000000043, -0.060000000000002274, 29.28, 0.19000000000000128, -0.010000000000001563, 0.0, -0.8100000000000023, 0.7100000000000009, -12.100000000000001, -0.23000000000000043, 1.1599999999999966, 0.28999999999999915, -0.009999999999999787, -0.0799999999999983, 0.060000000000002274, 0.030000000000001137, 0.7600000000000016, -0.23000000000000043, 0.22999999999999687, -0.4499999999999993, 0.010000000000001563, 0.4900000000000002, 0.0, -0.25, -0.16000000000000014, -14.43, -23.380000000000003, -0.019999999999999574, 0.6899999999999977, 0.21000000000000085, -0.25, -0.2699999999999996, 0.9299999999999997, -0.010000000000001563, -0.08999999999999986, 0.03999999999999915, 0.09000000000000341, 0.4299999999999997, 0.18999999999999773, -0.5100000000000016, -0.7600000000000016, -0.03999999999999915, 0.39000000000000057, 0.0, 0.9600000000000009, -0.07000000000000028, 0.21999999999999886, 0.03999999999999915, -0.6999999999999993, 0.75, 0.5600000000000023, -0.02999999999999936, 0.0, -0.21000000000000085, 0.3000000000000007, 0.07000000000000028, 0.5399999999999991, -0.07000000000000028, 0.23999999999999844, -0.009999999999999787, -0.010000000000005116, -0.07000000000000028, 0.3999999999999986, 0.0, -0.01999999999999602, -0.7000000000000028, -0.04999999999999716, -0.009999999999999787, -0.46999999999999886, -0.02999999999999936, -0.05000000000000071, -0.8299999999999983, 0.7999999999999972, 0.0799999999999983, 1.1099999999999994, 0.019999999999999574, 0.019999999999999574, 0.09000000000000341, -1.816098366709289, -0.09796501812204284, 0.40723996474353186, -0.14418500770000087, 0.0, -0.7992874549558522, 0.0, 0.3649842271293373, -2.1324844166467294, 0.10000000000000142, 0.0, -0.01289312795709563, 0.7928672375834509, -2.148839716152999, -0.37085858752550394, 0.0, -0.059262982568888845, -0.11173370477168909, -0.009130158409281108, -0.08544730824774938, 0.9499999999999993, 0.17036594294743246, -0.03138512570296825, 2.245509444018463, 0.0, 0.05247072281909837, 0.0, -0.03139240506329344, 1.1106297809241337, 0.3365063098465697, 0.0, -7.350000000000001, 0.16167035992506662, 0.3200000000000003, 0.0, 0.20783018633748007, -0.002877054347264263, -8.036086956521736, 0.0, 0.2194245493834046, -0.07223839595000214, 0.2072096723013388, -0.04624749121274796, 0.10000000000000142, -0.00880988874501476, -0.05886807578415265, -0.27845889660483003, -0.04489533368643528, 0.0, 0.0, -1.0345157842038812, -0.1009488737688482, -0.09409477946015343, 0.41000000000000014, -0.10713106709563647, 0.0113666863959061, 1.4699999999999989, 1.846763205395293, 0.021219485195370424, 0.22391362422373007, 1.8857491433454001, -1.0794416959285353, 4.809087643280829, 0.8914328770270714, 0.0, 0.15571089478054567, 0.0, -0.013821044170406083, -3.7510358253393576, 0.1673828978929297, 5.244787818544308, -0.7977056106893503, 3.9138253842696873, -0.19935353999846406, 0.2195455856132824, 0.0, 0.0, 0.12215189873417742, 0.4114494379208864, 11.340000000000003, 0.09189417655845666, -0.034912500485464903, -0.45681923888371756, 
3.5700000000000003, -0.5600000000000023, 0.0, -0.019999999999999574, 0.25, 0.25, 0.0, 0.41999999999999993, 0.030000000000001137, 0.0, 0.0, 0.6899999999999995, 0.0, -0.129999999999999, -0.75, 0.01999999999999602, 1.009999999999998, -0.129999999999999, -0.25, -7.659999999999997, 0.5999999999999996, 0.5099999999999998, -0.05000000000000071, 0.05000000000000071, 0.01999999999999602, 0.0, -0.129999999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4800000000000004, 0.0, -1.75, 0.05000000000000071, 0.5100000000000016, 0.21000000000000085, -2.6099999999999994, -15.36, 0.0, 0.6300000000000026, -0.48999999999999844, 0.21000000000000085, 0.0, -0.2400000000000002, -0.02999999999999936, -0.4299999999999997, -0.019999999999999574, 9.189999999999998, 0.0, 0.030000000000001137, 0.0, -0.8900000000000006, -0.5700000000000003, 0.0, 5.719999999999999, 0.2699999999999996, -0.21000000000000085, 6.240000000000002, 0.0, 0.03999999999999915, 0.23999999999999844, 0.0, 0.9100000000000001, 0.030000000000001137, -0.46000000000000085, 0.05000000000000071, -0.5099999999999998, 0.010000000000001563, 0.05999999999999872, 0.0, 0.019999999999999574, -18.42, 0.09999999999999432, 0.0, 0.39000000000000057, 0.5600000000000023, 8.04, 0.0, 0.9900000000000002, -0.019999999999999574, -0.03999999999999915, 0.10999999999999943, 1.7299999999999969, 0.0, 0.0, -0.7899999999999991, 0.0, 7.469999999999999, 0.0, 0.1899999999999995, 0.0, 1.3099999999999987, -0.5800000000000001, -0.20000000000000018, 0.0, 0.8800000000000008, 0.21000000000000085, 0.0, -0.05000000000000071, 21.32, 0.41999999999999993, -0.21000000000000085, 0.0, 0.14000000000000057, 0.29000000000000004, 0.13000000000000078, 0.15000000000000036, 0.009999999999999787, -0.02999999999999936, -0.0600000000000005, -0.5899999999999999, 0.0, -0.09999999999999964, -0.040000000000000036, 0.0, -0.02999999999999936, 0.0, 7.799999999999997, -0.03999999999999915, 0.7800000000000011, -0.08000000000000007, -0.15000000000000036, 0.0, 0.0, 0.0, -0.03999999999999915, -0.0600000000000005, 0.4800000000000004, 4.420000000000002, -0.1900000000000004, 6.759999999999998, 0.0, 0.0, 0.17999999999999972, -0.03999999999999915, 0.5500000000000007, 0.03999999999999915, -0.08999999999999986, 0.2400000000000002, -0.3100000000000005, 0.03999999999999915, 0.0, 0.3600000000000003, -0.2400000000000002, -1.25, 0.0, -0.09999999999999964, 0.0, -0.08000000000000007, 0.0, 3.1199999999999974, 0.17999999999999972, 0.0, 0.17999999999999972, -0.6899999999999995, 0.8900000000000006, 2.719999999999999, 0.0, 0.3100000000000005, -0.03999999999999915, -0.009999999999999787, 0.2699999999999996, -0.13999999999999968, -0.02999999999999936, 0.9800000000000004, 0.0, 2.8599999999999994, -0.055682606929178036, 1.8599999999999994, -0.598770003869312, 0.0, -0.013631578947354228, -0.1872730343369291, 0.749178662150733, -0.006180469151571444, 0.623239163113432, 0.40833057902567305, -0.006604572396270214, 0.12006771991296183, 0.0, -0.21761113890413242, -0.1301122975177318, -0.3807746975450268, 0.0, 0.0, 1.8200000000000003, -0.06494496189667132, -0.02922911129769723, -0.06477357164172304, -0.042183792443497126, 0.6879856058342746, 0.16804354146980316, 0.0, 0.0, -6.026193070218607, 0.0, 0.0, -3.0837360813804473, 0.2315809845582404, -0.060650971949117505, -0.33291280831983805, 0.3778529084774469, 0.4762336554092279, 0.0, -0.3236755730905685, 0.05154813954275106, -0.0691916470968792, -0.0022015241320900714, 0.08635010131360588, 1.158357324301459, 2.0459549633492813, 0.0, 1.8200000000000003, -0.023420178878728493, -0.0019719024933060325, -1.3913004120471726, -1.0019292014794878, -0.005503810330225178, 0.5979005882764987, 1.8200000000000003, 0.7974191639659978, 0.23609404657059763, 0.167607236698311, -0.22761599372203456, -0.018228817119034346, -0.005008261975371298, 0.696920912360202, 0.6715942967418123, -0.10051000992369197, 3.9454901909305953, 0.630382175529693, 0.21663940922838165, -0.21354659886387672, 1.8200000000000003, -0.2215080375516223, 0.556179661618005, 0.20693121289271232, 1.8200000000000003, -1.501702085687338, -0.3370321743341389, -0.3780063566140939, 0.0, -6.349172580161635, 0.0, -0.02936314008790042, -2.7181232631648093, 0.11150141227370192, 0.32312943274252603, 1.1454778719958512, 0.0, -0.06994498548087869, -0.8000000000000007, 0.8399999999999999, -1.2699999999999996, 0.010000000000001563, 0.6500000000000004, 1.0700000000000003, -0.02999999999999936, 0.7399999999999984, -0.02999999999999936, 0.02999999999999936, 0.0, 0.2400000000000002, -0.620000000000001, -8.46, -0.120000000000001, 0.4299999999999997, -0.8200000000000003, 0.1999999999999993, -8.46, 0.02999999999999936, -0.07000000000000028, -0.02999999999999936, 1.8299999999999983, -0.23000000000000043, 0.3000000000000007, 0.019999999999999574, 0.0, 0.08000000000000007, 0.0, 0.0, 10.799999999999997, 0.5200000000000031, -6.450000000000003, 0.9399999999999977, 0.0, 0.14000000000000057, 0.16999999999999993, -0.03999999999999915, 0.2699999999999996, -0.8299999999999983, -0.5999999999999996, 0.0, 0.03999999999999915, 1.4099999999999966, -0.05999999999999872, 0.9100000000000001, 0.3100000000000005, 0.08999999999999986, -0.02999999999999936, 0.02999999999999936, -8.46, -0.1899999999999995, 0.08999999999999986, -0.009999999999999787, -0.05999999999999872, 0.46999999999999886, -0.11999999999999922, -0.48999999999999844, 0.3200000000000003, 0.46999999999999886, -1.0300000000000011, 0.0, -0.16999999999999993, 6.109999999999999, 0.46999999999999886, -0.07000000000000028, 0.46999999999999886, 0.21000000000000085, 0.620000000000001, -0.02999999999999936, -0.6900000000000013, 0.3100000000000005, -0.019999999999999574, 0.46999999999999886, 1.4699999999999989, 0.0, -0.02999999999999936, -0.009999999999999787, -0.6499999999999986, 19.17, 0.019999999999999574, -0.5300000000000011, -0.14000000000000057, 0.03999999999999915, 0.9647426165908737, 4.172844680152711, 10.659999999999997, 0.20572402421329095, 9.880000000000003, -15.340000000000003, 5.968105360443619, 0.007957446808511293, 0.0, -0.0751757396235817, 0.259999999999998, 0.0, -0.5024464550788004, 0.48562226078959725, -0.12198919837833877, 0.0, -0.4317018046868668, 1.2315389163900594, 2.834052814228574, 
-0.46871291065815335, 9.880000000000003, -0.18486840951545958, 0.1700954626245199, -0.0032392320212260017, 0.39421512568857864, -0.24197530159449876, -0.024301798514555273, -0.14729019151315903, 0.0, -0.5200000000000031, 0.9335076210241606, 0.259999999999998, 0.044797688763132726, 0.03534256826718973, 0.259999999999998, 1.860198219108419, -0.5155744680850987, 0.723340321097476, 0.05037876547404707, 1.0399999999999991, 0.259999999999998, 0.2470178144062558, -0.08840409038700514, 1.5600000000000023, -0.7432666202163887, -3.8999999999999986, 0.7800000000000011, 0.044519144510571707, -0.0030029498316110903, 0.08741332673140612, 0.5408209454215829, 0.09317742242799021, 0.0466934672203827, -5.679463955637711, 0.0, -0.17457693749470593, -0.005870839860495636, 0.22728216674422974, -1.385184928700653, -0.001489361702116554, 0.32891866913123735, -0.3919741066581697, 0.3409745018983763, 0.08183546689896204, -0.21528413703904015, 0.13945658704613528, -8.579999999999998, -0.9602030020402204, 0.011845491307028055, 0.0, 0.0, 0.259999999999998, -0.5500000000000007, -2.3400000000000034, 0.0762176968367374, -0.8979269270657486, 0.0, -0.00705772146054251, 0.4841857255266504, -0.009605137355086057, -0.2369129013360407, 0.0, -0.3263064748148281, -0.4754390375513555]
    
    diff4 = [-0.09999999999999964, -0.7100000000000009, 0.009999999999999787, 0.0, -1.5399999999999991, -0.019999999999999574, -0.33999999999999986, 0.0, 0.0, 4.049279038718289, 0.0, 0.0, 0.11999999999999922, 0.5399999999999991, 0.27999999999999936, 0.009999999999999787, 0.3699999999999992, -0.009999999999999787, -0.6600000000000001, 
-0.01477970627505698, -0.5700000000000003, 0.2700000000000031, 0.0, 0.019999999999999574, 0.0, 0.019999999999999574, 0.0, 6.209279038718286, -1.3999999999999986, 
-0.0600000000000005, -0.40000000000000036, -0.009999999999999787, 5.129279038718288, 0.0, -0.08999999999999986, 0.09999999999999964, -0.019999999999999574, 0.3000000000000007, -0.08000000000000007, -0.006582109479332132, 0.8999999999999986, 0.0, 0.009999999999999787, 1.4299999999999997, -0.019999999999999574, 0.16999999999999993, 1.25, 0.009999999999999787, 0.2700000000000031, 0.10999999999999943, 0.0, 0.019999999999999574, 0.17999999999999972, -0.019999999999999574, -0.019999999999999574, 0.0, -0.07000000000000028, -0.009999999999999787, -0.27288384512683805, -0.08000000000000007, -0.019999999999999574, -0.08999999999999986, -0.7799999999999994, -0.15000000000000036, 0.08999999999999986, 0.02999999999999936, -0.3100000000000005, 0.0, 0.0, -0.3200000000000003, 0.0, 0.05000000000000071, 0.0, -0.00333778371163973, 0.0, -0.2599999999999998, -0.00036048064085747455, 0.0, 0.02999999999999936, -0.009999999999999787, -0.02999999999999936, 0.009999999999999787, 0.0, 0.08999999999999986, -0.23532479689407992, -0.013299101737294095, -0.446427796355799, 0.2246010454830607, 0.574413615715347, -0.06994542169707074, 1.223259745993822, -2.3250816003443155, 4.454879754817432, -0.44440309080266616, -0.0338270361338342, -0.9520676302254856, -0.017221197855029402, 1.0939072086306982, -0.6321527540550598, 1.223744544715771, 0.3979722952407272, -3.8492893482157093, 0.014220361972620665, -0.27410437657580466, 0.3979722952407272, 13.796363824577753, 0.10124262417059882, -0.4698796334357205, -2.426013364630423, -1.3318336910542534, -0.22662123852042626, 1.223259745993822, -1.420559122173378, 0.09936247739332416, 0.06608788341121752, 0.0039168989264659615, -0.012423292827545396, 6.117319207652756, 0.4029425656092833, 15.154353444990688, -1.494841660823175, -0.00962642125640123, -1.1934920247987062, 0.3979722952407272, 1.3722489589446312, 0.7180639166369449, -0.11880232112516609, -0.014707206242054127, -0.7280542815225424, -1.019090687673767, 0.10573023298734796, 0.24875229525837383, -0.0523555173240382, -0.04801125630859637, 1.0156029881038222, 6.0410442076398585, 0.029081101119516006, 1.1369216626201712, 0.12917069992317565, -0.0023834559483155715, -0.35278349753038185, -1.134336042874864, -0.4658896953199143, 0.11601073249022598, -0.7757168858294374, 
0.2672732895210004, -1.7825937781801997, -11.850306715318034, -0.4316027925420407, -0.042731432461863506, -0.9592991500357044, 1.223259745993822, -0.9019437567685866, 0.2034259018483091, -0.1658893468371918, -0.7682976505115509, -0.28835089715138196, -1.0286691583783956, -0.1948048194589127, 0.49561124971068793, -0.36719925911299, 0.18019037068199317, 0.06688129177890012, -0.002838616789485826, -4.385989105438149, 0.06734015003901916, 0.02893620619707704, -0.9192961404362343, -0.8604142003885062, -0.3600173596519589, -0.6575707249293306, -0.030648031649297636, 0.11863082937116509, -0.9688888888888982, 10.1377322850934, 0.01489421632375798, 
-0.41044937247597524, -0.8198370381177646, -0.0834360600491344, 0.2303419867151888, -0.7764663344034801, -0.3672728564627352, -0.3400000000000034, -0.019285265895433668, 2.5407407407407376, -0.03905166766549506, -0.5842400050528518, -1.5727423591045149, 0.20000000000000284, -0.24086107416202296, -3.1107082230639858, -0.16599853644417806, -0.002714349073485778, 0.027946330366969363, -0.013726410152677282, 0.20000000000000284, 0.004185063827121027, -0.86711609710642, 0.11270188827503702, -0.006609272710967673, 0.36189555708705434, 11.58296296296296, 0.0, 1.5669459489297068, 0.2693248532289676, 0.3596038893596063, 0.7972832473485116, -0.01401668011837387, -0.03365203698975172, -0.7530390689736297, -0.43728264654468774, -0.5676737399926104, 0.0, 0.021084626783830274, 0.002868698074607323, 0.8782881106510967, 2.2660122948378696, 0.7850270296567228, -0.8667741693143647, -0.3120058862651902, -0.36999999999999744, 3.301948677227969, 0.0040974389857852955, -0.10328125332478866, -0.21506386861313942, 0.11999999999999744, 7.124074074074066, -0.006975133574153247, 0.38827088285776057, -0.1676868946971677, -0.004868643702584663, 
-0.5007498178673444, 0.016517203240368872, 0.11060532687652369, 28.456666666666656, 0.3369985355052876, -0.05954177734494337, -0.0005436995840977232, 9.438512384239466, 1.137527874068084, -0.35999999999999943, -0.0148148148148195, 0.0041340257581605755, 1.5082265049503327, -0.5051527874922161, 6.479407881323004, -0.22639723548756763, 0.5041801829373478, 0.0, 1.5350328386143364, -0.9607254554605138, -0.8378973287041216, -0.07379639841224517, -0.03473801913920482, 0.03741177660145567, 0.21999999999999886, 0.1577105764905049, -0.06024490760744605, 2.8335474167777175, 14.354051149547546, -0.25769880906646137, -0.21024753270394214, 0.0068619559599998325, -0.08222399986972562, -0.02798902320629182, 0.0019816192169734848, -0.1003862660944197, -0.07883129418694423, -1.1688597510039305, 0.0, -0.14995009962759553, 2.57578938572307, 0.1037997698124844, -0.01258068162844772, -0.8373563495590606, 0.010935845307395553, -1.369978174200753, 0.00658041851239588, -0.014194587393351554, 0.21999999999999886, -2.3589353038240493, 0.07115955036953636, -0.07293398533007434, 0.0835781411360852, 0.0, -0.0010705596107065674, -0.1976522547190367, 1.9799999999999969, -0.4221430900925327, 0.21999999999999886, -0.07165167341216971, -0.11107602493399682, -0.10354553357664109, 0.17910183026011417, 2.0850979960543476, -2.527184677941106, 1.4075325079482184, 0.21999999999999886, 5.719999999999999, -0.07879198652602781, 0.14990072312363267, 0.9354810300370637, 0.027972972972969856, -0.21574393660878322, 0.01474751869500679, -0.4817779608531918, 0.21999999999999886, 0.30351351351350964, 0.21999999999999886, -0.4028888055892583, 
0.08471334315408008, -0.11569909336627937, 0.01694911926900211, 1.1865768864701174, -0.6018417472527648, 0.0025451207465803094, -10.340000000000003, 0.36179931828215395, 2.3894366087350978, -0.08222399986972562, 0.0, 0.09844044283428133, 0.24580947293362243, -0.005923401092930547, -0.12208669366584068, -0.19528599915451927, 2.771455312094556, 0.0, -0.10962736244712978, 7.094006801624321, -0.25924590564849126, -1.683467628695249, -0.6476657716981826, 0.21999999999999886, 0.018175675675674796, -0.40547445255474734, -0.037578843563942144, -0.002219783381388396, -0.01907890909500054, -0.5207568216807239, -1.5480431369127778, -0.15940868280309672, 0.9042230085321563, 0.0, -0.8404458675125426, 0.0, 0.0, 0.0, 0.0, -0.6499999999999986, -0.05000000000000071, 0.0, 0.1696599515938324, 0.0, -0.3299999999999983, 0.05000000000000071, -0.07546040537564735, -0.3299999999999983, 0.015486102968928606, -0.09333229694766487, -0.855325255130694, -0.3200000000000003, -0.04328495894788986, -0.370000000000001, 0.0, 0.0, 0.9876777264126027, 0.0, -0.0007018684367352535, -0.9292153589315575, -0.36131481614135197, -23.0, 0.5282830642498944, -0.08330349587479446, 0.49064537183809165, -0.00011701608971037558, -0.3200000000000003, -0.028342467617946454, 0.6247504206393657, -0.01704187284491354, 0.03999999999999915, -0.14999999999999858, -0.06820917913039537, 0.20145742192794458, 0.6336990072322592, -0.022071036271650968, 0.0, 0.0, -0.39230107761172306, 0.04686284756763115, -0.020657224031246813, 0.10194712399205841, -4.1423038397328895, 0.0, 0.0, -0.4830990943286757, -0.24156336687396518, -0.02525216274695552, -0.3200000000000003, -0.17445235164755601, 0.4725647148464027, -0.25131681514474025, -19.090000000000003, 0.0, -0.03512186198073408, 0.10124267410464327, -0.6418545161977107, 
0.06553848186504396, 0.9720319971936959, -0.43126439110345416, -0.26916666666664213, 0.96592036889799, 1.763545134338223, 0.987014409527255, -0.20515074781046394, -0.3085373917541929, 0.35825399979664496, -0.11799606186527001, -0.07864476569677592, -0.029308068704750667, -0.02349937733499985, -0.010026400996263618, -0.5911681195516856, -0.022742569972766802, -0.7365965025218477, -0.017894876852566632, 0.0, -0.0012999999999987466, -0.006850809464509666, -0.16915767812484095, 3.583895392013136, 0.0, -0.060100000000000264, -0.00939128268991496, -0.03502388689785807, -0.23000000000000043, -0.0021694405066066125, 0.1985018726591763, -0.9508052708444925, -0.15078023005249896, -0.5366666666666671, 0.01677500000000176, 0.3209106542778395, 1.7235076204551207, 0.0, 0.8717731906121209, -0.09024968789013599, -0.01530318224177396, -0.0023394846532776015, -1.1799999999999997, 0.0, -0.016957446005824828, -0.008371351114713654, -0.002822846441946858, -0.0006999999999992568, -0.24872218900460208, 5.182137427944949, -0.08766511555192924, -0.5390491790482113, 0.16197724589024443, 1.0726025603414238, -0.026674968866760906, -0.8499999999999996, -0.005337213803855434, -0.34183254414331543, -0.05652421972534061, 0.0022599003736019085, -2.395432637525449, -0.7400000000000002, -0.15135865430820328, 0.055755106549981726, -1.1211362391033717, 0.6812415689529079, 0.13880812225232741, -0.5336658716658729, -1.7724627646326354, 0.011708912259531523, -0.7199487064651322, -0.49033041150110357, -1.0305843891810555, -0.006850809464509666, -0.01941201210432375, 0.16505454545454512, 6.188550435865505, 0.19785000000000075, -0.5016384089735979, 0.01102850759858498, -0.0034357053682878558, 0.3507000000000007, -0.0056931977247653975, 0.0, 0.0, 0.49246732673267335, -0.00856793057119809, -0.3144591279799247, -0.17759219725342934, 0.06627661912248417, 1.665381818181821, -0.014971855541720203, -0.012702366127026465, -0.0035745464718690556, -0.003279512862471634, -0.02767099135417972, -1.7743681195516885, -0.06422062675363804, 0.010942138914543875, 1.0009343845149274, -0.05261543847844763, 0.06465303018580215, -0.08932572278406781, -0.10306241619674239, -0.0798847517005683, -0.20368733082307955, 0.34831846901283114, -0.09867296734874742, -0.1970381105413921, 0.6798242848361014, 0.3040618309769272, -0.46357272716558384, 0.19078341906652696, -0.5043763723534553, -1.2770146905209714, 0.0052584932089914105, -0.17342596340314032, -0.2852357705017212, 0.04457181957758394, -0.0015219442861109655, 0.5052374646007749, -0.06822780860191102, 0.48782627884369845, 0.9776860081460583, -0.11066757458135612, 0.7757378334799299, -2.587781875658578, 0.6846942231520359, -0.05397715292831862, 0.6411903450063434, 0.8149269128836778, -0.6299259583308512, 0.9676920397401716, 0.10987115529760416, -0.04728466658941244, -0.3662076727870982, -0.012034628792086721, -0.1261383230641897, -0.20921978619089998, -0.3092966812549989, -1.5435682491715, 1.0195609925334495, -0.24453940114499773, -0.33449900975391245, 0.0378276780166118, -0.007064432534079401, -0.030371511452479183, -0.6948280418819763, 2.9395463963417114, -1.2763945825181793, 1.0840589163821335, -0.37718706870914787, 2.6878343606900117, -0.4383951223360327, -0.6795546638158854, -0.07343873444269633, 0.5053602316647599, -0.06698240537531319, -0.7273547468257746, 0.09286995090239003, -1.0012015002394463, 0.8875060536324213, -0.06960907926006854, -0.14425713625479375, -0.038771458148559645, -0.7153336634886571, -0.013635258766178993, 0.17035339468959165, -0.058487134337134705, 0.2977992208038174, 0.29143000835793487, -0.32670029844376813, -0.5869909213482813, -0.9659464872764119, 0.6060445698253787, 0.8683820528879949, -0.46917685176386215, -0.2762008450687148, -0.32106048318561875, 0.11778994628245698, -0.326554014642376, -1.2696282399045806, -0.44781744325495154, -0.4606002108450902, 0.008009133303566784, -0.07195149603905904, -0.5638804071246852, 0.01055681201462022, -0.10000000000000142, -0.20202562177590977, -0.631879270220157, -0.34273611825405936, 0.26593485282417006, 0.5857222395259427, 0.2392467596572665, -0.6885673635125862, -0.06947369866832531, 0.7795184927650674, 0.3454642395124239, -0.493414633406406, 1.1154754372182367, 0.032029130607593714, -1.1052016829235392, 0.5548512200611455, 0.1395604395604373, 0.3857879041757215, 0.11570127642904282, -0.8722504314108903, 0.6130656015826403, 0.1973533110216934, 0.6421886805959787, 0.9726229418426868, 0.4184827546161891, 0.3390643307176475, 1.1485781778262236, 0.3495990285966215, 1.2325786459683101, 0.2589600228064626, 0.009176543980011331, 0.4289151444608663, -0.0799999999999983, 0.7862082527510594, -0.3827201689892519, -1.5623087364014374, 1.1799999999999997, 0.36999999999999744, -0.010586609602963648, 0.0553762334798904, 0.07978021978021843, 0.32730162240530447, 0.47147452585324245, 0.6186078817038023, 0.048029720119860286, -0.5409858273394441, 0.9699999999999989, -0.47019812878370715, 0.03794126584308799, -0.04999999999999716, 0.07739978274511472, -0.29057667593000147, -0.34273611825405936, 0.06849064206206279, 0.12414981545412473, -1.704738951032553, -0.25, -0.4497870918918281, -0.5171619983344442, 0.5889151444608665, 0.29103840722570595, 0.9500000000000028, 0.9899389012137014, 0.0639977801265239, -0.06477543812125575, -0.2654723398641998, -0.061144199608374805, -0.0671346142908753, -0.05387862203476246, 
-1.8709830656417346, -0.4022412590687239, 0.5231582028684088, -0.02817389578214602, 0.17709549841512384, 0.5916279719647832, 0.4274422614904516, 0.529219630878746, 0.0, 0.7600000000000016, 0.0, 1.129999999999999, 0.7400000000000002, 0.28000000000000025, -0.019999999999999574, -0.35999999999999943, 0.0799999999999983, 0.0, 
-0.17999999999999972, 0.040000000000000036, -0.05000000000000071, 3.5599999999999987, -0.16000000000000014, 0.0, 0.129999999999999, -0.019999999999999574, 0.0, 0.0, 3.3599999999999994, 0.0, -0.5700000000000003, 0.0, 0.25, -0.019999999999999574, 0.0, 0.0, 3.3599999999999994, 0.0, -1.6799999999999997, 0.040000000000000036, 0.21999999999999975, 0.46999999999999886, -0.019999999999999574, -0.019999999999999574, 16.8, 0.0, 0.9699999999999998, 0.009999999999999787, -0.33000000000000007, 
-0.1299999999999999, 0.21000000000000085, 0.5899999999999999, -1.8200000000000003, 0.1900000000000004, 0.0, 0.0, -1.8599999999999994, -6.84, 2.6099999999999994, 0.0, 0.0, -0.1900000000000004, -8.190000000000001, 0.0, 0.9199999999999999, -0.4800000000000004, -0.019999999999999574, -0.47000000000000064, -0.019999999999999574, 3.3599999999999994, 0.3899999999999997, 0.07000000000000028, 0.03000000000000025, 0.0, 0.9100000000000001, -0.009999999999999787, -1.8900000000000006, 0.0, 0.0, 0.4800000000000004, 0.0, 0.21000000000000085, 0.6799999999999997, 0.75, 0.010000000000001563, 2.520000000000003, 0.17999999999999972, -8.82, 7.350000000000001, 0.0, 0.0, -0.17999999999999972, -0.240000000000002, -0.25, 0.35999999999999943, 0.4799999999999969, -0.7999999999999972, -1.009999999999998, 0.6599999999999966, 0.11999999999999744, -0.11999999999999744, 0.38000000000000256, 0.03999999999999915, -0.3200000000000003, 0.7199999999999989, 1.1899999999999977, -0.21999999999999886, 1.0900000000000034, 0.10000000000000142, -0.07000000000000028, 0.4200000000000017, 1.0300000000000011, -0.10000000000000142, 1.3100000000000023, 0.6599999999999966, 0.8100000000000023, -0.6700000000000017, 0.1599999999999966, 0.5, 0.4200000000000017, 0.45000000000000284, -0.10999999999999943, 0.1599999999999966, -0.07000000000000028, -0.35999999999999943, -1.1400000000000006, 1.6000000000000014, 0.5499999999999972, -0.10000000000000142, 0.4799999999999969, 0.7700000000000031, 0.4799999999999969, 0.259999999999998, 1.1899999999999977, 0.20000000000000284, 0.14999999999999858, -0.6099999999999994, 0.36999999999999744, 0.8900000000000006, 
-0.8100000000000023, 0.7899999999999991, 0.03999999999999915, -0.4799999999999969, -0.4200000000000017, 0.22999999999999687, 0.7100000000000009, -0.14999999999999858, 0.29999999999999716, -0.46999999999999886, -1.009999999999998, -0.8599999999999994, 0.07000000000000028, 0.6400000000000006, -0.25, -0.1700000000000017, -0.14000000000000057, -0.9399999999999977, -0.4299999999999997, 1.3299999999999983, 0.10999999999999943, -0.6700000000000017, 1.5399999999999991, 0.1700000000000017, 
-0.5100000000000051, -0.22999999999999687, -0.1700000000000017, 0.060000000000002274, -0.18999999999999773, -0.20000000000000284, -0.21999999999999886, -0.9600000000000009, -0.39000000000000057, 0.6400000000000006, -0.5799999999999983, -0.060000000000002274, 0.060000000000002274, -0.027244832856980317, 0.0, -0.02915212300343306, 0.0, 0.06235350549245222, 0.0014641288433381305, 0.8061618265145949, 0.002887535242235728, -0.004921612352879023, -0.008735681660570371, -0.9570980604823323, -5.460000000000001, -0.08688250076041903, -0.015051918517456464, 0.0, 0.6389458156063483, 0.14000000000000057, 0.0034502714207143015, 0.0, 0.004799921153407105, -0.03392920620890294, -0.02284137944820852, 0.02352692168948778, -0.41641040065392954, -0.0032043586143224445, -0.004689932775084671, -0.005147058823531836, 0.04347826086956452, 0.07645988538541193, -0.002163484346613842, -0.0007002791838006672, 0.14122001984675592, 0.0, -0.0032716753561778944, 1.1326470588235278, -1.290445940097067, -0.09407532501516513, -0.0024074044397597305, 0.28463754956584175, 0.0, 0.5403004319490732, 0.0, 0.09893059934398707, 0.6006530687780813, 0.012669144272915212, -0.018705061523624167, 0.0, -0.0010294117647067225, -0.019109084002482568, -0.46947656140524785, 1.6493856067372672, 0.0, -0.0011742655938764557, -0.4095282619002436, 0.0, 1.8098930854807804, 0.21000000000000085, 0.0, 0.0, 0.0, -0.01910021928239658, -0.09407532501516513, 0.0, -0.005575572857274835, 0.0, 0.00852941176470523, 0.0, -0.023773162849332508, 0.05279411764705877, -0.046024410593555665, 0.14000000000000057, 0.7353813235684719, 0.14000000000000057, 0.002117695107880735, -0.04073001324882952, 0.0, -0.002822459528242227, -0.09747648683093146, -0.0033023055386891542, -0.10456458075016428, -0.0029610150548942116, -0.0029124915096367943, 0.0, -0.0085331202013057, 0.0, -3.6055297476021977, -0.20786915550223206, -0.3326123043618452, 0.0, -0.1329139732768656, -0.409580315715548, 0.0, -0.4079210122983472, -1.661695808030462, 2.5946364458622604, -0.11783769458336302, 0.09351851851851833, -0.20245307454723616, -0.358320218045014, 0.943508530772986, -0.3436810886138222, -0.16404813061027568, -0.013888888888892836, -0.01580942890840653, 0.34577748021594523, 0.0, 0.4322584572578041, 0.08152265211030674, -0.1946067757764336, 0.0, 3.9007229165022963, 0.0, 0.0, 0.0, 3.3158007113816144, 3.3711883267610823, -0.39613555575477477, 0.0, -0.0037037037037030984, 0.8532569583075755, -0.7133674234604364, 0.7876124285847013, -0.0714245014245023, 0.4715270416019841, -0.7939293816045794, -0.5932157088112184, 13.859074074074073, -0.15029251333103488, 0.7041639502033448, 0.1894931211191544, -0.0037037037037030984, 0.010168585660526475, -0.15059733224520855, 0.035857927417934476, 0.9770598055474018, -0.3217709487021736, -0.20286035791456491, 0.21999999999999886, 0.3570187208797364, 0.7475925925925857, -0.16102088154503136, -0.0037037037037030984, -0.0027777777777799884, 0.2799362081215335, -0.04256782736391784, 0.22868270952693948, -0.19840251801658937, 1.0017283428271746, 2.200000000000003, 0.0, -4.900925925925925, 0.222651311194916, 0.05643966385640198, -0.10474987163834193, 0.46789536086667205, 0.9602730532636112, 0.03189900276452562, 0.10351696234048902, 0.38786734548851953, -0.03518518518519187, -0.2532478320038489, -0.6637037037037032, -0.004629629629629761, 0.0, -1.3404970152592988, -1.1971759560994908, 0.7554970194696544, -0.10705365591258342, 3.969999999999999, -17.480000000000004, 0.0, 6.609999999999999, -0.3500000000000014, 0.4799999999999969, 0.38000000000000256, -0.740000000000002, 0.03999999999999915, 1.0300000000000011, 0.0, -0.9699999999999989, 4.569999999999993, 0.019999999999999574, -0.0799999999999983, -2.509999999999998, -0.7699999999999996, -0.08999999999999986, 0.5500000000000007, 2.219999999999999, 0.28999999999999915, 6.599999999999994, -0.6599999999999966, -0.05000000000000071, 0.0, 0.14000000000000057, -0.07000000000000028, 3.460000000000001, -2.8800000000000026, 0.3299999999999983, 0.030000000000001137, 0.6000000000000014, 0.7999999999999972, 1.3200000000000003, 0.3200000000000003, 0.23000000000000043, 0.1700000000000017, -1.0399999999999991, -0.019999999999999574, 0.0, -0.009999999999999787, 0.0, 0.010000000000001563, -0.03999999999999915, -0.05000000000000071, 0.25, 2.3299999999999983, 0.4800000000000004, -0.28999999999999915, 0.019999999999999574, 1.1899999999999977, -0.9100000000000001, 0.0, 0.48999999999999844, 0.120000000000001, 4.6299999999999955, -0.3299999999999983, 0.030000000000001137, -0.9199999999999999, 0.6000000000000014, 6.040000000000006, -0.5900000000000034, 0.240000000000002, 0.120000000000001, 0.7600000000000016, 0.0, 0.5199999999999996, -0.21999999999999886, -0.5700000000000003, 0.7300000000000004, 0.740000000000002, -0.03999999999999915, 0.8800000000000008, -0.7199999999999989, 5.109999999999999, -0.33999999999999986, 0.0, 2.6099999999999994, -0.07000000000000028, 0.08999999999999986, 0.07000000000000028, -0.46000000000000085, 0.870000000000001, 25.700000000000003, 2.222216316218722, -0.009312799654823678, 0.0, -0.0002806225667004725, 0.0, 0.23612173913043577, 0.2649842271293359, -0.9797928467127335, 4.583560934334212, 0.0, -0.057782766881882, -0.2735224378899055, 0.24805197593082084, -0.04359643693228321, 0.3650186784916052, 0.022151898734176, -0.5583578646486256, -3.1499999999999986, -10.337147826086959, 1.589268682502313, -1.402841076311578, 0.0, -0.04068164479120462, 0.8547038376074898, -0.27857946929899136, -0.019346534879110422, 0.16771927861794644, -0.00881161956885812, -0.039621228308579504, 0.20223386836502044, -0.004358744600672537, 4.957029214590671, -6.1273875567502625, -0.4200000000000017, -0.0809373912959046, 0.2000561850454048, 0.0, -0.07411775327489245, -7.077684241423594, -1.6412409445464178, -0.20313508701446636, 0.10956372198783981, -0.7643516813893694, 0.02580022661329373, -0.011498072293711914, 0.0, 0.09347341157661226, 3.155675176890094, 0.0018695537910371485, 0.009198251296444582, 0.0, 0.007533522083262412, 0.0, -0.35935191822624724, 0.19576101237206345, -0.002822963575649595, 0.26549571696275187, -0.010000076586093698, -0.001371350672425109, -0.0009684312103654946, 0.11340943547247484, -0.23104740125422119, -0.08101723310806364, 0.26940133507318365, -0.0534589844810327, -0.3653612828704009, 0.21933255413969732, 0.0, 0.0, -0.4106485257831931, -0.17726431633385786, -7.787147826086958, 0.0, -0.6703757517285442, -0.6026659065792579, -0.05871008864602878, -0.36761894953753504, 0.2927830457521221, -0.8794541815320098, -0.32769322383110655, -6.112746648907766, 11.759999999999998, 0.0, -0.04947637678882444, 0.9299999999999997, -0.09999999999999964, 0.009999999999999787, -0.5299999999999994, 0.0, 0.7100000000000009, 0.9600000000000009, -1.0199999999999996, 0.0, 0.20999999999999375, -0.009999999999999787, -0.4200000000000017, 0.7100000000000009, 0.0, 0.22000000000000064, 0.0, 0.0, 0.269999999999996, 0.0, -0.46999999999999886, -0.5899999999999999, -11.020000000000003, -0.3699999999999992, -0.6899999999999995, -22.42, 0.25, 0.7199999999999989, -9.869999999999997, 0.0, -1.2800000000000011, -0.7100000000000009, 0.0, -0.129999999999999, 
0.5799999999999983, 0.0, 0.0, 0.05000000000000071, -0.129999999999999, 0.28999999999999915, 0.1999999999999993, -0.08999999999999986, -0.46000000000000085, -7.590000000000003, 0.25, -0.4299999999999997, 0.0, 0.0, 0.0, 0.0, 0.6099999999999994, -13.030000000000001, 0.0, -0.4299999999999997, -1.2800000000000011, 0.0, 0.0, 0.75, 0.0, 0.3699999999999992, 0.620000000000001, 0.0, 0.0, 1.0700000000000003, 2.4399999999999977, -0.40000000000000036, 0.25, 0.03999999999999915, -0.9800000000000004, -0.02999999999999936, 0.120000000000001, 0.5500000000000007, -0.019999999999999574, -18.35, 0.0, 0.0, 0.21000000000000085, -0.23000000000000043, 0.6099999999999994, 0.0, 0.7000000000000028, -0.379999999999999, 0.0, 0.9600000000000009, -0.019999999999999574, -0.05000000000000071, -0.5899999999999999, 0.0, 0.0, 0.0, -0.3099999999999996, 0.41999999999999993, -3.2399999999999984, -0.08999999999999986, -0.05000000000000071, 0.0, -0.5700000000000003, 0.0, -0.8200000000000003, 0.0, 0.08999999999999986, -0.0600000000000005, -0.009999999999999787, 0.0, 3.8099999999999987, -1.9800000000000004, -0.009999999999999787, -0.02999999999999936, -2.16, 
0.0, 0.0, 0.0, -0.019999999999999574, -0.23000000000000043, 0.0, -0.21000000000000085, -0.02999999999999936, 0.0, 0.02999999999999936, 0.0, 0.0, 0.0, -0.35999999999999943, -0.019999999999999574, 0.05000000000000071, 0.6400000000000006, 0.0, 3.6000000000000014, -2.8299999999999983, -0.009999999999999787, 3.719999999999999, 
2.8599999999999994, -0.4800000000000004, -0.16000000000000014, -0.5700000000000003, -0.009999999999999787, -0.040000000000000036, -0.009999999999999787, 0.08000000000000007, -0.2400000000000002, -0.009999999999999787, 0.0, -0.16999999999999993, 0.0, 0.0, 0.0, 0.8100000000000005, -22.880000000000003, 0.0, 0.4299999999999997, 0.05000000000000071, -2.3200000000000003, 0.0, 0.09999999999999964, -0.019999999999999574, 0.0, 0.0, -0.16000000000000014, 0.0, -0.009999999999999787, 0.0, -0.15000000000000036, 0.0, 0.009999999999999787, 0.13000000000000078, -0.009999999999999787, 0.0, -0.5199999999999996, 7.690000000000001, 0.0, 0.0, -0.15576703009096882, 0.0, 0.0, -5.979999999999997, -0.7800000000000011, -1.8054709738842973, -3.7347123942014058, -2.181683491034974, -0.259999999999998, 0.06495975212591887, -0.14720215678508985, -1.0560469191770068, 0.19357892478937622, 3.229999999999997, 0.0, 0.0, 0.04217393196264396, 6.5, 0.0, -0.13333562903807383, 0.12306156074121688, -0.27517598765541607, -0.7800000000000011, 4.68, 0.0, -0.5821489869357599, 0.0, -0.5045858560157193, 0.0, -0.9446787301378112, -0.09861172352381153, -0.2025381903642849, -3.615574148982059, 0.1498589882969199, -0.045954822438812215, 0.935546843954187, 0.14770631510472576, -0.6494381531633859, -0.1404214303924789, 0.0, -7.555998315080025, -0.44828613756647684, 0.0, -2.2749692854450636, -0.4414320326958663, -4.840749608719886, 0.5616459096885773, -1.4146002353021636, 0.0, 0.01842915223988406, -0.015941214035159135, 0.08502002205304571, 0.06904802021903933, -2.3949484658457187, 0.10998931881442608, 0.0, 0.23696535019700882, -0.10432891592967586, -0.7193061355339303, 0.0, -4.87734723632725, 0.8967256392149991, 0.20854381361382934, 0.0, -4.362635470815935, 0.045709997081939235, -0.3306546551170868, -5.739388593471066, 0.0, 0.041048488151336215, -6.379388593471067, 0.0, -0.5311953288212319, 3.2886350385464738, 19.98561920808762, -4.803001546146522, -0.411833137485317, 0.0, 0.0, 0.01624930100502553, -5.252437604142326, -0.6610660601401293, -0.010000000000001563, -0.7799999999999994, -0.23000000000000043, 0.0, -0.21000000000000085, -0.05999999999999872, 0.019999999999999574, 0.2699999999999996, -0.26000000000000156, 0.0, 0.0, -0.08000000000000007, 0.0, 0.759999999999998, 0.0, 20.209999999999994, 0.0, 0.0, 0.0, 0.0, 0.0, 0.009999999999999787, 0.7600000000000016, 0.0, 22.58, 0.03999999999999915, 1.0700000000000003, -0.05999999999999872, 0.0, 0.3000000000000007, -0.6499999999999986, 0.5199999999999996, 0.0, 0.7399999999999984, -0.07000000000000028, 0.0, 0.1700000000000017, 0.28000000000000114, 0.6999999999999993, 0.0, 0.0, 0.5399999999999991, 0.0, 0.5799999999999983, 2.009999999999998, 0.11999999999999922, 0.0, 0.08999999999999986, 0.0, -0.17999999999999972, 0.3999999999999986, 0.019999999999999574, 0.009999999999999787, -8.93, 7.670000000000002, 0.1899999999999995, -0.33999999999999986, 0.0, 0.0, 0.0, -0.46999999999999886, -1.8800000000000026, 0.6999999999999993, 0.009999999999999787, 0.0, 0.0, 0.7200000000000006, 0.2699999999999996, -0.9399999999999977, 0.21000000000000085, 22.28, 0.9899999999999984, 0.009999999999999787, 0.07000000000000028, -0.21999999999999886, 1.0, 0.6099999999999994, 0.009999999999999787, -0.33000000000000007, -0.46999999999999886, 0.0, -0.10999999999999943, 0.0, 0.0, 0.0, 0.0, -0.2361082350194721, 0.06526636080370452, 0.2107476635513983, 0.26816809776929773, -0.02859628298051753, 0.2114326282767749, -0.12014608244776781, 0.4092029622479174, 0.0, -0.28314860955769383, 0.43963637405889244, -0.33882239376295686, 0.8756152751721782, -0.01039346442311917, 0.22377795747454954, -0.16259639253639335, 0.23487809576037222, -0.5018347234877734, -0.6630431288440679, 0.28991845883630063, -0.0009489566100278068, 0.11956495121982869, -0.629096484019291, 0.597400395115578, -0.14301602639643107, 0.4782810693608308, 0.18297767883819027, 0.7303743336033008, -13.259999999999998, -0.1648259366637852, -0.13852026468068246, 0.17033306670143844, -0.0030693815987987705, 0.45249270192505264, 0.0, -0.00638511765893135, 0.28129960295761336, -0.0007272727272713553, 0.2092018044639481, -0.004504947936945669, 1.8667623905161754, -0.6734190970530083, 0.0, 0.22431123932385866, -16.64, -0.09333104741359577, 0.0, -0.5200000000000031, -25.740000000000002, -0.5291338059428625, 0.057813714803401695, -0.009427096133387991, 0.0, -0.17520302869872317, -0.06962797878415294, -0.16184549282326444, 0.0, 0.8481999827215816, 0.864773283683947, 0.6651796431379253, -0.008831103275602104, -0.017136830812612303, 0.08801151182156453, 0.38317530248979104, 0.133985105877521, 0.0, -0.8265194975139689, 0.0, -0.5308001166306493, -0.0030276428189193183, -0.02788276175326132, -5.734267070568503, -0.2703297133691649, -0.29074824605315186, 0.13013732762937558, -0.1887574436918893, 3.730582742126785, 0.0, -0.3865933724904753, -0.7139036390671194, 0.6643412312293293, 0.07974961489291132]
    return diff2, diff3, diff4

def Diffs_box_plt (pltPath, label):
    
    diff2,diff3,diff4 = get_pwr_diffs()
    # diff2,diff3,diff4 = get_perf_diffs()

    diffs = []
    diffs = diffs + diff2
    diffs = diffs + diff3
    diffs = diffs + diff4

    features_list = []
    for i in range(19*84):
        features_list.append('2-Features')
    for i in range(19*84):
        features_list.append('3-Features')
    for i in range(19*84):
        features_list.append('4-Features')
    
    models = []
    for i in range(19*84*3):
        models.append('Power Model')

    diff5,diff6,diff7 = get_perf_diffs()

    diffs = diffs + diff5
    diffs = diffs + diff6
    diffs = diffs + diff7

    for i in range(19*84):
        features_list.append('2-Features')
    for i in range(19*84):
        features_list.append('3-Features')
    for i in range(19*84):
        features_list.append('4-Features')
    
    for i in range(19*84*3):
        models.append('Performance Model')

    d = {'Feature Set':features_list,label:diffs,'Model':models}
    df = pd.DataFrame(data=d)
    #****
    # figsize=(12, 5)
    # rgbkymc,RdYlGn,RdYlBu_r
    ax = plt.figure().add_subplot(111)
    # plt.style.use('classic')

    sns.set_context("paper",font_scale=1.5)    
    sns.set_style("whitegrid")

    # df = df.sort_values('Models')
    #  ['firebrick','forestgreen']
    # graph = df.plot(ax=ax, kind='bar', x="GPU Microarchitecture",y="Power (MW)",color=['mediumseagreen','tomato'],legend=False) palette='rocket'
    graph = sns.barplot(x="Feature Set",y=label,hue='Model',data=df, palette="hot")
    
    # graph.axhline(20, label='Exascale Power Threshold',lw=2, color='r',ls='--')

    import itertools 
    hatches = itertools.cycle(['/','+', '-', 'x', '//', '*', 'o', 'O', '.', '\\'])
    num_locations = len(df['Feature Set'].unique())
    for i, patch in enumerate(graph.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)

    # for index, row in df.iterrows():
        # graph.text(row.name, row['Importance Score'],round(row['Importance Score'],2), color='black', ha="center",fontsize=10,weight='bold')
    # ncol=2, 
    ax.legend(loc=0,  fontsize= 12) #bbox_to_anchor=(1, 1)
    
    ax.set_xlabel("Feature Set",weight='bold',fontsize=ls)
    ax.set_ylabel(label,weight='bold',fontsize=ls)
    # plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.xticks(fontsize=12)
    plt.grid(True)
    # change_width(ax, .25)

    # _show_on_single_plot(ax)

    plt.tight_layout()

    plt.savefig(pltPath+'features_set_prediction_error.png', transparent=True,dpi=600)
    # #****
    # plt.figure() 
    # # sns.set_style('whitegrid')
    # # sns.set_context("paper",font_scale=1.5)
    # # plt.style.use('classic')
    # plt.style.use('default')
    # # plot_order = df.groupby("app")["power.draw [W]"].mean().fillna(0).sort_values()[::-1].index

    # bplot = sns.boxplot(y=label, x='Feature Set', 
    #              data=df, hue='Model' 
    #             #  width=0.5,
    #             #  palette="hot"
    #             )
    # bplot.set_xlabel('Feature Set',fontsize = 14, weight='bold')
    # bplot.set_ylabel(label,fontsize = 14, weight='bold' ) 
    # plt.legend(loc=0, fontsize=12)
    # plt.grid(True)
    # plt.savefig(pltPath+'features_set_prediction_error.png',transparent=True, bbox_inches='tight',dpi = 600)

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

def _show_on_single_plot(ax):
 
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = int(p.get_height())
        ax.text(_x, _y, value, ha="center")
    

def gpus_pwr_comparison(plthPath):
    # df = pd.DataFrame(np.random.rand(3, 2), columns=['a', 'b'])
    # ,'GP100','GV100','GA100' ,23.59,19.1,2.6 ,'32-bit FP','32-bit FP','32-bit FP'
    df = pd.DataFrame({
        "GPU Microarchitecture":['GP100','GV100','GA100'],
        "Power (MW)":[47.17,38.46,20.5]
        # "Floating Point (FP)":['64-bit FP','64-bit FP','64-bit FP']
    })
    # figsize=(12, 5)
    # rgbkymc,RdYlGn,RdYlBu_r
    ax = plt.figure().add_subplot(111)
    # plt.style.use('classic')

    sns.set_context("paper",font_scale=1.7)    
    sns.set_style("whitegrid")

    # ,hue='Floating Point (FP)'  ,'forestgreen'
    # graph = df.plot(ax=ax, kind='bar', x="GPU Microarchitecture",y="Power (MW)",color=['mediumseagreen','tomato'],legend=False) palette='rocket'
    graph = sns.barplot(x="GPU Microarchitecture",y="Power (MW)",data=df, palette=['#FC5A50'])
    
    graph.axhline(20, label='Exascale Power Threshold',lw=2, color='r',ls='--')

    import itertools 
    hatches = itertools.cycle(['/','+', '-', 'x', '//', '*', 'o', 'O', '.', '\\'])
    # num_locations = len(df['GPU Microarchitecture'].unique())
    # for i, patch in enumerate(graph.patches):
    #     if i % num_locations == 0:
    #         hatch = next(hatches)
    #     patch.set_hatch(hatch)

    for index, row in df.iterrows():
        graph.text(row.name, row['Power (MW)'],round(row['Power (MW)'],2), color='black', ha="center",fontsize=10,weight='bold')

    ax.legend(loc=0,  ncol=4, fontsize= 12) #bbox_to_anchor=(1, 1)
    
    ax.set_xlabel("GPU Microarchitecture",weight='bold',fontsize=ls)
    ax.set_ylabel('Power (MW)',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    change_width(ax, .25)

    # _show_on_single_plot(ax)

    plt.tight_layout()

    plt.savefig(plthPath+'gpus_estimated_pwr_comparison.png', transparent=True,dpi=600)

def foi(plthPath):
    # df = pd.DataFrame(np.random.rand(3, 2), columns=['a', 'b'])
    # ,'GP100','GV100','GA100' ,23.59,19.1,2.6 ,'32-bit FP','32-bit FP','32-bit FP'
     
    df = pd.DataFrame({
        "Features":['utilization.gpu[%]', 'clocks.current.sm[MHz]', 'utilization.memory[%]','utilization.gpu[%]',  'clocks.current.sm[MHz]','utilization.memory[%]'],
        "Importance Score":[0.17, 0.23, 0.61,0.45, 0.41, 0.12],
        "Models":['Power Model','Power Model','Power Model','Performance Model','Performance Model','Performance Model']
    })
    # figsize=(12, 5)
    # rgbkymc,RdYlGn,RdYlBu_r
    ax = plt.figure().add_subplot(111)
    # plt.style.use('classic')

    sns.set_context("paper",font_scale=1.7)    
    sns.set_style("whitegrid")

    # df = df.sort_values('Models')
    #  
    # graph = df.plot(ax=ax, kind='bar', x="GPU Microarchitecture",y="Power (MW)",color=['mediumseagreen','tomato'],legend=False) palette='rocket'
    graph = sns.barplot(x="Features",y="Importance Score",hue='Models',data=df, palette=['firebrick','forestgreen'])
    
    # graph.axhline(20, label='Exascale Power Threshold',lw=2, color='r',ls='--')

    import itertools 
    hatches = itertools.cycle(['/','+', '-', 'x', '//', '*', 'o', 'O', '.', '\\'])
    num_locations = len(df['Features'].unique())
    for i, patch in enumerate(graph.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)

    # for index, row in df.iterrows():
        # graph.text(row.name, row['Importance Score'],round(row['Importance Score'],2), color='black', ha="center",fontsize=10,weight='bold')
    # ncol=2, 
    ax.legend(loc=0,  fontsize= 12) #bbox_to_anchor=(1, 1)
    
    ax.set_xlabel("Features",weight='bold',fontsize=ls)
    ax.set_ylabel('Importance Score',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # change_width(ax, .25)

    # _show_on_single_plot(ax)

    plt.tight_layout()

    plt.savefig(plthPath+'pwr_perf_foi.png', transparent=True,dpi=600)

def NC(plthPath):

    df = pd.DataFrame({
        "Metrics":['SM[%]', 'Memory[%]','Power (W)','SM[%]', 'Memory[%]','Power (W)'],
        "Values":[0.19, 1.11, 97.42,11.57, 93.11, 115.88],
        "Number of SMs":['Single SM','Single SM','Single SM','Multiple SM','Multiple SM','Multiple SM']
    })
    # figsize=(12, 5)
    # rgbkymc,RdYlGn,RdYlBu_r
    ax = plt.figure().add_subplot(111)
    # plt.style.use('classic')

    sns.set_context("paper",font_scale=1.7)    
    sns.set_style("whitegrid")

    # df = df.sort_values('Models')
    #  
    # graph = df.plot(ax=ax, kind='bar', x="GPU Microarchitecture",y="Power (MW)",color=['mediumseagreen','tomato'],legend=False) palette='rocket'
    graph = sns.barplot(x="Metrics",y="Values",hue='Number of SMs',data=df, palette=['firebrick','forestgreen'])
    
    # graph.axhline(20, label='Exascale Power Threshold',lw=2, color='r',ls='--')

    import itertools 
    hatches = itertools.cycle(['/','+', '-', 'x', '//', '*', 'o', 'O', '.', '\\'])
    num_locations = len(df['Metrics'].unique())
    for i, patch in enumerate(graph.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)

    # for index, row in df.iterrows():
        # graph.text(row.name, row['Importance Score'],round(row['Importance Score'],2), color='black', ha="center",fontsize=10,weight='bold')
    # ncol=2, 
    ax.legend(loc=0,  fontsize= 12) #bbox_to_anchor=(1, 1)
    
    ax.set_xlabel("Metrics",weight='bold',fontsize=ls)
    ax.set_ylabel('Values',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # change_width(ax, .25)

    # _show_on_single_plot(ax)

    plt.tight_layout()

    plt.savefig(plthPath+'nsight-compute-single-vs-multiple-SMs-utilization.png', transparent=True,dpi=600)

def SMI(plthPath):

    df = pd.DataFrame({
        "Metrics":['utilization.gpu[%]', 'utilization.memory[%]','Power (W)','utilization.gpu[%]', 'utilization.memory[%]','Power (W)'],
        "Values":[74, 2, 97.42,62, 11, 115.88],
        "Number of SMs":['Single SM','Single SM','Single SM','Multiple SM','Multiple SM','Multiple SM']
    })
    # figsize=(12, 5)
    # rgbkymc,RdYlGn,RdYlBu_r
    ax = plt.figure().add_subplot(111)
    # plt.style.use('classic')

    sns.set_context("paper",font_scale=1.7)    
    sns.set_style("whitegrid")

    # df = df.sort_values('Models')
    #  
    # graph = df.plot(ax=ax, kind='bar', x="GPU Microarchitecture",y="Power (MW)",color=['mediumseagreen','tomato'],legend=False) palette='rocket'
    graph = sns.barplot(x="Metrics",y="Values",hue='Number of SMs',data=df, palette=['firebrick','forestgreen'])
    
    # graph.axhline(20, label='Exascale Power Threshold',lw=2, color='r',ls='--')

    import itertools 
    hatches = itertools.cycle(['/','+', '-', 'x', '//', '*', 'o', 'O', '.', '\\'])
    num_locations = len(df['Metrics'].unique())
    for i, patch in enumerate(graph.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)

    # for index, row in df.iterrows():
        # graph.text(row.name, row['Importance Score'],round(row['Importance Score'],2), color='black', ha="center",fontsize=10,weight='bold')
    # ncol=2, 
    ax.legend(loc=0,  fontsize= 12) #bbox_to_anchor=(1, 1)
    
    ax.set_xlabel("Metrics",weight='bold',fontsize=ls)
    ax.set_ylabel('Values',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # change_width(ax, .25)

    # _show_on_single_plot(ax)

    plt.tight_layout()

    plt.savefig(plthPath+'nvidia-smi-single-vs-multiple-SMs-utilization.png', transparent=True,dpi=600)

def DGEMM_STREAM(plthPath):

    df = pd.DataFrame({
        "Metrics":['SM[%]', 'Memory[%]','Power (W)','SM[%]', 'Memory[%]','Power (W)'],
        "Values":[18.37, 97.94, 185.1,99.82, 4.55, 215],
        "Benchmarks":['BabelStream','BabelStream','BabelStream','DGEMM','DGEMM','DGEMM']
    })
    # figsize=(12, 5)
    # rgbkymc,RdYlGn,RdYlBu_r
    ax = plt.figure().add_subplot(111)
    # plt.style.use('classic')

    sns.set_context("paper",font_scale=1.7)    
    sns.set_style("whitegrid")

    # df = df.sort_values('Models')
    #  
    # graph = df.plot(ax=ax, kind='bar', x="GPU Microarchitecture",y="Power (MW)",color=['mediumseagreen','tomato'],legend=False) palette='rocket'
    graph = sns.barplot(x="Metrics",y="Values",hue='Benchmarks',data=df, palette=['mediumseagreen','tomato'])
    
    # graph.axhline(20, label='Exascale Power Threshold',lw=2, color='r',ls='--')

    import itertools 
    hatches = itertools.cycle(['/','+', '-', 'x', '//', '*', 'o', 'O', '.', '\\'])
    num_locations = len(df['Metrics'].unique())
    for i, patch in enumerate(graph.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)

    # for index, row in df.iterrows():
        # graph.text(row.name, row['Importance Score'],round(row['Importance Score'],2), color='black', ha="center",fontsize=10,weight='bold')
    # ncol=2, 
    ax.legend(loc=0,  fontsize= 12) #bbox_to_anchor=(1, 1)
    
    ax.set_xlabel("Benchmarks",weight='bold',fontsize=ls)
    ax.set_ylabel('Values',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # change_width(ax, .25)

    # _show_on_single_plot(ax)

    plt.tight_layout()

    plt.savefig(plthPath+'dgemm_stream_utilization_nc_smi.png', transparent=True,dpi=600)


def DCGM_DGEMM_STREAM(plthPath):

    df = pd.DataFrame({
        "Metrics":['Activity[Accumulative]', 'Power (W)','Activity[Accumulative]','Power (W)'],
        "Values":[0.51, 66.4, 1.58,204.2],
        "Benchmarks":['STREAM','STREAM','DGEMM','DGEMM']
    })
    # figsize=(12, 5)
    # rgbkymc,RdYlGn,RdYlBu_r
    ax = plt.figure().add_subplot(111)
    # plt.style.use('classic')

    sns.set_context("paper",font_scale=1.7)    
    sns.set_style("whitegrid")

    # df = df.sort_values('Models')
    #  
    # graph = df.plot(ax=ax, kind='bar', x="GPU Microarchitecture",y="Power (MW)",color=['mediumseagreen','tomato'],legend=False) palette='rocket'
    graph = sns.barplot(x="Metrics",y="Values",hue='Benchmarks',data=df, palette=['mediumseagreen','tomato'])
    
    # graph.axhline(20, label='Exascale Power Threshold',lw=2, color='r',ls='--')

    import itertools 
    hatches = itertools.cycle(['/','+', '-', 'x', '//', '*', 'o', 'O', '.', '\\'])
    num_locations = len(df['Metrics'].unique())
    for i, patch in enumerate(graph.patches):
        if i % num_locations == 0:
            hatch = next(hatches)
        patch.set_hatch(hatch)

    # for index, row in df.iterrows():
        # graph.text(row.name, row['Importance Score'],round(row['Importance Score'],2), color='black', ha="center",fontsize=10,weight='bold')
    # ncol=2, 
    ax.legend(loc=0,  fontsize= 12) #bbox_to_anchor=(1, 1)
    
    ax.set_xlabel("Benchmarks",weight='bold',fontsize=ls)
    ax.set_ylabel('Values (Activity/Power)',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # change_width(ax, .25)

    # _show_on_single_plot(ax)

    plt.tight_layout()

    plt.savefig(plthPath+'dgemm_stream_utilization_dcgm.png', transparent=True,dpi=600)
    
pltPath = "C:/rf/results/misc/perf/"
# pltPath = "/home/ghali/"
# pwr_label = 'Power Prediction Error (W)'
# runtime_label = 'Run Time Prediction Error (S)'
# label = 'Prediction Error'
# Diffs_box_plt (pltPath,label)
# Plt_models_error(pltPath)
# GPUCharacteristic2(pltPath)
# gpus_pwr_comparison(pltPath)
# foi(pltPath)
# Plt_models_MAE (pltPath)
# NC(pltPath)
# SMI(pltPath)
# DGEMM_STREAM(pltPath)
DCGM_DGEMM_STREAM(pltPath)