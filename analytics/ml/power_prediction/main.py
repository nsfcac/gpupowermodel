import pandas as pd
import sys
from sklearn import linear_model
import tkinter as tk 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

'''
def toggle_fs(dummy=None):
    state = False if root.attributes('-fullscreen') else True
    root.attributes('-fullscreen', state)
    if not state:
        root.geometry('300x300+100+100')
'''

from os import error
import numpy as np
from numpy.lib import type_check
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import sklearn

from sklearn import metrics
import itertools

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import sys

sys.path.insert(1, 'C:/rf/projects/gpupowermodel/analytics/ml analysis/ml models')

import mlr
import svr
import rfr
# import xgbr

# plt.rcParams['axes.grid'] = True

ls = 14
ts = 14
ms = 75

def ReadGPUFeatures():
    p100_freqs = [544, 556, 569, 582, 594, 607, 620, 632, 645, 658, 670, 683, 696, 708, 721, 734, 746, 759, 772, 784, 797, 810, 822, 835, 847, 860, 873, 885, 898, 911, 923, 936, 949, 961, 974, 987, 999, 1012, 1025, 1037, 1050, 1063, 1075, 1088, 1101, 1113, 1126, 1139, 1151, 1164, 1177, 1189, 1202, 1215, 1227, 1240, 1252, 1265, 1278, 1290, 1303, 1316, 1328]
    v100_freqs = [135, 142, 150, 157, 165, 172, 180, 187, 195, 202, 210, 217, 225, 232, 240, 247, 255, 262, 270, 277, 285, 292, 300, 307, 315, 322, 330, 337, 345, 352, 360, 367, 375, 382, 390, 397, 405, 412, 420, 427, 435, 442, 450, 457, 465, 472, 480, 487, 495, 502, 510, 517, 525, 532, 540, 547, 555, 562, 570, 577, 585, 592, 600, 607, 615, 622, 630, 637, 645, 652, 660, 667, 675, 682, 690, 697, 705, 712, 720, 727, 735, 742, 750, 757, 765, 772, 780, 787, 795, 802, 810, 817, 825, 832, 840, 847, 855, 862, 870, 877, 885, 892, 900, 907, 915, 922, 930, 937, 945, 952, 960, 967, 975, 982, 990, 997, 1005, 1012, 1020, 1027, 1035, 1042, 1050, 1057, 1065, 1072, 1080, 1087, 1095, 1102, 1110, 1117, 1125, 1132, 1140, 1147, 1155, 1162, 1170, 1177, 1185, 1192, 1200, 1207, 1215, 1222, 1230, 1237, 1245, 1252, 1260, 1267, 1275, 1282, 1290, 1297, 1305, 1312, 1320, 1327, 1335, 1342, 1350, 1357, 1365, 1372, 1380]
    
    gpu = pd.read_csv("C:/rf/SPEC_gpu_p100_v100_metrics.csv")
    gpu.drop(['Unnamed: 0'], inplace = True, axis=1)
    # gpu = gpu[gpu['clocks.current.sm [MHz]'].isin(p100_freqs)]
    # gpu = gpu[gpu['clocks.current.sm [MHz]'].isin(v100_freqs)]
    p100_data = gpu[gpu["arch"] == 'P100']  
    v100_data = gpu[gpu["arch"] == 'V100']

    p100_data = p100_data[p100_data['clocks.current.sm [MHz]'].isin(p100_freqs)]
    v100_data = v100_data[v100_data['clocks.current.sm [MHz]'].isin(v100_freqs)]

    print(p100_data.shape[0])
    print(v100_data.shape[0])

    cols = list(gpu.columns)
    print (cols)

    gpu.info()
    gpu.describe()

    # p100_data.to_csv("C:/rf/SPEC_GPU_P100_Features.csv", encoding='utf-8-sig') #mode='a', header=False,index=False,
    # v100_data.to_csv("C:/rf/SPEC_GPU_V100_Features.csv", encoding='utf-8-sig') #mode='a', header=False,index=False,

    return p100_data, v100_data, gpu

def RemoveGPUFeatures(p100_data, data):
    # data.drop(['arch','clocks.current.graphics [MHz]','cores','transistors','die_size','TDP','memory.total [MiB]','base_run_time'], inplace = True, axis=1)
    # p100_data.drop(['arch','clocks.current.graphics [MHz]','cores','transistors','die_size','TDP','memory.total [MiB]','base_run_time'], inplace = True, axis=1)
    data.drop(['arch','clocks.current.graphics [MHz]','TDP','memory.total [MiB]','base_run_time'], inplace = True, axis=1)
    p100_data.drop(['arch','clocks.current.graphics [MHz]','TDP','memory.total [MiB]','base_run_time'], inplace = True, axis=1)

    # data['timestamp'] = pd.to_datetime(data['timestamp']).astype(np.int64) // 10**9
    # p100_data['timestamp'] = pd.to_datetime(p100_data['timestamp']).astype(np.int64) // 10**9
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.second
    p100_data['timestamp'] = pd.to_datetime(p100_data['timestamp']).dt.second

    return p100_data, data

def Plt_GPU_Mem_Util(p100_data, data):
    a_df = p100_data.groupby(['app']).mean().reset_index()
    app_df = a_df[['app','utilization.gpu [%]','utilization.memory [%]']]
    # app_df = app_df.sort_values(by=['utilization.gpu [%]','utilization.memory [%]']).reset_index()

    data_a_df = data.groupby(['app']).mean().reset_index()
    data_app_df = data_a_df[['app','utilization.gpu [%]','utilization.memory [%]']]
    # data_app_df = data_app_df.sort_values(by=['utilization.gpu [%]','utilization.memory [%]']).reset_index()

    combine_dict = {'GP100.utilization.gpu [%]': app_df['utilization.gpu [%]'].to_list(),'GP100.utilization.memory [%]': app_df['utilization.memory [%]'].to_list(),'GV100.utilization.gpu [%]': data_app_df['utilization.gpu [%]'].to_list(),'GV100.utilization.memory [%]': data_app_df['utilization.memory [%]'].to_list()}
    # .reset_index()
    plt_df = pd.DataFrame(combine_dict,index=app_df['app'])
    plt_df = plt_df.sort_values(by=['GP100.utilization.gpu [%]','GP100.utilization.memory [%]'],ascending=False)
    
    # rgbkymc,RdYlGn,RdYlBu_r
    # print (plt_df.head())

    ax = plt.figure(figsize=(12, 5)).add_subplot(111)
    # ax = plt.axes() colormap='gist_rainbow',
    plt.style.use('classic')
    plt_df.plot(ax=ax, kind='bar', rot=70, legend=False)
    bars = ax.patches
    hatches = ''.join(h*len(plt_df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='best', fontsize = 12, ncol=2)
    # , 
    ax.set_xlabel("Benchmarks",weight='bold',fontsize=ls)
    ax.set_ylabel('Utilization (%)',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig('C:/rf/results/apps_gpu_mem_utilization.png',transparent=True, bbox_inches='tight')
    # plt.show()

def Plt_PWR_VS_Features(data):
    data_df = data.copy()
    # apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    # fig, ax = plt.subplots(3, figsize=(8, 10))
    fig, axes = plt.subplots(2, 2, sharey=True)
    plt.style.use('classic')
    sns.set_context("paper", font_scale=2)
    apps = ["hotspot"]
    for app in apps:
        d = data_df[(data_df['app'] == app) & (data_df['clocks.current.sm [MHz]'] == 1380)]
        
        grpby = 'utilization.gpu [%]'
        df = d.copy()
        df = df.groupby(grpby).mean()
        sns.scatterplot(ax=axes[0,0], data=df, x=grpby, y='power.draw [W]', s=35, color="b", marker="s")
        axes[0,0].set_xlim([-5, 105])
        axes[0,0].set_xlabel( '(a) GPU Utilization (%)',weight='bold',fontsize=10) 
        axes[0,0].set_ylabel( 'Power (W)',weight='bold',fontsize=10) 
        axes[0,0].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP')
        axes[0,0].legend(loc="lower right", prop={'size': 8})

        grpby = 'utilization.memory [%]' #'clocks.current.sm [MHz]'  'utilization.gpu [%]'
        df = d.copy()
        df = df.groupby(grpby).mean().reset_index()        
        sns.scatterplot(ax=axes[0,1], data=df, x=grpby, y='power.draw [W]', s=35, color="g", marker="d")
        axes[0,1].set_xlim([-5, 105])
        axes[0,1].set_xlabel('(b) Memory Utilization (%)', weight='bold',fontsize=10)
        axes[0,1].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP') 
        axes[0,1].legend(loc="lower right", prop={'size': 8})
        
        grpby = 'clocks.current.sm [MHz]'
        df = data_df[data_df['app'] == app]
        df = df.groupby(grpby).max().reset_index()
        sns.scatterplot(ax=axes[1,0], data=df, x=grpby, y='power.draw [W]', s=50, color="r", marker=".") 
        axes[1,0].set_xlabel( '(c) Core Frequency (MHz)',weight='bold',fontsize=10) 
        axes[1,0].set_ylabel( 'Power (W)',weight='bold',fontsize=10)
        axes[1,0].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP')
        axes[1,0].legend(loc="lower right", prop={'size': 8})

        grpby = 'temperature.gpu'
        df = d.copy()
        df = df.groupby(grpby).mean().reset_index()
        sns.scatterplot(ax=axes[1,1], data=df, x=grpby, y='power.draw [W]', s=70, color="m", marker="^")
        axes[1,1].set_xlabel( u"(d) Temperature(\N{DEGREE SIGN}C)",weight='bold',fontsize=10)
        axes[1,1].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP')
        axes[1,1].legend(loc="lower right", prop={'size': 8})

        fig.tight_layout()
        # plt.rcParams['axes.grid'] = True
        # plt.grid(True)
        plt.savefig('C:/rf/results/pwr-vs-features.png') #transparent=True



def Plt_Apps_Pwr(gpu_data):
    # from sklearn.preprocessing import LabelEncoder
    # enc = LabelEncoder()
    # enc.fit(gpu_data['arch'])
    # gpu_data['arch'] = enc.transform(gpu_data['arch'])
    # data = gpu_data.groupby(['app','arch'],as_index=False).agg({'power.draw [W]':'mean'})
    data = gpu_data.groupby(['arch','app', 'clocks.current.sm [MHz]']).mean().reset_index()
    data.rename(columns={'arch': 'GPU Microarchitecture'}, inplace=True)
    data.replace(to_replace ="P100", value ="NVIDIA GP100", inplace=True) 
    data.replace(to_replace ="V100", value ="NVIDIA GV100", inplace=True)

    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    plot_order = data.groupby("app")["power.draw [W]"].mean().fillna(0).sort_values()[::-1].index

    sns.boxplot(x="app", y="power.draw [W]", hue='GPU Microarchitecture',data=data, orient="v", palette="hot",order=plot_order)
    # Set labels
    plt.ylabel("Power (W)", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/apps_pwr_pattern.png',transparent=True, bbox_inches='tight')


p100_data, data, gpu_data = ReadGPUFeatures()

p100_data, data = RemoveGPUFeatures(p100_data, data)

# Plt_GPU_Mem_Util(p100_data, data)
# Plt_Apps_Pwr(gpu_data)

# cols = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]']
# cols = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]','timestamp','temperature.gpu','clocks.current.memory [MHz]','memory.used [MiB]','memory.free [MiB]']        
# cls = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]','timestamp','temperature.gpu','clocks.current.memory [MHz]','memory.used [MiB]','memory.free [MiB]']
cls = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]']
# ,'die_size','transistors','cores' ,'timestamp','temperature.gpu' ,'cores'
n = 0
n_features = 7
for i in range(n_features):
    if i == 1:
        break
    cols = cls[:-i]
    if i == 0:
        cols = cls
    n = len(cols)-2
    if i == 6:
        cols = cls[:-i+1]
    print (i,n,cols)
    
    num_features = str(n)+'_features_'

    p100_data = p100_data.reindex(columns=cols)
    data = data.reindex(columns=cols)
    gpu_data = gpu_data.reindex(columns=cols)

    # print(p100_data.shape[0])
    # print(data.shape[0])
    # print(gpu_data.shape[0])

    # Plt_PWR_VS_Features(data)
    # sys.exit(0)
    
    # gpu_data = data[data['utilization.gpu [%]'] > 0]
    # print ('utilization.gpu [%] > 0 :',gpu_data.shape[0])
    # print (gpu_data.head())
    # mem_data = data[data['utilization.memory [%]'] > 0]
    # print ('utilization.memory [%] > 0 :',mem_data.shape[0])
    # print (mem_data.head())

    data = data[data['power.draw [W]'] > 28]
    p100_data = p100_data[p100_data['power.draw [W]'] > 28]
    # print ('power.draw [W] > 0 :',pwr_data.shape[0])
    # print (pwr_data.head())
    # print ('Frequencies:',p100_data.groupby('app')['clocks.current.sm [MHz]'].nunique())
    # print ('Apps:',len(gpu_data.groupby('app')))
    # sys.exit(0)

    p100_data = p100_data.groupby(['app', 'clocks.current.sm [MHz]']).mean().reset_index()
    # gpu_data = gpu_data.groupby(['app', 'clocks.current.sm [MHz]']).mean().reset_index()
    data = data.groupby(['app', 'clocks.current.sm [MHz]']).mean().reset_index()
    gpu_data = pd.concat([p100_data,data]).reset_index()

    if num_features == '2_features_':
        cols = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]']
    p100_data = p100_data.reindex(columns=cols)
    data = data.reindex(columns=cols)
    gpu_data = gpu_data.reindex(columns=cols)
    p100_data.rename(columns={'power.draw [W]': 'power.draw (W)','utilization.gpu [%]':'utilization.gpu (%)','utilization.memory [%]':'utilization.memory (%)','clocks.current.sm [MHz]':'clocks.current.sm (MHz)'}, inplace=True)
    data.rename(columns={'power.draw [W]': 'power.draw (W)','utilization.gpu [%]':'utilization.gpu (%)','utilization.memory [%]':'utilization.memory (%)','clocks.current.sm [MHz]':'clocks.current.sm (MHz)'}, inplace=True)
    gpu_data.rename(columns={'power.draw [W]': 'power.draw (W)','utilization.gpu [%]':'utilization.gpu (%)','utilization.memory [%]':'utilization.memory (%)','clocks.current.sm [MHz]':'clocks.current.sm (MHz)'}, inplace=True)
    
    data.fillna(method='pad')
    p100_data.fillna(method='pad')
    gpu_data.fillna(method='pad')
    p100_data.head()
    data.head()
    gpu_data.head()

    x = data.iloc[:, 2:]#.values.reshape(-1, 1)
    y = data.iloc[:, 0]#.values.reshape(-1, 1)

    x_p100 = p100_data.iloc[:, 2:]
    y_p100 = p100_data.iloc[:, 0]

    x_gpu = gpu_data.iloc[:, 2:]
    y_gpu = gpu_data.iloc[:, 0]

    x_p100_df = {}
    y_p100_df = {}

    x_v100_df = {}
    y_v100_df = {}
    apps_list = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    for ap in apps_list:
        df = p100_data[p100_data['app'] == ap]
        x_p100_df[ap] = df.iloc[:, 2:] 
        y_p100_df[ap] = df.iloc[:, 0]
        df1 = data[data['app'] == ap]
        x_v100_df[ap] = df1.iloc[:, 2:] 
        y_v100_df[ap] = df1.iloc[:, 0]

    size = 0.5000

    '''
    #CALLING MULTI-VARIATE LINEAR REGRESSION
    # mlr.global_fit_predict(p100_data,x_p100,y_p100,x_p100_df,y_p100_df,apps_list,ls,ts,ms,num_features,size)
    mlr.global_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size)
    # mlr.app_fit_predict(p100_data,x_p100,y_p100,x_p100_df,y_p100_df,apps_list,ls,ts,ms,num_features,size)
    mlr.app_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size)
    #mlr.interarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size)
    mlr.crossarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size)
    # from sklearn.preprocessing import StandardScaler
    # sc_x = StandardScaler()
    # x = sc_x.fit_transform(x.astype(float))
    # X_p100 = sc_x.fit_transform(X_p100.astype(float))
    '''

    # pltPath = 'C:/rf/results/svr/'
    # svr.app_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # svr.global_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # svr.crossarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # pltPath = 'C:/rf/results/inter-arch/'
    pltPath = 'C:/rf/results/rfr/'
    label = 'Power (S)'
    # rfr.app_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # rfr.global_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    rfr.crossarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath,label)
    # pltPath = 'C:/rf/results/mlr/'
    # mlr.app_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # mlr.global_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # mlr.crossarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # pltPath = 'C:/rf/results/xgbr/'
    # xgbr.app_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # xgbr.global_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # xgbr.crossarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)

    
    '''
    x = x_p100.copy()
    y = y_p100.copy()
    x_p100 = x
    y_p100 = y
    
    '''


    # from sklearn import metrics
    # MSE_train = metrics.mean_squared_error(y_train, y_train_pred)
    # RMSE_train = np.sqrt(MSE_train)                                           
    # print(MSE_train,RMSE_train)

    # #GIVE YOUR ANSWER FOR TASK-9 IN THIS CELL
    # # MSE_test = metrics.mean_squared_error(y_test, y_test_pred)
    # MSE_test = metrics.mean_squared_error(Y_test, y_test_pred)
    # RMSE_test = np.sqrt(MSE_test)  
    # #print(MSE_test,RMSE_test)

    # #from metrics import mean_absolute_percentage_error
    # #from sklearn.metrics import mean_absolute_percentage_error
    # # print (metrics.mean_absolute_percentage_error(y_test, y_test_pred))
    # print (metrics.mean_absolute_percentage_error(Y_test, y_test_pred))

    # **************************************************************************************************************************************
    
    '''
    root = tk.Tk()
    root.attributes('-fullscreen', True) # make main window full-screen

    canvas1 = tk.Canvas(root, bg='white', highlightthickness=0)
    canvas1.pack(fill=tk.BOTH, expand=True) # configure canvas to occupy the whole main window
    
    label1 = tk.Label(root, text='Timestamp: ', font='Helvetica 10 bold')
    canvas1.create_window(450, 100, window=label1)

    ts = tk.Entry (root) # create 1st entry box
    canvas1.create_window(650, 100, window=ts)

    # Temperature label and input box
    label2 = tk.Label(root, text=' temperature.gpu: ', font='Helvetica 10 bold')
    canvas1.create_window(800, 100, window=label2)

    temp = tk.Entry (root) # create 2nd entry box
    canvas1.create_window(1000, 100, window=temp)

    # Utilization.gpu label and input box
    label3 = tk.Label(root, text=' utilization.gpu [%]: ', font='Helvetica 10 bold')
    canvas1.create_window(1180, 100, window=label3)

    util_gpu = tk.Entry (root) # create 3rd entry box
    canvas1.create_window(1355, 100, window=util_gpu)


    # timestamp label and input box
    label4 = tk.Label(root, text='utilization.memory [%]: ', font='Helvetica 10 bold')
    canvas1.create_window(450, 160, window=label4)

    util_mem = tk.Entry (root) # create 4th entry box
    canvas1.create_window(650, 160, window=util_mem)

    # Temperature label and input box
    label5 = tk.Label(root, text=' clocks.current.sm [MHz]: ', font='Helvetica 10 bold')
    canvas1.create_window(800, 160, window=label5)

    sm_freq = tk.Entry (root) # create 5th entry box
    canvas1.create_window(1000, 160, window=sm_freq)

    # Utilization.gpu label and input box
    label6 = tk.Label(root, text=' memory.used [MiB]: ', font='Helvetica 10 bold')
    canvas1.create_window(1180, 160, window=label6)

    mem_used = tk.Entry (root) # create 6th entry box
    canvas1.create_window(1355, 160, window=mem_used)

    def values(): 
        global New_Time_Stamp #our 1st input variable
        New_Time_Stamp = float(ts.get()) 
        
        global New_Temperature #our 2nd input variable
        New_Temperature = float(temp.get()) 
        
        global New_Utilization_GPU #our 3rd input variable
        New_Utilization_GPU = float(util_gpu.get())

        global New_Utilization_Mem #our 4th input variable
        New_Utilization_Mem = float(util_mem.get())

        global New_SM_Freq #our 5th input variable
        New_SM_Freq = float(sm_freq.get())

        global New_Mem_Used #our 5th input variable
        New_Mem_Used = float(mem_used.get()) 

        test_df = pd.DataFrame({
        "Features":[New_Time_Stamp,New_Temperature,New_Utilization_GPU,New_Utilization_Mem,New_SM_Freq,New_Mem_Used]
        }, 
        index=['timestamp','temperature.gpu','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]' ,'memory.used [MiB]']
        )
        
        x_test_df = sc_x.fit_transform(test_df)
        Prediction_result  = lm.predict(x_test_df.reshape(1,-1))
        label_Prediction = tk.Label(root, text= 'Predicted Power (W): '+str(Prediction_result[0]), font='Verdana 10 bold',bg='orange')
        canvas1.create_window(950, 220, window=label_Prediction)
        
    button1 = tk.Button (root, text='Predict Power (W)',command=values, bg='deepskyblue') # button to call the 'values' command above 
    canvas1.create_window(570, 220, window=button1)
    ''' 
    
    '''
    #plot 1st scatter 
    figure3 = plt.Figure(figsize=(5,4), dpi=100)
    ax3 = figure3.add_subplot(111)
    ax3.scatter(df['Interest_Rate'].astype(float),df['Stock_Index_Price'].astype(float), color = 'r')
    scatter3 = FigureCanvasTkAgg(figure3, root) 
    scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    ax3.legend(['Stock_Index_Price']) 
    ax3.set_xlabel('Interest Rate')
    ax3.set_title('Interest Rate Vs. Stock Index Price')

    #plot 2nd scatter 
    figure4 = plt.Figure(figsize=(5,4), dpi=100)
    ax4 = figure4.add_subplot(111)
    ax4.scatter(df['Unemployment_Rate'].astype(float),df['Stock_Index_Price'].astype(float), color = 'g')
    scatter4 = FigureCanvasTkAgg(figure4, root) 
    scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    ax4.legend(['Stock_Index_Price']) 
    ax4.set_xlabel('Unemployment_Rate')
    ax4.set_title('Unemployment_Rate Vs. Stock Index Price')
    '''

    '''
    # root.bind('<Escape>', toggle_fs)

    # root.mainloop()
    '''