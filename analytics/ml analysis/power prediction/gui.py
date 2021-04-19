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

ls = 14
ts = 14

def ReadGPUFeatures():
    gpu = pd.read_csv("C:/rf/SPEC_gpu_p100_v100_metrics.csv")
    gpu.drop(['Unnamed: 0'], inplace = True, axis=1)
    p100_data = gpu[gpu["arch"] == 'P100']  
    v100_data = gpu[gpu["arch"] == 'V100']

    p100_freqs = [544, 556, 569, 582, 594, 607, 620, 632, 645, 658, 670, 683, 696, 708, 721, 734, 746, 759, 772, 784, 797, 810, 822, 835, 847, 860, 873, 885, 898, 911, 923, 936, 949, 961, 974, 987, 999, 1012, 1025, 1037, 1050, 1063, 1075, 1088, 1101, 1113, 1126, 1139, 1151, 1164, 1177, 1189, 1202, 1215, 1227, 1240, 1252, 1265, 1278, 1290, 1303, 1316, 1328]
    v100_freqs = [135, 142, 150, 157, 165, 172, 180, 187, 195, 202, 210, 217, 225, 232, 240, 247, 255, 262, 270, 277, 285, 292, 300, 307, 315, 322, 330, 337, 345, 352, 360, 367, 375, 382, 390, 397, 405, 412, 420, 427, 435, 442, 450, 457, 465, 472, 480, 487, 495, 502, 510, 517, 525, 532, 540, 547, 555, 562, 570, 577, 585, 592, 600, 607, 615, 622, 630, 637, 645, 652, 660, 667, 675, 682, 690, 697, 705, 712, 720, 727, 735, 742, 750, 757, 765, 772, 780, 787, 795, 802, 810, 817, 825, 832, 840, 847, 855, 862, 870, 877, 885, 892, 900, 907, 915, 922, 930, 937, 945, 952, 960, 967, 975, 982, 990, 997, 1005, 1012, 1020, 1027, 1035, 1042, 1050, 1057, 1065, 1072, 1080, 1087, 1095, 1102, 1110, 1117, 1125, 1132, 1140, 1147, 1155, 1162, 1170, 1177, 1185, 1192, 1200, 1207, 1215, 1222, 1230, 1237, 1245, 1252, 1260, 1267, 1275, 1282, 1290, 1297, 1305, 1312, 1320, 1327, 1335, 1342, 1350, 1357, 1365, 1372, 1380]
    # df = p100_df[p100_df['app'] == app]
    p100_data = p100_data[p100_data['clocks.current.sm [MHz]'].isin(p100_freqs)]
    # print (len(p100_data[p100_data['app'] == 'stencil']['clocks.current.sm [MHz]'].unique()))
    v100_data = v100_data[v100_data['clocks.current.sm [MHz]'].isin(v100_freqs)]
    # print (len(v100_data[v100_data['app'] == 'stencil']['clocks.current.sm [MHz]'].unique()))

    print(p100_data.shape[0])
    print(v100_data.shape[0])

    cols = list(gpu.columns)
    print (cols)

    gpu.info()
    gpu.describe()

    # p100_data.to_csv("C:/rf/SPEC_GPU_P100_Features.csv", encoding='utf-8-sig') #mode='a', header=False,index=False,
    # v100_data.to_csv("C:/rf/SPEC_GPU_V100_Features.csv", encoding='utf-8-sig') #mode='a', header=False,index=False,

    return p100_data, v100_data

def RemoveGPUFeatures(p100_data, data):
    data.drop(['arch','clocks.current.graphics [MHz]','cores','transistors','die_size','TDP','memory.total [MiB]','base_run_time'], inplace = True, axis=1)
    p100_data.drop(['arch','clocks.current.graphics [MHz]','cores','transistors','die_size','TDP','memory.total [MiB]','base_run_time'], inplace = True, axis=1)

    # data['timestamp'] = pd.to_datetime(data['timestamp']).astype(np.int64) // 10**9
    # p100_data['timestamp'] = pd.to_datetime(p100_data['timestamp']).astype(np.int64) // 10**9

    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.second
    p100_data['timestamp'] = pd.to_datetime(p100_data['timestamp']).dt.second

    return p100_data, data

def Plt_GPU_Mem_Util(p100_data, data):
    a_df = p100_data.groupby(['app']).mean().reset_index()
    app_df = a_df[['app','utilization.gpu [%]','utilization.memory [%]']]
    app_df = app_df.sort_values(by=['utilization.gpu [%]','utilization.memory [%]']).reset_index()

    plt_df = pd.DataFrame({'utilization.gpu [%]': app_df['utilization.gpu [%]'].to_list(),'utilization.memory [%]': app_df['utilization.memory [%]'].to_list()}, index=app_df['app'])
    # labels = app_df['app']
    plt.figure()
    # rgbkymc
    ax = plt_df.plot.bar(rot=70, colormap='RdYlGn')
    ax.set_xlabel("Micro-benchmarks",weight='bold',fontsize=ls)
    ax.set_ylabel('Utilization (%)',weight='bold',fontsize=ls)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=ts)

    # ax.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('C:/rf/apps_gpu_mem_utilization.png')
    # plt.show()

def Plt_PWR_VS_Features(p100_data):
    
    # foi = ['timestamp', 'temperature.gpu','utilization.gpu [%]','utilization.memory [%]','memory.free [MiB]','memory.used [MiB]','clocks.current.memory [MHz]']
    if i == 0:
        p100_df = p100_data.copy()
        app = 'stencil'#'hotspot'#'stencil'#'nw' tpacf
        for feature in cols:
            plt.figure()
            plt.style.use('classic')
            # grpby = 'clocks.current.sm [MHz]'
            # print (len(p100_df[p100_df['app'] == app]['clocks.current.sm [MHz]'].unique()))
            df = p100_df[(p100_df['app'] == app) & (p100_df['clocks.current.sm [MHz]'] == 1328)]
            # df = p100_df[p100_df['app'] == app]
            # df = df.sort_values(by=['timestamp'])
            df = df.groupby([feature]).mean().reset_index()
            print(feature,len(df.index))
            # print (df['timestamp'])
            x = df[feature]
            y = df['power.draw [W]']
            # plt.scatter(x, y,'xb-')
            # plt.plot(x,y,'.r-',linewidth=2, markersize=12)
            plt.scatter(x, y, color='orangered', marker ='H',  edgecolor =None,  s = 75)
            plt.xlabel(feature, weight='bold',fontsize=ls)  
            plt.ylabel('Power (W)', weight='bold',fontsize=ls)
            plt.tick_params(axis='both', which='major', labelsize=ts)
            plt.grid(True)
            plt.savefig('C:/rf/'+feature+'pwr.png',transparent=True)
        # freqs = [544,...1328]
        plt.figure()
        plt.style.use('classic')
        feature = 'clocks.current.sm [MHz]'
        # print (len(p100_df[p100_df['app'] == app]['clocks.current.sm [MHz]'].unique()))
        df = p100_df[p100_df['app'] == app]
        # df = df.sort_values(by=['timestamp'])
        df = df.groupby([feature]).mean().reset_index()
        print(feature,len(df.index))
        # print ('timestamp')
        x = df[feature]
        y = df['power.draw [W]']
        # plt.plot(x,y,'.r-',linewidth=2, markersize=12)
        plt.scatter(x, y, color='orangered', marker ='H',  edgecolor =None,  s = 75)
        plt.xlabel(feature, weight='bold',fontsize=ls)  
        plt.ylabel('Power (W)', weight='bold',fontsize=ls)
        plt.tick_params(axis='both', which='major', labelsize=ts)
        plt.grid(True)
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        # plt.plot(x,p(x),"r--")
        plt.savefig('C:/rf/'+feature+'pwr.png',transparent=True)

p100_data, data = ReadGPUFeatures()

p100_data, data = RemoveGPUFeatures(p100_data, data)

Plt_GPU_Mem_Util(p100_data, data)

# cols = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]']
# cols = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]','timestamp','temperature.gpu','clocks.current.memory [MHz]','memory.used [MiB]','memory.free [MiB]']        
cls = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]','timestamp','temperature.gpu','clocks.current.memory [MHz]','memory.used [MiB]','memory.free [MiB]']
n = 0
for i in range(7):
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

    print(p100_data.shape[0])
    print(data.shape[0])

    Plt_PWR_VS_Features(p100_data)

    p100_data = p100_data.groupby(['app', 'clocks.current.sm [MHz]']).mean().reset_index()
    data = data.groupby(['app', 'clocks.current.sm [MHz]']).mean().reset_index()
    if num_features == '2_features_':
        cols = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]']
    p100_data = p100_data.reindex(columns=cols)
    data = data.reindex(columns=cols)

    data.fillna(method='pad')
    p100_data.fillna(method='pad')
    p100_data.head()

    x = data.iloc[:, 2:]#.values.reshape(-1, 1)
    y = data.iloc[:, 0]#.values.reshape(-1, 1)

    X_p100 = p100_data.iloc[:, 2:]
    Y_p100 = p100_data.iloc[:, 0]

    # from sklearn.preprocessing import StandardScaler
    # sc_x = StandardScaler()
    # x = sc_x.fit_transform(x.astype(float))
    # X_p100 = sc_x.fit_transform(X_p100.astype(float))

    x = X_p100.copy()
    y = Y_p100.copy()

    # fig.suptitle('This is the figure title', fontsize=12)
    '''
    X_p100 = x
    Y_p100 = y
    '''
    # ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    tpacf = p100_data[p100_data['app'] == 'tpacf']
    Xtpacf = tpacf.iloc[:, 2:]
    Ytpacf = tpacf.iloc[:, 0]

    stencil = p100_data[p100_data['app'] == 'stencil']
    Xstencil = stencil.iloc[:, 2:]
    Ystencil = stencil.iloc[:, 0]

    lbm = p100_data[p100_data['app'] == 'lbm']
    Xlbm = lbm.iloc[:, 2:]
    Ylbm = lbm.iloc[:, 0]

    fft = p100_data[p100_data['app'] == 'fft']
    Xfft = fft.iloc[:, 2:]
    Yfft = fft.iloc[:, 0]

    spmv = p100_data[p100_data['app'] == 'spmv']
    Xspmv = spmv.iloc[:, 2:]
    Yspmv = spmv.iloc[:, 0]

    mriq = p100_data[p100_data['app'] == 'mriq']
    Xmriq = mriq.iloc[:, 2:]
    Ymriq = mriq.iloc[:, 0]

    histo = p100_data[p100_data['app'] == 'histo']
    Xhisto = histo.iloc[:, 2:]
    Yhisto = histo.iloc[:, 0]

    bfs = p100_data[p100_data['app'] == 'bfs']
    Xbfs = bfs.iloc[:, 2:]
    Ybfs = bfs.iloc[:, 0]

    cutcp = p100_data[p100_data['app'] == 'cutcp']
    Xcutcp = cutcp.iloc[:, 2:]
    Ycutcp = cutcp.iloc[:, 0]

    kmeans = p100_data[p100_data['app'] == 'kmeans']
    Xkmeans = kmeans.iloc[:, 2:]
    Ykmeans = kmeans.iloc[:, 0]

    lavamd = p100_data[p100_data['app'] == 'lavamd']
    Xlavamd = lavamd.iloc[:, 2:]
    Ylavamd = lavamd.iloc[:, 0]

    cfd = p100_data[p100_data['app'] == 'cfd']
    Xcfd = cfd.iloc[:, 2:]
    Ycfd = cfd.iloc[:, 0]

    #"nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    nw = p100_data[p100_data['app'] == 'nw']
    Xnw = nw.iloc[:, 2:]
    Ynw = nw.iloc[:, 0]

    hotspot = p100_data[p100_data['app'] == 'hotspot']
    Xhotspot = hotspot.iloc[:, 2:]
    Yhotspot = hotspot.iloc[:, 0]

    lud = p100_data[p100_data['app'] == 'lud']
    Xlud = lud.iloc[:, 2:]
    Ylud = lud.iloc[:, 0]

    ge = p100_data[p100_data['app'] == 'ge']
    Xge = ge.iloc[:, 2:]
    Yge = ge.iloc[:, 0]

    srad = p100_data[p100_data['app'] == 'srad']
    Xsrad = srad.iloc[:, 2:]
    Ysrad = srad.iloc[:, 0]

    heartwall = p100_data[p100_data['app'] == 'heartwall']
    Xheartwall = heartwall.iloc[:, 2:]
    Yheartwall = heartwall.iloc[:, 0]

    bplustree = p100_data[p100_data['app'] == 'bplustree']
    Xbplustree = bplustree.iloc[:, 2:]
    Ybplustree = bplustree.iloc[:, 0]

    # Xtpacf = sc_x.fit_transform(Xtpacf.astype(float))
    # Xstencil = sc_x.fit_transform(Xstencil.astype(float))
    # Xlbm = sc_x.fit_transform(Xlbm.astype(float))
    # Xfft = sc_x.fit_transform(Xfft.astype(float))
    # Xspmv = sc_x.fit_transform(Xspmv.astype(float))
    # Xmriq = sc_x.fit_transform(Xmriq.astype(float))
    # Xhisto = sc_x.fit_transform(Xhisto.astype(float))
    # Xbfs = sc_x.fit_transform(Xbfs.astype(float))
    # Xcutcp = sc_x.fit_transform(Xcutcp.astype(float))
    # Xkmeans = sc_x.fit_transform(Xkmeans.astype(float))
    # Xlavamd = sc_x.fit_transform(Xlavamd.astype(float))
    # Xcfd = sc_x.fit_transform(Xcfd.astype(float))
    # Xnw = sc_x.fit_transform(Xnw.astype(float))
    # Xhotspot = sc_x.fit_transform(Xhotspot.astype(float))
    # Xlud = sc_x.fit_transform(Xlud.astype(float))
    # Xge = sc_x.fit_transform(Xge.astype(float))
    # Xsrad = sc_x.fit_transform(Xsrad.astype(float))
    # Xheartwall = sc_x.fit_transform(Xheartwall.astype(float))
    # Xbplustree = sc_x.fit_transform(Xbplustree.astype(float))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3333)

    from sklearn.linear_model import LinearRegression
    lm = LinearRegression(normalize=True)
    lm.fit(x_train,y_train)
    # get importance
    importance = lm.coef_

    vals = []
    for i,v in enumerate(importance):
        vals.append(v)
        print('Feature: %0d, Score: %.5f' % (i,v))

    plt.figure(1)
    labels = list(data.columns)
    plotdata = pd.DataFrame({
        "Features":vals
    }, 
        index=labels[2:]
    )
    colors = 'rgbkymc'  #red, green, blue, black, etc.
    plotdata.plot(kind="bar",color = colors,rot=60,legend = None )
    plt.xlabel("Features",weight='bold',fontsize=ls)
    plt.ylabel('Importance Score',weight='bold',fontsize=ls)
    # plt.margins(0.05)

    plt.xticks(fontsize=ts)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.tight_layout()
    plt.savefig('C:/rf/'+num_features+'featuresofimportance.png')
    # plt.show()
    print (labels[2:])

    plt.figure(2)

    y_train_pred = lm.predict(x_train)
    plt.scatter(y_train,y_train_pred,color='blue')
    plt.plot([y_train.min(), y_train.max()], [y_train_pred.min(), y_train_pred.max()], color = 'red', lw=2)
    plt.xlabel("Measured Power (Watt)",weight='bold',fontsize=ls)
    plt.ylabel("Predicted Power (Watt)",weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    #plt.title("Predictions vs. actual power values in the train set")
    # plt.xticks(fontsize=ts)
    plt.grid(True)
    plt.savefig('C:/rf/'+num_features+'pp100-prediction.png')

    ms = 75
    ls = 14
    ts = 14

    avg_pred_pwr = []
    max_orig_pwr = []
    diff_pwr = []

    # colors = itertools.cycle(['b', 'c', 'y', 'm', 'r'])
    colors = iter(cm.rainbow(np.linspace(0, 1, 19)))

    markers = ['o','+','*','.','x','_','|','s','d','^','v','>','<','p','h','X','8','1','2']

    #GIVE YOUR ANSWER FOR TASK-7 IN THIS CELL
    # plt.figure(3)
    plt.figure(figsize=(11, 11)) #figsize=(10, 10)

    # y_test_pred = lm.predict(x_test)
    Y_p100_pred = lm.predict(X_p100)
    # plt.scatter(Y_test,y_test_pred,color='r')
    plt.plot([Y_p100.min(), Y_p100.max()], [Y_p100_pred.min(), Y_p100_pred.max()], '--',color = 'red', lw=2)

    Ytpacf_pred = lm.predict(Xtpacf)
    p_tpacf = plt.scatter(Ytpacf, Ytpacf_pred, color=next(colors), linewidths = 1, marker =markers[0],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ytpacf_pred))
    max_orig_pwr.append(np.mean(Ytpacf))
    diff_pwr.append(np.mean(Ytpacf_pred)-np.mean(Ytpacf))
    err=[]
    e = metrics.mean_absolute_percentage_error(Ytpacf, Ytpacf_pred)
    err.append(e)

    Ystencil_pred = lm.predict(Xstencil)
    p_stencil = plt.scatter(Ystencil, Ystencil_pred, color=next(colors), linewidths = 1, marker =markers[1],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ystencil_pred))
    max_orig_pwr.append(np.mean(Ystencil))
    diff_pwr.append(np.mean(Ystencil_pred)-np.mean(Ystencil))
    e = metrics.mean_absolute_percentage_error(Ystencil, Ystencil_pred)
    err.append(e)

    Ylbm_pred = lm.predict(Xlbm)
    p_lbm = plt.scatter(Ylbm, Ylbm_pred, color=next(colors), linewidths = 1, marker =markers[2],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ylbm_pred))
    max_orig_pwr.append(np.mean(Ylbm))
    diff_pwr.append(np.mean(Ylbm_pred)-np.mean(Ylbm))
    e = metrics.mean_absolute_percentage_error(Ylbm, Ylbm_pred)
    err.append(e)

    Yfft_pred = lm.predict(Xfft)
    p_fft = plt.scatter(Yfft, Yfft_pred, color=next(colors), linewidths = 1, marker =markers[3],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Yfft_pred))
    max_orig_pwr.append(np.mean(Yfft))
    diff_pwr.append(np.mean(Yfft_pred)-np.mean(Yfft))
    e = metrics.mean_absolute_percentage_error(Yfft, Yfft_pred)
    err.append(e)

    Yspmv_pred = lm.predict(Xspmv)
    p_spmv = plt.scatter(Yspmv, Yspmv_pred, color=next(colors), linewidths = 1, marker =markers[4],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Yspmv_pred))
    max_orig_pwr.append(np.mean(Yspmv))
    diff_pwr.append(np.mean(Yspmv_pred)-np.mean(Yspmv))
    e = metrics.mean_absolute_percentage_error(Yspmv, Yspmv_pred)
    err.append(e)

    Ymriq_pred = lm.predict(Xmriq)
    p_mriq = plt.scatter(Ymriq, Ymriq_pred, color=next(colors), linewidths = 1, marker =markers[5],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ymriq_pred))
    max_orig_pwr.append(np.mean(Ymriq))
    diff_pwr.append(np.mean(Ymriq_pred)-np.mean(Ymriq))
    e = metrics.mean_absolute_percentage_error(Ymriq, Ymriq_pred)
    err.append(e)

    Yhisto_pred = lm.predict(Xhisto)
    p_histo = plt.scatter(Yhisto, Yhisto_pred, color=next(colors), linewidths = 1, marker =markers[6],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Yhisto_pred))
    max_orig_pwr.append(np.mean(Yhisto))
    diff_pwr.append(np.mean(Yhisto_pred)-np.mean(Yhisto))
    e = metrics.mean_absolute_percentage_error(Yhisto, Yhisto_pred)
    err.append(e)

    Ybfs_pred = lm.predict(Xbfs)
    p_bfs = plt.scatter(Ybfs, Ybfs_pred, color=next(colors), linewidths = 1, marker =markers[7],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ybfs_pred))
    max_orig_pwr.append(np.mean(Ybfs))
    diff_pwr.append(np.mean(Ybfs_pred)-np.mean(Ybfs))
    e = metrics.mean_absolute_percentage_error(Ybfs, Ybfs_pred)
    err.append(e)

    Ycutcp_pred = lm.predict(Xcutcp)
    p_cutcp = plt.scatter(Ycutcp, Ycutcp_pred, color=next(colors), linewidths = 1, marker =markers[8],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ycutcp_pred))
    max_orig_pwr.append(np.mean(Ycutcp))
    diff_pwr.append(np.mean(Ycutcp_pred)-np.mean(Ycutcp))
    e = metrics.mean_absolute_percentage_error(Ycutcp, Ycutcp_pred)
    err.append(e)

    Ykmeans_pred = lm.predict(Xkmeans)
    p_kmeans = plt.scatter(Ykmeans, Ykmeans_pred, color=next(colors), linewidths = 1, marker =markers[9],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ykmeans_pred))
    max_orig_pwr.append(np.mean(Ykmeans))
    diff_pwr.append(np.mean(Ykmeans_pred)-np.mean(Ykmeans))
    e = metrics.mean_absolute_percentage_error(Ykmeans, Ykmeans_pred)
    err.append(e)

    Ylavamd_pred = lm.predict(Xlavamd)
    p_lavamd = plt.scatter(Ylavamd, Ylavamd_pred, color=next(colors), linewidths = 1, marker =markers[10],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ylavamd_pred))
    max_orig_pwr.append(np.mean(Ylavamd))
    diff_pwr.append(np.mean(Ylavamd_pred)-np.mean(Ylavamd))
    e = metrics.mean_absolute_percentage_error(Ylavamd, Ylavamd_pred)
    err.append(e)

    Ycfd_pred = lm.predict(Xcfd)
    p_cfd = plt.scatter(Ycfd, Ycfd_pred, color=next(colors), linewidths = 1, marker =markers[11],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ycfd_pred))
    max_orig_pwr.append(np.mean(Ycfd))
    diff_pwr.append(np.mean(Ycfd_pred)-np.mean(Ycfd))
    e = metrics.mean_absolute_percentage_error(Ycfd, Ycfd_pred)
    err.append(e)

    Ynw_pred = lm.predict(Xnw)
    p_nw = plt.scatter(Ynw, Ynw_pred, color=next(colors), linewidths = 1, marker =markers[12],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ynw_pred))
    max_orig_pwr.append(np.mean(Ynw))
    diff_pwr.append(np.mean(Ynw_pred)-np.mean(Ynw))
    e = metrics.mean_absolute_percentage_error(Ynw, Ynw_pred)
    err.append(e)

    Yhotspot_pred = lm.predict(Xhotspot)
    p_hotspot = plt.scatter(Yhotspot, Yhotspot_pred, color=next(colors), linewidths = 1, marker =markers[13],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Yhotspot_pred))
    max_orig_pwr.append(np.mean(Yhotspot))
    diff_pwr.append(np.mean(Yhotspot_pred)-np.mean(Yhotspot))
    e = metrics.mean_absolute_percentage_error(Yhotspot, Yhotspot_pred)
    err.append(e)

    Ylud_pred = lm.predict(Xlud)
    p_lud = plt.scatter(Ylud, Ylud_pred, color=next(colors), linewidths = 1, marker =markers[14],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ylud_pred))
    max_orig_pwr.append(np.mean(Ylud))
    diff_pwr.append(np.mean(Ylud_pred)-np.mean(Ylud))
    e = metrics.mean_absolute_percentage_error(Ylud, Ylud_pred)
    err.append(e)

    Yge_pred = lm.predict(Xge)
    p_ge = plt.scatter(Yge, Yge_pred, color=next(colors), linewidths = 1, marker =markers[15],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Yge_pred))
    max_orig_pwr.append(np.mean(Yge))
    diff_pwr.append(np.mean(Yge_pred)-np.mean(Yge))
    e = metrics.mean_absolute_percentage_error(Yge, Yge_pred)
    err.append(e)

    Ysrad_pred = lm.predict(Xsrad)
    p_srad = plt.scatter(Ysrad, Ysrad_pred, color=next(colors), linewidths = 1, marker =markers[16],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ysrad_pred))
    max_orig_pwr.append(np.mean(Ysrad))
    diff_pwr.append(np.mean(Ysrad_pred)-np.mean(Ysrad))
    e = metrics.mean_absolute_percentage_error(Ysrad, Ysrad_pred)
    err.append(e)
    # ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    Yheartwall_pred = lm.predict(Xheartwall)
    p_heartwall = plt.scatter(Yheartwall, Yheartwall_pred, color=next(colors), linewidths = 1, marker =markers[17],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Yheartwall_pred))
    max_orig_pwr.append(np.mean(Yheartwall))
    diff_pwr.append(np.mean(Yheartwall_pred)-np.mean(Yheartwall))
    e = metrics.mean_absolute_percentage_error(Yheartwall, Yheartwall_pred)
    err.append(e)

    Ybplustree_pred = lm.predict(Xbplustree)
    p_bplustree = plt.scatter(Ybplustree, Ybplustree_pred, color=next(colors), linewidths = 1, marker =markers[18],  edgecolor =None,  s = ms)
    avg_pred_pwr.append(np.mean(Ybplustree_pred))
    max_orig_pwr.append(np.mean(Ybplustree))
    diff_pwr.append(np.mean(Ybplustree_pred)-np.mean(Ybplustree))
    e = metrics.mean_absolute_percentage_error(Ybplustree, Ybplustree_pred)
    err.append(e)

    plt.legend((p_tpacf,p_stencil,p_lbm,p_fft,p_spmv,p_mriq,p_histo,p_bfs,p_cutcp,p_kmeans,p_lavamd,p_cfd,p_nw,p_hotspot,p_lud,p_ge,p_srad,p_heartwall,p_bplustree),
            ("tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"),
            scatterpoints=1,
            loc='upper left',
            ncol=7,
            fontsize=10)
    # plt.plot(X_test, y_test_pred, color = "red")
    # plt.scatter(Y_test,y_test_pred,color='blue')
    # plt.plot([y_test.min(), y_test.max()], [y_test_pred.min(), y_test_pred.max()], color = 'black', lw=2)
    plt.xlabel("Measured Power (Watt)",weight='bold',fontsize=ls)
    plt.ylabel("Predicted Power (Watt)",weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.ylim(0,165)
    print ('Mean Percentage Error:',np.mean(err))
    # plt.title("Predictions vs. actual power values in the test set")
    plt.xticks(fontsize=ts)
    plt.grid(True)
    plt.savefig('C:/rf/'+num_features+'p100-bm-prediction.png')

    # Plot Error 
    apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    err = [i*100 for i in err]
    plt.figure()
    plt.style.use('classic')
    index = apps
    df = pd.DataFrame({'Measured Power (W)': max_orig_pwr,'Predicted Power (W)': avg_pred_pwr}, index=index)
    ax = df.plot.bar(color = 'rbg',rot=70) #rgbkymc
    plt.xlabel("Micro-benchmarks",weight='bold',fontsize=ls)
    plt.ylabel('Power (W)',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # plt.tick_params(axis='both', which='major', labelsize=12)
    plt.suptitle(chr(37)+' Error= Min: '+str(round(min(err),2))+',   Mean: '+str(round(np.mean(np.array(err)),2))+',   Max: '+str(round(max(err),2))+',   Std: '+str(round(np.std(np.array(err)),2)), fontsize=16,y=0.95)
    plt.tight_layout()
    plt.savefig('C:/rf/'+num_features+'bm_orig_measured.png')
    # plt.show()

    print ('avg length:', len(avg_pred_pwr),avg_pred_pwr)
    print ('Error:', len(err),err)

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