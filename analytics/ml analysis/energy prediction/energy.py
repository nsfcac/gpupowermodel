import sys
from numpy.core.fromnumeric import mean

import pandas as pd
from seaborn.palettes import color_palette
from sklearn import linear_model
import tkinter as tk 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# def toggle_fs(dummy=None):
#     state = False if root.attributes('-fullscreen') else True
#     root.attributes('-fullscreen', state)
#     if not state:
#         root.geometry('300x300+100+100')


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
import statistics

sys.path.insert(1, 'C:/rf/projects/gpupowermodel/analytics/ml analysis/ml models')

import mlr
import svr
import rfr
import xgbr

# plt.rcParams['axes.grid'] = True

ls = 14
ts = 14
ms = 75


import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class MyProblem(Problem):

    def __init__(self,xll,xuu):
        super().__init__(n_var=2,
                         n_obj=2,
                        #  n_constr=2,
                         xl=xll,
                         xu=xuu)

    def _evaluate(self, x, out, *args, **kwargs):
        print (x)
        f1 = x[0] ** 2 + x[1] ** 2
        f2 = (x[0] - 1) ** 2 + x[1] ** 2

        # g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        # g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        out["F"] = [f1, f2]
        # out["G"] = [g1, g2]

def pareto_decision(l,u,app):


    problem = MyProblem(l,u)
    algorithm = NSGA2(pop_size=2) #pop_size=100

    res = minimize(problem,
                   algorithm,
                #    ("n_gen", 100),
                   verbose=True,
                    seed=1)

    from pymoo.factory import get_problem, get_visualization, get_decomposition
    weights = np.array([0.5, 0.5])
    decomp = get_decomposition("asf")
    I = get_decomposition("asf").do(res.F, weights).argmin()
    print("Best regarding decomposition: Point %s - %s" % (I, res.F[I]))
    from pymoo.factory import get_visualization

    plot = get_visualization("scatter")
    plot.add(res.F, color="blue", alpha=0.2, s=10)
    plot.add(res.F[I], color="red", s=30)
    plot.do() #0.36, 0.16
    plot.apply(lambda ax: ax.arrow(0, 0, res.F[I][0], res.F[I][1], color='green',
                                   head_width=0.04, head_length=0.03, alpha=0.4))
    plot.show()

    # plot = Scatter()
    # plot.add(res.F, color="red")
    plot.save('C:/rf/results/misc/energy/pareto_'+app+'.png',transparent=True, bbox_inches='tight',dpi=600)
    # plot.show()
    # sys.exit(0)
# l = np.array([8,544])
# # u = np.array([57,1836])
# l = np.array([8,544])
# u = np.array([57,1836])
# app = 'bf'
# pareto_decision(l,u,app)
# sys.exit(0)

def ReadGPUFeatures():
    p100_freqs = [544, 556, 569, 582, 594, 607, 620, 632, 645, 658, 670, 683, 696, 708, 721, 734, 746, 759, 772, 784, 797, 810, 822, 835, 847, 860, 873, 885, 898, 911, 923, 936, 949, 961, 974, 987, 999, 1012, 1025, 1037, 1050, 1063, 1075, 1088, 1101, 1113, 1126, 1139, 1151, 1164, 1177, 1189, 1202, 1215, 1227, 1240, 1252, 1265, 1278, 1290, 1303, 1316, 1328]
    v100_freqs = [135, 142, 150, 157, 165, 172, 180, 187, 195, 202, 210, 217, 225, 232, 240, 247, 255, 262, 270, 277, 285, 292, 300, 307, 315, 322, 330, 337, 345, 352, 360, 367, 375, 382, 390, 397, 405, 412, 420, 427, 435, 442, 450, 457, 465, 472, 480, 487, 495, 502, 510, 517, 525, 532, 540, 547, 555, 562, 570, 577, 585, 592, 600, 607, 615, 622, 630, 637, 645, 652, 660, 667, 675, 682, 690, 697, 705, 712, 720, 727, 735, 742, 750, 757, 765, 772, 780, 787, 795, 802, 810, 817, 825, 832, 840, 847, 855, 862, 870, 877, 885, 892, 900, 907, 915, 922, 930, 937, 945, 952, 960, 967, 975, 982, 990, 997, 1005, 1012, 1020, 1027, 1035, 1042, 1050, 1057, 1065, 1072, 1080, 1087, 1095, 1102, 1110, 1117, 1125, 1132, 1140, 1147, 1155, 1162, 1170, 1177, 1185, 1192, 1200, 1207, 1215, 1222, 1230, 1237, 1245, 1252, 1260, 1267, 1275, 1282, 1290, 1297, 1305, 1312, 1320, 1327, 1335, 1342, 1350, 1357, 1365, 1372, 1380]
    
    gpu = pd.read_csv("C:/rf/SPEC_gpu_p100_v100_metrics.csv")
    dgemm = pd.read_csv("C:/rf/dgemm_v100_metrics.csv")
    stream = pd.read_csv("C:/rf/stream_v100_metrics.csv")
    print ('dgemm')
    print (dgemm)
    print ('stream')
    print (stream)
    dgemm.drop(['Unnamed: 0'], inplace = True, axis=1)
    stream.drop(['Unnamed: 0'], inplace = True, axis=1)
    gpu.drop(['Unnamed: 0'], inplace = True, axis=1)
    # gpu = gpu[gpu['clocks.current.sm [MHz]'].isin(p100_freqs)]
    # gpu = gpu[gpu['clocks.current.sm [MHz]'].isin(v100_freqs)]
    p100_data = gpu[gpu["arch"] == 'P100']  
    v100_data = gpu[gpu["arch"] == 'V100']

    p100_data = p100_data[p100_data['clocks.current.sm [MHz]'].isin(p100_freqs)]
    v100_data = v100_data[v100_data['clocks.current.sm [MHz]'].isin(v100_freqs)]
    dgemm = dgemm[dgemm['clocks.current.sm [MHz]'].isin(v100_freqs)]
    stream = stream[stream['clocks.current.sm [MHz]'].isin(v100_freqs)]

    print(p100_data.shape[0])
    print(v100_data.shape[0])

    cols = list(gpu.columns)
    print (cols)

    gpu.info()
    gpu.describe()

    # p100_data.to_csv("C:/rf/SPEC_GPU_P100_Features.csv", encoding='utf-8-sig') #mode='a', header=False,index=False,
    # v100_data.to_csv("C:/rf/SPEC_GPU_V100_Features.csv", encoding='utf-8-sig') #mode='a', header=False,index=False,

    return p100_data, v100_data, gpu, dgemm, stream

def RemoveGPUFeatures(p100_data, data, dgemm, stream):
    # data.drop(['arch','clocks.current.graphics [MHz]','cores','transistors','die_size','TDP','memory.total [MiB]','base_run_time'], inplace = True, axis=1)
    # p100_data.drop(['arch','clocks.current.graphics [MHz]','cores','transistors','die_size','TDP','memory.total [MiB]','base_run_time'], inplace = True, axis=1)
    data.drop(['arch','clocks.current.graphics [MHz]','TDP','memory.total [MiB]'], inplace = True, axis=1) #
    p100_data.drop(['arch','clocks.current.graphics [MHz]','TDP','memory.total [MiB]'], inplace = True, axis=1)
    # ,'base_run_time','app_type'
    dgemm.drop(['arch','clocks.current.graphics [MHz]','TDP','memory.total [MiB]'], inplace = True, axis=1) #'base_run_time'
    stream.drop(['arch','clocks.current.graphics [MHz]','TDP','memory.total [MiB]'], inplace = True, axis=1)

    # data['timestamp'] = pd.to_datetime(data['timestamp']).astype(np.int64) // 10**9
    # p100_data['timestamp'] = pd.to_datetime(p100_data['timestamp']).astype(np.int64) // 10**9
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.second
    p100_data['timestamp'] = pd.to_datetime(p100_data['timestamp']).dt.second
    dgemm['timestamp'] = pd.to_datetime(dgemm['timestamp']).dt.second
    stream['timestamp'] = pd.to_datetime(stream['timestamp']).dt.second

    return p100_data, data, dgemm, stream

def Plt_Single_Mem_Util_Box(data):
    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    plot_order = data.groupby("app")["utilization.memory (%)"].median().fillna(0).sort_values()[::-1].index

    sns.boxplot(x="app", y="utilization.memory (%)",data=data, orient="v", palette="hot",order=plot_order)
    # Set labels
    plt.ylabel("Memory Utilization (%)", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/v100_apps_gpu_mem_utilization_box_mem.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_Mem_GB_Util_Box(data):
    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    plot_order = data.groupby("app")["memory.used (MiB)"].median().fillna(0).sort_values()[::-1].index

    sns.boxplot(x="app", y="memory.used (MiB)",data=data, orient="v", palette="hot",order=plot_order)
    # Set labels
    plt.ylabel("Memory Used (MiB)", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/v100_apps_gpu_mem_utilization_box_mem_GB.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_GPU_Util_Box(data):
    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    plot_order = data.groupby("app")["utilization.gpu (%)"].median().fillna(0).sort_values()[::-1].index

    sns.boxplot(x="app", y="utilization.gpu (%)",data=data, orient="v", palette="hot",order=plot_order)
    # Set labels
    plt.ylabel("GPU Utilization (%)", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/v100_apps_gpu_mem_utilization_box_comp.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_GPU_Mem_Util_UR_Box():

    dat = pd.read_csv('C:/rf/SPEC_GPU_V100_Features_UR.csv')
    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    # plot_order = data.groupby("app")["utilization.gpu (%)"].median().fillna(0).sort_values()[::-1].index
    # data['combine_util'] = data["utilization.gpu (%)"] / data["utilization.memory (%)"]
    # data = data.sort_values(by=['combine_util'], ascending=False)
    plot_order = dat.groupby("app")["utilization.ratio"].median().fillna(0).sort_values()[::-1].index

    sns.boxplot(x="app", y="utilization.ratio",data=dat, orient="v", palette="hot", order=plot_order)
    # Set labels
    plt.ylabel("UR (%)", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/v100_apps_gpu_mem_utilization_box_ur.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_GPU_Mem_Util_ratio_Box (data):
    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    # plot_order = data.groupby("app")["utilization.gpu (%)"].median().fillna(0).sort_values()[::-1].index
    # data = data.loc[data["utilization.memory (%)"] >= 1]
    # (~(df.a < 0), other=0, inplace=True)
    # print ('LAVAMD ALL DATA: ',round(data[data['app']=='lavamd']["utilization.gpu (%)"]).tolist())
    noAppList = data.loc[data["utilization.memory (%)"] < 1]['app'].tolist()
    # data['utilization.memory (%)'].where(~(data['utilization.memory (%)'] < 1), other=1, inplace=True)
    
    # data.loc[(data["utilization.memory (%)"] < 1), ('["utilization.memory (%)"]')] = 1
    # gpua = data[data['app'] == 'lavamd']["utilization.gpu (%)"].tolist()
    # mema = data[data['app'] == 'lavamd']["utilization.memory (%)"].tolist()
    # print ('lavamd gpua',len(gpua),gpua)
    # print ('lavamd mem',len(mema),mema)
    print ('***TOTAL ELEMENTS***',noAppList)

    data['ratio_util'] = round((data["utilization.gpu (%)"] / data["utilization.memory (%)"]),2)
    # data = data.sort_values(by=['combine_util'], ascending=False)
    print ('RATIO',data)
    plot_order = data.groupby("app")["ratio_util"].median().fillna(0).sort_values()[::-1].index

    sns.boxplot(x="app", y="ratio_util",data=data, orient="v", palette="hot",order=plot_order)
    # Set labels
    plt.ylabel("GPU/Mem Utilization Ratio", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/v100_apps_gpu_mem_utilization_box_ratio.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_GPU_Mem_Util_combine_Box(data):

    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    # plot_order = data.groupby("app")["utilization.gpu (%)"].median().fillna(0).sort_values()[::-1].index
    data['combine_util'] = 0.3 * data["utilization.gpu (%)"] + 0.40 * data["utilization.memory (%)"]
    # data = data.sort_values(by=['combine_util'], ascending=False)
    plot_order = data.groupby("app")["combine_util"].median().fillna(0).sort_values()[::-1].index

    sns.boxplot(x="app", y="combine_util",data=data, orient="v", palette="hot",order=plot_order)
    # Set labels
    plt.ylabel("GPU/Mem Average Utilization", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/v100_apps_gpu_mem_utilization_box_combine.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_GPU_Mem_Util(data):
    
    data_a_df = data.groupby(['app']).mean().reset_index()
    data_app_df = data_a_df[['app','utilization.gpu [%]','utilization.memory [%]']]
    
    data_app_df = data_app_df.set_index('app')
    data_app_df = data_app_df.sort_values(by=['utilization.gpu [%]'],ascending=False)
    # rgbkymc,RdYlGn,RdYlBu_r
    # print (plt_df.head())
    ax = plt.figure(figsize=(12, 5)).add_subplot(111)
    # ax = plt.axes() colormap='gist_rainbow',
    plt.style.use('classic')
    data_app_df.plot(ax=ax, kind='bar', rot=70, legend=False,colormap='gist_rainbow')
    bars = ax.patches
    hatches = ''.join(h*len(data_app_df) for h in 'x/O.')
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
    plt.savefig('C:/rf/results/v100_apps_gpu_mem_utilization.png',transparent=True, bbox_inches='tight',dpi=300)

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

def Plt_Single_GPU_Mem_Util_ratio_BAR (data):
    
    data_a_df = data.copy()

    # noAppList = data.loc[data["utilization.memory (%)"] < 1]['app'].tolist()
    # print ('***TOTAL ELEMENTS***',noAppList)

    min_gpu_util= data_a_df[data_a_df['clocks.current.sm (MHz)'] == 135]
    max_gpu_util = data_a_df[data_a_df['clocks.current.sm (MHz)'] == 1380]
    min_gpu = min_gpu_util[['app','utilization.gpu (%)']]
    max_gpu = max_gpu_util[['app','utilization.gpu (%)']]
    min_mem = min_gpu_util[['app','utilization.memory (%)']]
    max_mem = max_gpu_util[['app','utilization.memory (%)']]

    max_gpu['max_ratio_util'] = round((max_gpu["utilization.gpu (%)"] / max_mem["utilization.memory (%)"]),2)
    max_gpu['min_ratio_util'] = round((min_gpu["utilization.gpu (%)"] / min_mem["utilization.memory (%)"]),2)
    
    df = pd.DataFrame({"GPU/Mem Ratio @ Min DVFS": max_gpu['min_ratio_util'].tolist(),
                    "GPU/Mem Ratio @ Max DVFS" : max_gpu['max_ratio_util'].tolist(),
                    "Benchmarks": max_gpu['app'].tolist()})

    print (df)
    ax = plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    df = df.sort_values(by=['GPU/Mem Ratio @ Max DVFS'],ascending=False)
    ax.bar(df["Benchmarks"], df["GPU/Mem Ratio @ Max DVFS"], color='deepskyblue', label = 'GPU/Mem Ratio @ Max DVFS')
    ax.bar(df["Benchmarks"],  df["GPU/Mem Ratio @ Min DVFS"], color='salmon', bottom=df['GPU/Mem Ratio @ Max DVFS'],label = 'GPU/Mem Ratio @ Min DVFS')
    bars = ax.patches
    hatches = ''.join(h*len(df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='best', fontsize = 12, ncol=2)
    plt.xticks(rotation=70)
    ax.set_xlabel("Benchmarks",weight='bold',fontsize=ls)
    ax.set_ylabel('GPU/Mem Utilization Ratio',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    plt.savefig('C:/rf/results/v100_apps_gpu_min_max_utilization_ratio.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_Mem_GB_Util_BAR(data):

    data_a_df = data.copy()
    min_gpu_util= data_a_df[data_a_df['clocks.current.sm (MHz)'] == 135]
    max_gpu_util = data_a_df[data_a_df['clocks.current.sm (MHz)'] == 1380]
    min_df = min_gpu_util[['app','memory.used (MiB)']]
    max_df = max_gpu_util[['app','memory.used (MiB)']]

    df = pd.DataFrame({"Memory Used (MiB) @ Min DVFS": min_df['memory.used (MiB)'].tolist(),
                    # "Memory Used (MiB) @ Max DVFS" : max_df['memory.used (MiB)'].tolist(),
                    "Benchmarks": min_df['app'].tolist()})

    df = df.set_index('Benchmarks')
    df = df.sort_values(by=['Memory Used (MiB) @ Min DVFS'],ascending=False)

    # sns.boxplot(x="app", y="memory.used (MiB)",data=data, orient="v", palette="hot",order=plot_order)
    # rgbkymc,RdYlGn,RdYlBu_r
    # print (plt_df.head())
    ax = plt.figure(figsize=(12, 5)).add_subplot(111)
    # ax = plt.axes() colormap='gist_rainbow',
    plt.style.use('classic')
    df.plot(ax=ax, kind='bar', rot=70, legend=False,colormap='gist_rainbow')
    bars = ax.patches
    hatches = ''.join(h*len(df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    # ax.legend(loc='best', fontsize = 12)
    # Set labels
    plt.ylabel("Memory Used (MiB)", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/v100_apps_gpu_mem_utilization_box_mem_GB_BAR.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_GPU_Util_BAR(data):
    data_a_df = data.copy()
    min_gpu_util= data_a_df[data_a_df['clocks.current.sm (MHz)'] == 135]
    max_gpu_util = data_a_df[data_a_df['clocks.current.sm (MHz)'] == 1380]
    min_df = min_gpu_util[['app','utilization.gpu (%)']]
    max_df = max_gpu_util[['app','utilization.gpu (%)']]

    df = pd.DataFrame({"GPU Util @ Min DVFS": min_df['utilization.gpu (%)'].tolist(),
                    "GPU Util @ Max DVFS" : max_df['utilization.gpu (%)'].tolist(),
                    "Benchmarks": min_df['app'].tolist()})

    print (df)
    ax = plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    df = df.sort_values(by=['GPU Util @ Max DVFS'],ascending=False)
    ax.bar(df["Benchmarks"], df["GPU Util @ Max DVFS"], color='deepskyblue', label = 'GPU Util @ Max DVFS',alpha=0.5)
    ax.bar(df["Benchmarks"],  df["GPU Util @ Min DVFS"], color='salmon', label = 'GPU Util @ Min DVFS',alpha=0.5)
    # ax.bar(df["Benchmarks"], df["GPU Util @ Max DVFS"], color='deepskyblue', label = 'GPU Util @ Max DVFS')
    # percent_df = df.apply(lambda x: (x * 100) / sum(x), axis=1)
    # df.plot.bar(x='Benchmarks', y=["GPU Util @ Max DVFS", "GPU Util @ Min DVFS"],  stacked=True,  width = 0.4, alpha=0.5) 
    # percent_df.plot.barh(stacked=True,align='center',alpha=0.5)
    bars = ax.patches
    hatches = ''.join(h*len(df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='best', fontsize = 12, ncol=2)
    # , 
    plt.xticks(rotation=70)
    plt.xlabel("Benchmarks",weight='bold',fontsize=ls)
    plt.ylabel('GPU Utilization (%)',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig('C:/rf/results/v100_apps_gpu_min_max_utilization.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_Mem_Util_BAR(data):
    data_a_df = data.copy()
    min_gpu_util= data_a_df[data_a_df['clocks.current.sm (MHz)'] == 135]
    max_gpu_util = data_a_df[data_a_df['clocks.current.sm (MHz)'] == 1380]
    min_df = min_gpu_util[['app','utilization.memory (%)']]
    max_df = max_gpu_util[['app','utilization.memory (%)']]

    df = pd.DataFrame({"Mem Util @ Min DVFS": min_df['utilization.memory (%)'].tolist(),
                    "Mem Util @ Max DVFS" : max_df['utilization.memory (%)'].tolist(),
                    "Benchmarks": min_df['app'].tolist()})

    print (df)
    ax = plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    df = df.sort_values(by=['Mem Util @ Max DVFS'],ascending=False)
    ax.bar(df["Benchmarks"], df["Mem Util @ Max DVFS"], color='deepskyblue', label = 'Mem Util @ Max DVFS',alpha=0.5)
    ax.bar(df["Benchmarks"],  df["Mem Util @ Min DVFS"], color='salmon',label = 'Mem Util @ Min DVFS',alpha=0.5)

    bars = ax.patches
    hatches = ''.join(h*len(df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='best', fontsize = 12, ncol=2)
    # , 
    plt.xticks(rotation=70)
    ax.set_xlabel("Benchmarks",weight='bold',fontsize=ls)
    ax.set_ylabel('Mem Utilization (%)',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig('C:/rf/results/v100_apps_mem_min_max_utilization.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Single_PWR_Util_BAR(data):
    data_a_df = data.copy()
    min_gpu_util= data_a_df[data_a_df['clocks.current.sm (MHz)'] == 135]
    max_gpu_util = data_a_df[data_a_df['clocks.current.sm (MHz)'] == 1380]
    min_df = min_gpu_util[['app','power.draw (W)']]
    max_df = max_gpu_util[['app','power.draw (W)']]

    df = pd.DataFrame({"Power Util @ Min DVFS": min_df['power.draw (W)'].tolist(),
                    "Power Util @ Max DVFS" : max_df['power.draw (W)'].tolist(),
                    "Benchmarks": min_df['app'].tolist()})

    print (df)
    ax = plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    df = df.sort_values(by=['Power Util @ Max DVFS'],ascending=False)
    ax.bar(df["Benchmarks"], df["Power Util @ Max DVFS"], color='deepskyblue', label = 'Power @ Max DVFS',alpha=0.5)
    ax.bar(df["Benchmarks"],  df["Power Util @ Min DVFS"], color='salmon', label = 'Power @ Min DVFS',alpha=0.5)

    bars = ax.patches
    hatches = ''.join(h*len(df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.axhline(250,label = "TDP",linestyle='dashed',c="r",lw=2)
    ax.set(ylim=(0, 255))
    ax.legend(loc='best', fontsize = 12, ncol=2)
    # , 
    plt.xticks(rotation=70)
    ax.set_xlabel("Benchmarks",weight='bold',fontsize=ls)
    ax.set_ylabel('Power (W)',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig('C:/rf/results/v100_apps_pwr_min_max_consumption.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_PERF_VS_Features(data):
    data_df = data.copy()
    apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    # fig, ax = plt.subplots(3, figsize=(8, 10))
    fig, axes = plt.subplots(2, 2, sharey=True)
    plt.style.use('classic')
    sns.set_context("paper", font_scale=2)
    # apps = ["hotspot"]
    for app in apps:
        d = data_df[data_df['app'] == app] # ) & (data_df['clocks.current.sm [MHz]'] == 1380)
        
        grpby = 'utilization.gpu [%]'
        df = d.copy()
        df = df.groupby(grpby).mean()
        print ('df size:',df.shape[0])
        df = df.sort_values(by=[grpby]) #,ascending=False
        sns.scatterplot(ax=axes[0,0], data=df, x=grpby, y='base_run_time', s=35, color="b", marker="s")
        axes[0,0].set_xlim([-5, 105])
        axes[0,0].set_xlabel( '(a) GPU Utilization (%)',weight='bold',fontsize=10) 
        axes[0,0].set_ylabel( 'Run Time (S)',weight='bold',fontsize=10) 
        # axes[0,0].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP')
        # axes[0,0].legend(loc="lower right", prop={'size': 8})

        grpby = 'utilization.memory [%]' #'clocks.current.sm [MHz]'  'utilization.gpu [%]'
        df = d.copy()
        df = df.groupby(grpby).mean().reset_index()
        df = df.sort_values(by=[grpby])        
        sns.scatterplot(ax=axes[0,1], data=df, x=grpby, y='base_run_time', s=35, color="g", marker="d")
        axes[0,1].set_xlim([-5, 105])
        axes[0,1].set_xlabel('(b) Memory Utilization (%)', weight='bold',fontsize=10)
        # axes[0,1].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP') 
        # axes[0,1].legend(loc="lower right", prop={'size': 8})
        
        grpby = 'clocks.current.sm [MHz]'
        df = data_df[data_df['app'] == app]
        df = df.groupby(grpby).mean().reset_index()
        df = df.sort_values(by=[grpby])
        sns.scatterplot(ax=axes[1,0], data=df, x=grpby, y='base_run_time', s=50, color="r", marker=".") 
        axes[1,0].set_xlabel( '(c) Core Frequency (MHz)',weight='bold',fontsize=10) 
        axes[1,0].set_ylabel( 'Run Time (S)',weight='bold',fontsize=10)
        # axes[1,0].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP')
        # axes[1,0].legend(loc="lower right", prop={'size': 8})

        grpby = 'power.draw [W]'
        df = d.copy()
        df = df.groupby(grpby).mean().reset_index()
        df = df.sort_values(by=[grpby])
        sns.scatterplot(ax=axes[1,1], data=df, x=grpby, y='base_run_time', s=70, color="m", marker="^")
        axes[1,1].set_xlabel( "(d) Power (W)",weight='bold',fontsize=10)
        # axes[1,1].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP')
        # axes[1,1].legend(loc="lower right", prop={'size': 8})

        fig.tight_layout()
        # plt.rcParams['axes.grid'] = True
        # plt.grid(True)
        plt.savefig('C:/rf/results/misc/perf/perf-vs-features'+app+'.png') #transparent=True



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

def Plt_Apps_Pwr_v100(data):
    
    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    plot_order = data.groupby("app")["power.draw (W)"].median().fillna(0).sort_values()[::-1].index

    g = sns.boxplot(x="app", y="power.draw (W)",data=data, orient="v", palette="hot",order=plot_order)
    # Set labels
    plt.ylabel("Power (W)", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    g.axhline(250,label = "TDP",linestyle='dashed',c="r",lw=2)
    g.set(ylim=(0, 255))

    plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/apps_pwr_pattern.png',transparent=True, bbox_inches='tight',dpi=300)

def Plt_Apps_Perf_v100():
    
    data = pd.read_csv('C:/rf/SPEC-DVFS-RUNTIME-V100.csv')
    # app_runtime_df.sort_values('Core Frequency (MHz)',ascending=True, inplace=True)

    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    # sns.set_context("paper", font_scale=1)
    # make boxplot with Seaborn
    # plot_order = data.groupby(by=["app"])["power.draw [W]"].median().iloc[::-1].index
    plot_order = data.groupby("Benchmarks")["EstBaseRunTime"].median().fillna(0).sort_values()[::-1].index
    

    sns.boxplot(x="Benchmarks", y="EstBaseRunTime",data=data, orient="v", palette="Set2",order=plot_order)
    # Set labels
    plt.ylabel("Execution Time (S)", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)
    plt.savefig('C:/rf/results/apps_perf_pattern.png',transparent=True, bbox_inches='tight',dpi=600)

def Plt_Apps_Energy_v100(data):
    data_df = data.copy()
    
    print ('Plt Apps Energy *3173*',data_df.shape[0])
    apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    runtime_df = pd.read_csv('C:/rf/SPEC-DVFS-RUNTIME-V100.csv')
    # apps = ["lud"] #hotspot
    s_size = 20
    app_df = pd.DataFrame()
    i = 0
    for app in apps:
        
        df = data_df[data_df['app'] == app] #) & (data_df['clocks.current.sm (MHz)'] == 1380)]
        df = df.copy()
        df.reset_index(drop=True)
        # grpby = 'utilization.gpu [%]'
        app_runtime_df = runtime_df[runtime_df['Benchmarks'] == app][['EstBaseRunTime','Core Frequency (MHz)']]
        app_runtime_df = app_runtime_df.copy()
        app_runtime_df.sort_values('Core Frequency (MHz)',ascending=True, inplace=True)
        
        app_runtime_df.index = df.index
        # df['base_run_time'] = app_runtime
        df[['base_run_time','clocks.current.sm (MHz)']] = app_runtime_df[['EstBaseRunTime','Core Frequency (MHz)']]
        df = df.reset_index(drop=True)
        app_df[app] = df['power.draw (W)'] * df['base_run_time']
        # app_df = app_df.set_index(app_df.index)
        # app_df = app_df.assign(app = df['power.draw (W)'] * df['base_run_time'])
        # app_df.insert(i, app, df['power.draw (W)'] * df['base_run_time'], True)
        # i = i + 1
    # app_df.reset_index()
    app_df = app_df.sort_values(by = apps)
    # plt_order = df.groupby(by=apps).median().iloc[::-1].index
    print (app_df)
    plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    sns.set_context("paper", font_scale=1.5)
    
    sns.boxplot(data=app_df, orient="v", palette="Set2",order=plt_order)
    # Set labels
    plt.ylabel("Energy (J)", weight='bold',fontsize=ls)
    plt.xlabel("Benchmarks", weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(rotation=70)
    plt.xticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12,ncol=1)
    plt.grid(True)

    plt.savefig('C:/rf/results/apps_energy_pattern.png', transparent=True, bbox_inches='tight', dpi=600)

def Plt_PWR_Feature_Cor(data):
    data_df = data.copy()
    
    print ('Plt_PWR_VS_FEATURES *3173*',data_df.shape[0])
    apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    
    apps = ["lud"] #hotspot
    s_size = 20
    for app in apps:
        
        # app_runtime = sorted(app_runtime)
        # fig, ax = plt.subplots(3, figsize=(8, 10))
        fig, axes = plt.subplots(2, 2, sharey=True)
        plt.style.use('classic')
        sns.set_context("paper", font_scale=1)

        df = data_df[data_df['app'] == app] #) & (data_df['clocks.current.sm (MHz)'] == 1380)]
        df = df.copy()
        df.reset_index(drop=True)
        # grpby = 'utilization.gpu [%]'
        # app_runtime_df = runtime_df[runtime_df['Benchmarks'] == app][['EstBaseRunTime','Core Frequency (MHz)']]
        # app_runtime_df = app_runtime_df.copy()
        # app_runtime_df.sort_values('Core Frequency (MHz)',ascending=True, inplace=True)
        
        # app_runtime_df.index = df.index
        # df[['base_run_time','clocks.current.sm (MHz)']] = app_runtime_df[['EstBaseRunTime','Core Frequency (MHz)']]
        # df = df.copy()

        # grpby = 'utilization.memory (%)' #'clocks.current.sm [MHz]'  'utilization.gpu [%]'
        # df = d.copy()
        # df = df.groupby(grpby).mean().reset_index()
        # df = df.sort_values(by=[grpby])
        df = df.sort_values(by='clocks.current.sm (MHz)')        
        sns.scatterplot(ax=axes[0,0], data=df, x='clocks.current.sm (MHz)', y='power.draw (W)', s=s_size, color="g", marker="d")
        # axes[0,0].set_xlim([100, 250])
        # axes[0,0].set_ylim([89, 86])
        axes[0,0].set_ylabel('Power (W)', weight='bold',fontsize=10)
        axes[0,0].set_xlabel('(a) Core Frequency (MHz)', weight='bold',fontsize=10)
        axes[0,0].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP') 
        axes[0,0].legend(loc=0, prop={'size': 8})

        # df = df[df['clocks.current.sm (MHz)'] == 1380]
        # df = df.copy()
        # df.reset_index(drop=True)

        df = df.sort_values(by='utilization.gpu (%)')
        sns.scatterplot(ax=axes[0,1], data=df, x='utilization.gpu (%)', y='power.draw (W)', s=s_size, color="b", marker="s")
        # axes[0,0].set_xlim([-5, 105])
        axes[0,1].set_xlabel( '(b) GPU Utilization (%)',weight='bold',fontsize=10) 
        axes[0,1].set_ylabel( 'Power (W)',weight='bold',fontsize=10) 
        axes[0,1].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP') 
        axes[0,1].legend(loc=0, prop={'size': 8})

        df = df.sort_values(by='utilization.memory (%)')
        sns.scatterplot(ax=axes[1,0], data=df, x='utilization.memory (%)', y='power.draw (W)', s=s_size, color="r", marker=".") 
        axes[1,0].set_xlabel( '(c) Memory Utilization (%)',weight='bold',fontsize=10) 
        axes[1,0].set_ylabel( 'Power (W)',weight='bold',fontsize=10)
        axes[1,0].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP') 
        axes[1,0].legend(loc=0, prop={'size': 8})

        df = df.sort_values(by='power.draw (W)')
        sns.scatterplot(ax=axes[1,1], data=df, y='power.draw (W)', x='temperature.gpu', s=s_size, color="m", marker="^")
        axes[1,1].set_xlabel( "(d) Temperature (C)",weight='bold',fontsize=10)
        axes[1,1].set_ylabel( "Power (W)",weight='bold',fontsize=10)
        axes[1,1].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP') 
        axes[1,1].legend(loc=0, prop={'size': 8})

        fig.tight_layout()
        plt.savefig('C:/rf/results/misc/pwr-features-corr-'+app+'.png', transparent=True,dpi=600)

def Plt_PERF_Feature_Cor(data):
    data_df = data.copy()
    
    print ('Plt_PERF_VS_Energy *3173*',data_df.shape[0])
    apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    runtime_df = pd.read_csv('C:/rf/SPEC-DVFS-RUNTIME-V100.csv')
    # apps = ["lud"] #hotspot
    s_size = 20
    for app in apps:
        
        # app_runtime = sorted(app_runtime)
        # fig, ax = plt.subplots(3, figsize=(8, 10))
        fig, axes = plt.subplots(2, 2, sharey=True)
        plt.style.use('classic')
        sns.set_context("paper", font_scale=1)

        df = data_df[data_df['app'] == app] #) & (data_df['clocks.current.sm (MHz)'] == 1380)]
        df = df.copy()
        df.reset_index(drop=True)
        # grpby = 'utilization.gpu [%]'
        app_runtime_df = runtime_df[runtime_df['Benchmarks'] == app][['EstBaseRunTime','Core Frequency (MHz)']]
        app_runtime_df = app_runtime_df.copy()
        app_runtime_df.sort_values('Core Frequency (MHz)',ascending=True, inplace=True)
        
        app_runtime_df.index = df.index
        df[['base_run_time','clocks.current.sm (MHz)']] = app_runtime_df[['EstBaseRunTime','Core Frequency (MHz)']]
        df = df.copy()

        # grpby = 'utilization.memory (%)' #'clocks.current.sm [MHz]'  'utilization.gpu [%]'
        # df = d.copy()
        # df = df.groupby(grpby).mean().reset_index()
        # df = df.sort_values(by=[grpby])
        df = df.sort_values(by='clocks.current.sm (MHz)')        
        sns.scatterplot(ax=axes[0,0], data=df, x='clocks.current.sm (MHz)', y='base_run_time', s=s_size, color="g", marker="d")
        # axes[0,0].set_xlim([100, 250])
        # axes[0,0].set_ylim([89, 86])
        axes[0,0].set_ylabel('Execution Time (S)', weight='bold',fontsize=10)
        axes[0,0].set_xlabel('(a) Core Frequency (MHz)', weight='bold',fontsize=10)
        # axes[0,1].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP') 
        # axes[0,1].legend(loc="lower right", prop={'size': 8})

        # df = df[df['clocks.current.sm (MHz)'] == 1380]
        # df = df.copy()
        # df.reset_index(drop=True)

        df = df.sort_values(by='utilization.gpu (%)')
        sns.scatterplot(ax=axes[0,1], data=df, x='utilization.gpu (%)', y='base_run_time', s=s_size, color="b", marker="s")
        # axes[0,0].set_xlim([-5, 105])
        axes[0,1].set_xlabel( '(b) GPU Utilization (%)',weight='bold',fontsize=10) 
        axes[0,1].set_ylabel( 'Execution Time (S)',weight='bold',fontsize=10) 
 
        df = df.sort_values(by='utilization.memory (%)')
        sns.scatterplot(ax=axes[1,0], data=df, x='utilization.memory (%)', y='base_run_time', s=s_size, color="r", marker=".") 
        axes[1,0].set_xlabel( '(c) Memory Utilization (%)',weight='bold',fontsize=10) 
        axes[1,0].set_ylabel( 'Execution Time (S)',weight='bold',fontsize=10)
 
        df = df.sort_values(by='power.draw (W)')
        sns.scatterplot(ax=axes[1,1], data=df, x='power.draw (W)', y='base_run_time', s=s_size, color="m", marker="^")
        axes[1,1].set_xlabel( "(d) Power (W)",weight='bold',fontsize=10)
        axes[1,1].set_ylabel( "Execution Time (S)",weight='bold',fontsize=10)
 
        fig.tight_layout()
        plt.savefig('C:/rf/results/energy/perf-features-corr-'+app+'.png', transparent=True,dpi=600)

def Plt_PERF_VS_Energy(data):
    data_df = data.copy()
    
    print ('Plt_PERF_VS_Energy *3173*',data_df.shape[0])
    apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    runtime_df = pd.read_csv('C:/rf/SPEC-DVFS-RUNTIME-V100.csv')
    # apps = ["lud"] #hotspot
    s_size = 20
    for app in apps:
        
        # app_runtime = sorted(app_runtime)
        # fig, ax = plt.subplots(3, figsize=(8, 10))
        fig, axes = plt.subplots(2, 3) #, sharey=True
        plt.style.use('classic')
        sns.set_context("paper", font_scale=1)

        df = data_df[data_df['app'] == app] #) & (data_df['clocks.current.sm (MHz)'] == 1380)]
        df = df.copy()
        df.reset_index(drop=True)
        # grpby = 'utilization.gpu [%]'
        app_runtime_df = runtime_df[runtime_df['Benchmarks'] == app][['EstBaseRunTime','Core Frequency (MHz)']]
        app_runtime_df = app_runtime_df.copy()
        app_runtime_df.sort_values('Core Frequency (MHz)',ascending=True, inplace=True)
        
        app_runtime_df.index = df.index
        # df['base_run_time'] = app_runtime
        df[['base_run_time','clocks.current.sm (MHz)']] = app_runtime_df[['EstBaseRunTime','Core Frequency (MHz)']]
        df['energy'] = df['power.draw (W)'] * df['base_run_time']
        # *******************for energy data for app
        # df = df.groupby(grpby).mean()
        # df = df.sort_values(by=[grpby]) #,ascending=False
        # print (app,'  ***DF SIZE:',df.shape[0])
        # print('Min energy:',df['energy'].min())
        # print('Max energy:',df['energy'].max())
        # print('Min base_run_time:',df['base_run_time'].min())
        # print('Max base_run_time:',df['base_run_time'].max())
        # df.to_csv('C:/rf/results/apps_energy_dvfs_runtime/'+app+'.csv')
        # continue
        # *******************for energy data for app

        # l = np.array([df['energy'].min(),df['base_run_time'].min()]) 
        # u = np.array([df['energy'].max(),df['base_run_time'].max()])
        # pareto_decision(l,u,app)

        # grpby = 'utilization.memory [%]' #'clocks.current.sm [MHz]'  'utilization.gpu [%]'
        # df = d.copy()
        # df = df.groupby(grpby).mean().reset_index()
        # df = df.sort_values(by=[grpby])
        df = df.sort_values(by='clocks.current.sm (MHz)')        
        sns.scatterplot(ax=axes[0,0], data=df, x='clocks.current.sm (MHz)', y='base_run_time', s=s_size, color="g", marker="d")
        # axes[0,0].set_xlim([100, 250])
        # axes[0,0].set_ylim([89, 86])
        axes[0,0].set_ylabel('Run Time (S)', weight='bold',fontsize=10)
        axes[0,0].set_xlabel('(a) Core Frequency (MHz)', weight='bold',fontsize=10)
        # axes[0,1].axhline(y=250,linestyle='dashed',c="y",linewidth=1,zorder=0,label='TDP') 
        # axes[0,1].legend(loc="lower right", prop={'size': 8})

        df = df.sort_values(by='base_run_time')
        sns.scatterplot(ax=axes[0,1], data=df, x='base_run_time', y='energy', s=s_size, color="b", marker="s")
        # axes[0,0].set_xlim([-5, 105])
        axes[0,1].set_xlabel( '(b) Run Time (S)',weight='bold',fontsize=10) 
        axes[0,1].set_ylabel( 'Energy (Joule)',weight='bold',fontsize=10) 
 
        df = df.sort_values(by='clocks.current.sm (MHz)')
        sns.scatterplot(ax=axes[1,0], data=df, x='clocks.current.sm (MHz)', y='energy', s=s_size, color="r", marker=".") 
        axes[1,0].set_xlabel( '(c) Core Frequency (MHz)',weight='bold',fontsize=10) 
        axes[1,0].set_ylabel( 'Energy (Joule)',weight='bold',fontsize=10)
 
        df = df.sort_values(by='clocks.current.sm (MHz)')
        sns.scatterplot(ax=axes[1,1], data=df, x='clocks.current.sm (MHz)', y='power.draw (W)', s=s_size, color="m", marker="^")
        axes[1,1].set_xlabel( "(d) Core Frequency (MHz)",weight='bold',fontsize=10)
        axes[1,1].set_ylabel( "Power (W)",weight='bold',fontsize=10)
 
        fig.tight_layout()
        plt.savefig('C:/rf/results/energy/perf-vs-energy'+app+'.png', transparent=True,dpi=600) 

def mcdm():
    OptimalExecutionTime = []
    OptimalEnergy = []
    DefaultExecutionTime = []
    DefaultEnergy = []
    
    apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    
    # apps = ["lud"] #hotspot 
    fig, axes = plt.subplots(4,3,figsize=(8, 8)) #,sharey=True 
    # plt.style.use('classic')
    sns.set_context("paper", font_scale=1)
    s_size = 20
    i,j,k = 0,0,0
    lbl = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
    app_freq = {}
    for app in apps:
        if j == 3:
            i += 1
            j = 0
        # fig, axes = plt.subplots() 
        # plt.style.use('classic')
        # sns.set_context("paper", font_scale=1)

        df = pd.read_csv('C:/rf/results/apps_energy_dvfs_runtime/'+app+'.csv')
        df = df[['clocks.current.sm (MHz)','base_run_time','energy']]
        df = df.copy()

        # df['base_run_time_su'] = (df['base_run_time']-df['base_run_time'].mean())/df['base_run_time'].std()
        # df['energy_su'] = (df['energy'] - df['energy'].mean())/df['energy'].std()

        df['n_base_run_time'] = df['base_run_time'].min()/df['base_run_time']
        df['n_energy'] = df['energy'].min()/df['energy']
        wt = 0.5
        we = 0.5
        df['n_base_run_time'] = wt * df['n_base_run_time']
        df['n_energy'] = we * df['n_energy']

        df['score'] =  df['n_base_run_time'] * df['n_energy']

        optimum_sol = df[df['score'] == max(df['score'])]

        freq = int(optimum_sol.iloc[0]['clocks.current.sm (MHz)'])
        x = int(optimum_sol.iloc[0]['base_run_time'])
        y = optimum_sol.iloc[0]['energy']

        app_freq[app] = freq

        OptimalExecutionTime.append(x)
        OptimalEnergy.append(y)
        DefaultExecutionTime.append(int(df[df['clocks.current.sm (MHz)'] == 1380]['base_run_time'].values[0]))
        DefaultEnergy.append(float(df[df['clocks.current.sm (MHz)'] == 1380]['energy'].values[0]))

    

    df_mcdn = pd.DataFrame({
        "Optimal Execution Time" : OptimalExecutionTime,
        'Optimal Energy' : OptimalEnergy,
        'Default Execution Time' : DefaultExecutionTime,
        'Default Energy' : DefaultEnergy
    },
    index = apps
    )
    df_mcdn2=df_mcdn.copy()
    mcdm_perf_eval(df_mcdn,'MCDM')
    mcdm_energy_save_eval(df_mcdn2,'MCDM')

    print ('*** MCDM APP FREQ ***')
    print (app_freq)

def edp():
    OptimalExecutionTime = []
    OptimalEnergy = []
    DefaultExecutionTime = []
    DefaultEnergy = []
    
    apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    
    # apps = ["lud"] #hotspot 
    fig, axes = plt.subplots(4,3,figsize=(8, 8)) #,sharey=True 
    # plt.style.use('classic')
    sns.set_context("paper", font_scale=1)
    s_size = 20
    i,j,k = 0,0,0
    app_freq = {}
    lbl = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
    for app in apps:
        if j == 3:
            i += 1
            j = 0
        # fig, axes = plt.subplots() 
        # plt.style.use('classic')
        # sns.set_context("paper", font_scale=1)

        df = pd.read_csv('C:/rf/results/apps_energy_dvfs_runtime/'+app+'.csv')
        df = df[['clocks.current.sm (MHz)','base_run_time','energy']]
        df = df.copy()

        df['n_base_run_time'] = (df['base_run_time']-df['base_run_time'].mean())/df['base_run_time'].std()
        df['n_energy'] = (df['energy'] - df['energy'].mean())/df['energy'].std()
        wt = 0.5
        we = 0.5
        df['n_base_run_time'] = wt * df['n_base_run_time']
        df['n_energy'] = we * df['n_energy']

        df['score'] =  (wt*df['base_run_time']) * (we*df['energy'])

        optimum_sol = df[df['score'] == min(df['score'])]

        freq = int(optimum_sol.iloc[0]['clocks.current.sm (MHz)'])
        x = int(optimum_sol.iloc[0]['base_run_time'])
        y = optimum_sol.iloc[0]['energy']

        app_freq[app] = freq
        app_freq[app] = x
        app_freq[app] = y

        OptimalExecutionTime.append(x)
        OptimalEnergy.append(y)
        DefaultExecutionTime.append(int(df[df['clocks.current.sm (MHz)'] == 1380]['base_run_time'].values[0]))
        DefaultEnergy.append(float(df[df['clocks.current.sm (MHz)'] == 1380]['energy'].values[0]))

    

    df_mcdn = pd.DataFrame({
        "Optimal Execution Time" : OptimalExecutionTime,
        'Optimal Energy' : OptimalEnergy,
        'Default Execution Time' : DefaultExecutionTime,
        'Default Energy' : DefaultEnergy
    },
    index = apps
    )
    df_mcdn2=df_mcdn.copy()
    mcdm_perf_eval(df_mcdn,'EDP')
    mcdm_energy_save_eval(df_mcdn2,'EDP')

    print ('*** EDP APP ***')
    print (app_freq)

def plt_edp():
    
    # apps = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    apps = ["kmeans","histo","lavamd","nw","lud","heartwall"]

    fig, axes = plt.subplots(2, 3,figsize=(9, 5)) #, sharey=True
 
    plt.style.use('classic')
    sns.set_context("paper", font_scale=1)

    s_size = 20
    i,j,k = 0,0,0
    app_freq = {}
    lbl = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
    for app in apps:
        if j == 3:
            i += 1
            j = 0
        
        df = pd.read_csv('C:/rf/results/apps_energy_dvfs_runtime/'+app+'.csv')
        df = df[['clocks.current.sm (MHz)','base_run_time','energy']]
        df = df.copy()

        wt = 0.5
        we = 0.5

        df['score'] =  (wt*df['base_run_time']) * (we*df['energy'])

        optimum_sol = df[df['score'] == min(df['score'])]
        freq = int(optimum_sol.iloc[0]['clocks.current.sm (MHz)'])
        x = int(optimum_sol.iloc[0]['base_run_time'])
        y = round(optimum_sol.iloc[0]['energy'])
        print (freq,x,y)
        sns.scatterplot(ax=axes[i,j], data=df, x='base_run_time', y='energy', s=s_size, color="blue", marker="d")
        
        axes[i,j].set_xlabel('('+lbl[k]+') '+app+' - Run Time (S)', weight='bold',fontsize=10)
        axes[i,j].set_ylabel('Energy (J)', weight='bold',fontsize=10)
        # Reference the optimal energy profile
        axes[i,j].annotate(
            text=str(freq)+'MHz '+str(y)+'J '+str(x)+'S',
            xy=(x, y),
            xycoords='data',
            fontsize=11,
            xytext=(df['base_run_time'].min(), df['energy'].max()),
            # textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red')  # Use color black
            # horizontalalignment='left',  # Center horizontally
            # verticalalignment='left'
            )  # Center vertically
        j = j + 1
        k = k + 1
    fig.tight_layout()
    plt.savefig('C:/rf/results/edp/edp_perf-energy.png', transparent=True,dpi=600)

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))
def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.mean())
def geo_mean_api(vals):
    return statistics.geometric_mean(vals)

def mcdm_perf_eval (df,lbl):
    perf_df = df[['Optimal Execution Time','Default Execution Time']]
    perf_df = perf_df.copy()

    print('PERF Geometric Mean START')
    perf_ratio =  perf_df['Optimal Execution Time'] / perf_df['Default Execution Time']
    # s = (perf_df['diff'] != 0)#.any(axis=1)
    # new_df = perf_df.loc[s]
    # new_df = new_df.copy()
    # new_df = perf_df
    # new_df = new_df.copy()
    '''
    new_df['percent_diff'] = new_df['diff']/new_df['Default Execution Time'] * 100
    print ('NEW DF',new_df)
    '''
    print ('GM-PROD:',geo_mean(perf_ratio.values))
    print ('GM-LOG:',geo_mean_overflow(perf_ratio.values))
    print ('GM-API:',geo_mean_api(perf_ratio.to_list()))
    print('PERF Geometric Mean END')

    print (perf_df[(perf_df['Default Execution Time'] - perf_df['Optimal Execution Time']) != 0])

    plt_df = perf_df.sort_values(by=['Default Execution Time'],ascending=False)
    
    # rgbkymc,RdYlGn,RdYlBu_r
    # print (plt_df.head())

    ax = plt.figure(figsize=(12, 5)).add_subplot(111)
    # ax = plt.axes() colormap='gist_rainbow',
    plt.style.use('classic')
    plt_df.plot(ax=ax, kind='bar', rot=70, colormap='gist_rainbow',legend=False)
    bars = ax.patches
    hatches = ''.join(h*len(plt_df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='best', fontsize = 12, ncol=2)
    # , 
    ax.set_xlabel("Benchmarks",weight='bold',fontsize=ls)
    ax.set_ylabel('Execution Time (S)',weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12)
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig('C:/rf/results/'+lbl+'-perf-energy.png',transparent=True, bbox_inches='tight',dpi=600)

def mcdm_energy_save_eval (df,lbl):
    # print ('ENERGY')
    # print (df)
    # df.to_csv('C:/rf/ep.csv')
    # print ('ENERGY')
    energy_ratio = df['Optimal Energy'] / df['Default Energy']
    
    df['Percent'] = round(((df['Default Energy'] - df['Optimal Energy'])/df['Default Energy'])*100,2)
    
    percent_df = df['Percent']
    percent_df = percent_df.copy()
    print (percent_df)

    print('ENERGY Geometric Mean START')
    print ('GM-PROD:',geo_mean(energy_ratio.values))
    print ('GM-LOG:',geo_mean_overflow(energy_ratio.values))
    print ('GM-API:',geo_mean_api(energy_ratio.to_list()))
    print('ENERGY Geometric Mean END')

    colors = 'Spectral'  # red, green, blue, black, etc.
    # pltdata = percent_df.sort_values('Percent')
    percent_df = percent_df.to_frame()
    pltdata = percent_df.sort_values('Percent',ascending=False)
    ax = plt.figure(figsize=(12, 5)).add_subplot(111)
    plt.style.use('classic')
    pltdata.plot(kind="bar", colormap=colors, rot=60, legend=None)
    plt.xlabel("Benchmarks", weight='bold', fontsize=ls)
    plt.ylabel('Energy Saving (%)', weight='bold', fontsize=ls)
    # plt.margins(0.05)
    plt.xticks(fontsize=ts)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.tight_layout()
    plt.savefig('C:/rf/results/'+lbl+'-eneghy_saving.png',transparent=True, bbox_inches='tight',dpi=600)

def remove_2_apps (data,apps):
    print ('data:', data.shape[0])
    df_17apps = data[(data.app != apps[0]) & (data.app != apps[1])]
    print ('df_17apps:', df_17apps.shape[0])

    df_app1 = data[data.app == apps[0]]
    print ('df_app1:', apps[0],' size:', df_app1.shape[0] )

    df_app2 = data[data.app == apps[1]]
    print ('df_app2:', apps[1],' size:', df_app2.shape[0] )

    return df_17apps, df_app1, df_app2
# mcdm()
# edp()
# plt_edp()
Plt_Single_GPU_Mem_Util_UR_Box()
sys.exit()

p100_data, data, gpu_data, dgemm, stream = ReadGPUFeatures()

p100_data, data, dgemm, stream = RemoveGPUFeatures(p100_data, data, dgemm, stream)

# Plt_Single_GPU_Mem_Util(data)
# sys.exit(0)
# Plt_GPU_Mem_Util(p100_data, data)
# Plt_Apps_Pwr(gpu_data)

# cols = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]']
# cols = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]','timestamp','temperature.gpu','clocks.current.memory [MHz]','memory.used [MiB]','memory.free [MiB]']        
# cls = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]','timestamp','temperature.gpu','clocks.current.memory [MHz]','memory.used [MiB]','memory.free [MiB]']
cls = ['power.draw [W]', 'app', 'utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]']
# cols = ['power.draw [W]', 'utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]']
# , 'memory.used [MiB]'
# 'base_run_time', ,'die_size','transistors','cores' ,'timestamp','temperature.gpu' ,'cores'
n = 0
n_features = 2
for i in range(n_features):
    if i == 1:
        break
    # cols = cls[:-i]
    # if i == 0:
    #     cols = cls
    # n = len(cols)-2
    # if i == 6:
    #     cols = cls[:-i+1]
    # print (i,n,cols)
    
    # num_features = str(n)+'_features_'

    p100_data = p100_data.reindex(columns=cls)
    data = data.reindex(columns=cls)
    gpu_data = gpu_data.reindex(columns=cls)
    dgemm = dgemm.reindex(columns=cls)
    stream = stream.reindex(columns=cls)


    # print(p100_data.shape[0])
    # print(data.shape[0])
    # print(gpu_data.shape[0])

    # Plt_PERF_VS_Features(data)
    # sys.exit(0)
    
    # gpu_data = data[data['utilization.gpu [%]'] > 0]
    # print ('utilization.gpu [%] > 0 :',gpu_data.shape[0])
    # print (gpu_data.head())
    # mem_data = data[data['utilization.memory [%]'] > 0]
    # print ('utilization.memory [%] > 0 :',mem_data.shape[0])
    # print (mem_data.head())

    data = data[data['power.draw [W]'] > 28]
    p100_data = p100_data[p100_data['power.draw [W]'] > 28]
    dgemm = dgemm[dgemm['power.draw [W]'] > 45]
    stream = stream[stream['power.draw [W]'] > 45]
    # print ('power.draw [W] > 0 :',pwr_data.shape[0])
    # print (pwr_data.head())
    # print ('Frequencies:',p100_data.groupby('app')['clocks.current.sm [MHz]'].nunique())
    # print ('Apps:',len(gpu_data.groupby('app')))
    # sys.exit(0)

    p100_data = p100_data.groupby(['app', 'clocks.current.sm [MHz]']).mean().reset_index()
    data = data.groupby(['app', 'clocks.current.sm [MHz]']).mean().reset_index()
    gpu_data = pd.concat([p100_data,data]).reset_index()
    dgemm = dgemm.groupby(['app','clocks.current.sm [MHz]']).mean().reset_index()
    stream = stream.groupby(['app','clocks.current.sm [MHz]']).mean().reset_index()
    print ('data size',data.shape[0])
    print ('dgemm',dgemm.shape[0])
    print ('stream',stream.shape[0])
    data_dgemm_stream = pd.concat([data,dgemm,stream])
    data_dgemm_stream.reset_index(inplace=True,drop=True)
    # data.append(dgemm, ignore_index = True)
    # data.append(stream, ignore_index = True)
    # data = data.groupby(['app', 'clocks.current.sm [MHz]']).mean().reset_index()
    # print ('***data_dgemm_stream***',data_dgemm_stream)
    # if num_features == '2_features_':
        # cols = ['power.draw [W]', 'app','utilization.gpu [%]','utilization.memory [%]']
    p100_data = p100_data.reindex(columns=cls)
    data = data.reindex(columns=cls)
    gpu_data = gpu_data.reindex(columns=cls)
    dgemm = dgemm.reindex(columns=cls)
    stream = stream.reindex(columns=cls)
    data_dgemm_stream = data_dgemm_stream.reindex(columns=cls)
    
    p100_data.rename(columns={'memory.used [MiB]':'memory.used (MiB)','power.draw [W]': 'power.draw (W)','utilization.gpu [%]':'utilization.gpu (%)','utilization.memory [%]':'utilization.memory (%)','clocks.current.sm [MHz]':'clocks.current.sm (MHz)'}, inplace=True)
    data.rename(columns={'memory.used [MiB]':'memory.used (MiB)','power.draw [W]': 'power.draw (W)','utilization.gpu [%]':'utilization.gpu (%)','utilization.memory [%]':'utilization.memory (%)','clocks.current.sm [MHz]':'clocks.current.sm (MHz)'}, inplace=True)
    dgemm.rename(columns={'memory.used [MiB]':'memory.used (MiB)','power.draw [W]': 'power.draw (W)','utilization.gpu [%]':'utilization.gpu (%)','utilization.memory [%]':'utilization.memory (%)','clocks.current.sm [MHz]':'clocks.current.sm (MHz)'}, inplace=True)
    stream.rename(columns={'memory.used [MiB]':'memory.used (MiB)','power.draw [W]': 'power.draw (W)','utilization.gpu [%]':'utilization.gpu (%)','utilization.memory [%]':'utilization.memory (%)','clocks.current.sm [MHz]':'clocks.current.sm (MHz)'}, inplace=True)
    gpu_data.rename(columns={'memory.used [MiB]':'memory.used (MiB)','power.draw [W]': 'power.draw (W)','utilization.gpu [%]':'utilization.gpu (%)','utilization.memory [%]':'utilization.memory (%)','clocks.current.sm [MHz]':'clocks.current.sm (MHz)'}, inplace=True)
    data_dgemm_stream.rename(columns={'memory.used [MiB]':'memory.used (MiB)','power.draw [W]': 'power.draw (W)','utilization.gpu [%]':'utilization.gpu (%)','utilization.memory [%]':'utilization.memory (%)','clocks.current.sm [MHz]':'clocks.current.sm (MHz)'}, inplace=True)
    # print (data_dgemm_stream)
    
    # Plt_PERF_VS_Energy(data)
    
    # Plt_PERF_Feature_Cor(data)
    # Plt_PWR_Feature_Cor(data)
    # Plt_Single_GPU_Util_Box(data_dgemm_stream)
    # Plt_Single_Mem_Util_Box (data_dgemm_stream)
    # Plt_Single_Mem_GB_Util_Box (data_dgemm_stream)
    # Plt_Apps_Pwr_v100 (data_dgemm_stream)
    # Plt_Apps_Perf_v100()
    # Plt_Apps_Energy_v100(data)

    # STACKED BAR PLOT    
    # Plt_Single_PWR_Util_BAR(data_dgemm_stream)
    # Plt_Single_Mem_Util_BAR(data_dgemm_stream)
    # Plt_Single_GPU_Util_BAR(data_dgemm_stream)
    # Plt_Single_Mem_GB_Util_BAR (data_dgemm_stream)
    # Plt_Single_GPU_Mem_Util_ratio_BAR (data_dgemm_stream)
    
    # Plt_Single_GPU_Mem_Util_ratio_Box (data_dgemm_stream)
    # Plt_Single_GPU_Mem_Util_combine_Box (data_dgemm_stream)
    
    # sys.exit(0)# ,,'clocks.current.sm (MHz)'

    # p100_data.drop(['power.draw (W)'], inplace = True, axis=1)
    # data.drop(['power.draw (W)'], inplace = True, axis=1)
    # gpu_data.drop(['power.draw (W)'], inplace = True, axis=1)
    apps_list = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree"]
    # ,'dgemm','stream'
    twenty1_apps = False
    if twenty1_apps:
        data = data_dgemm_stream
        apps_list = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","ge","srad","heartwall","bplustree",'dgemm','stream']
    num_features = '3_features_'

    data.fillna(method='pad')
    p100_data.fillna(method='pad')
    gpu_data.fillna(method='pad')
    dgemm.fillna(method='pad')
    stream.fillna(method='pad')
    p100_data.head()
    data.head()
    gpu_data.head()

    x = data.iloc[:, 2:]#.values.reshape(-1, 1)
    y = data.iloc[:, 0]#.values.reshape(-1, 1)
    
    seventeen_apps = True
    if seventeen_apps:
        apps_list = ["tpacf","stencil","lbm","fft","spmv","mriq","histo","bfs","cutcp","kmeans","lavamd","cfd","nw","hotspot","lud","heartwall","bplustree"]
        apps = ['ge','srad']
        data, app1_data, app2_data = remove_2_apps(data,apps)
        x_app1 = app1_data.iloc[:, 2:]
        y_app1 = app1_data.iloc[:, 0]
        x_app2 = app2_data.iloc[:, 2:]
        y_app2 = app2_data.iloc[:, 0]

    print ('***DATA***')
    print (data)
    print ('***DATA***')

    x_p100 = p100_data.iloc[:, 2:]
    y_p100 = p100_data.iloc[:, 0]

    x_gpu = gpu_data.iloc[:, 2:]
    y_gpu = gpu_data.iloc[:, 0]

    x_dgemm = dgemm.iloc[:, 2:]#.values.reshape(-1, 1)
    y_dgemm = dgemm.iloc[:, 0]#.values.reshape(-1, 1)
    x_stream = stream.iloc[:, 2:]#.values.reshape(-1, 1)
    y_stream = stream.iloc[:, 0]#.values.reshape(-1, 1)

    x_p100_df = {}
    y_p100_df = {}

    x_v100_df = {}
    y_v100_df = {}
    
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

    # pltPath = 'C:/rf/results/svr/perf/'
    # print ('******************************************* SVR ************************************************')
    # svr.app_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # svr.global_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # svr.crossarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # pltPath = 'C:/rf/results/inter-arch/'
    pltPath = 'C:/rf/results/rfr/pwr/'
    print ('******************************************* RFR ************************************************')
    # label = 'Run Time (S)'
    label = 'Power (W)'
    # rfr.app_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath,label)
    # rfr.global_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath,label)
    # rfr.crossarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath,label)
    rfr.new_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath,x_dgemm, y_dgemm, x_stream, y_stream, label,'dgemm','stream')
    rfr.new_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath,x_app1, y_app1, x_app2, y_app2, label,apps[0],apps[1])
    
    # pltPath = 'C:/rf/results/mlr/perf/'
    # print ('******************************************* MLR ************************************************')
    # mlr.app_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # mlr.global_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # mlr.crossarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # pltPath = 'C:/rf/results/xgbr/pwr/'
    # print ('******************************************* XGBR ************************************************')
    # xgbr.app_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # xgbr.global_fit_predict(data,x,y,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # xgbr.crossarch_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath)
    # xgbr.new_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath,x_dgemm, y_dgemm, x_stream, y_stream, label)
    
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