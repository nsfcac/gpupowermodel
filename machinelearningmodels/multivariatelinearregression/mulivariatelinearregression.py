import pandas as pd
import sys
from sklearn import linear_model
import tkinter as tk 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

def Plt_Importance (p100_data,importance,num_features,app,ls,ts):
    vals = []
    for i,v in enumerate(importance):
        vals.append(v)
        print('Feature: %0d, Score: %.5f' % (i,v))

    plt.figure()
    labels = list(p100_data.columns)
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
    plt.savefig('C:/rf/'+app+'-'+num_features+'featuresofimportance.png')
    # plt.show()
    print (labels[2:])

def global_fit_predict(p100_data,x_p100,y_p100,x_df,y_df,apps_list,ls,ts,ms,num_features,size):   
    
    x_p100_train, x_p100_test, y_p100_train, y_p100_test = train_test_split(x_p100, y_p100, test_size=size)

    lm = LinearRegression()
    lm.fit(x_p100_train,y_p100_train)
    # get importance
    importance = lm.coef_

    Plt_Importance(p100_data,importance,num_features,'global',ls,ts)
    
    avg_pred_pwr = []
    max_orig_pwr = []
    diff_pwr = []
    plt_apps = []
    err=[]
    # colors = itertools.cycle(['b', 'c', 'y', 'm', 'r'])
    colors = iter(cm.rainbow(np.linspace(0, 1, 19)))

    markers = ['o','+','*','.','x','_','|','s','d','^','v','>','<','p','h','X','8','1','2']
    mrk = 0
    plt.figure(figsize=(11, 11)) #figsize=(10, 10)
    y_p100_test_pred = lm.predict(x_p100_test)
    plt.plot([y_p100_test.min(), y_p100_test.max()], [y_p100_test_pred.min(), y_p100_test_pred.max()], '--',color = 'red', lw=2)

    for a in apps_list:
        y_pred = lm.predict(x_df[a])
        p_app=plt.scatter(y_df[a], y_pred, color=next(colors), linewidths = 1, marker =markers[mrk],  edgecolor =None,  s = ms)
        mrk += 1
        plt_apps.append(p_app)
        avg_pred_pwr.append(np.mean(y_pred))
        max_orig_pwr.append(np.mean(y_df[a]))
        diff_pwr.append(np.mean(y_pred)-np.mean(y_df[a]))
        e = metrics.mean_absolute_percentage_error(y_df[a], y_pred)
        err.append(e)

    plt.legend((plt_apps),
            (apps_list),
            scatterpoints=1,
            loc='upper left',
            ncol=7,
            fontsize=10)
    
    plt.xlabel("Measured Power (Watt)",weight='bold',fontsize=ls)
    plt.ylabel("Predicted Power (Watt)",weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.ylim(0,165)
    print ('Mean Percentage Error:',np.mean(err))
    # plt.title("Predictions vs. actual power values in the test set")
    plt.xticks(fontsize=ts)
    plt.grid(True)
    plt.savefig('C:/rf/'+num_features+'p100-bm-prediction.png')

    # Plot original vs predicted powers 
    err = [i*100 for i in err]
    plt.figure()
    plt.style.use('classic')
    index = apps_list
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

def app_fit_predict(p100_data,x_p100,y_p100,x_df,y_df,apps_list,ls,ts,ms,num_features,size):   
    
    lms = {}
    x_traindf = {}
    x_testdf = {}
    y_traindf = {}
    y_testdf = {}

    # x_p100_train, x_p100_test, y_p100_train, y_p100_test = train_test_split(x_p100, y_p100, test_size=0.3333)
    # lm = LinearRegression()
    # lm.fit(x_p100_train,y_p100_train)
    # importance = lm.coef_
    # Plt_Importance(importance,num_features,app)

    for app in apps_list:
        x_train, x_test, y_train, y_test = train_test_split(x_df[app], y_df[app], test_size=size)
        lm = LinearRegression()
        lm.fit(x_train,y_train)
        lms[app] = lm
        x_traindf[app] = x_train
        x_testdf[app] = x_test
        y_traindf[app] = y_train
        y_testdf[app] = y_test

        importance = lm.coef_
        # Plt_Importance(p100_data,importance,num_features,app,ls,ts)

    avg_pred_pwr = []
    max_orig_pwr = []
    diff_pwr = []
    plt_apps = []
    err=[]
    # colors = itertools.cycle(['b', 'c', 'y', 'm', 'r'])
    colors = iter(cm.rainbow(np.linspace(0, 1, 19)))
    markers = ['o','+','*','.','x','_','|','s','d','^','v','>','<','p','h','X','8','1','2']

    plt.figure(figsize=(11, 11)) #figsize=(10, 10)
    # y_p100_test_pred = lm.predict(x_p100_test)
    # plt.plot([y_p100_test.min(), y_p100_test.max()], [y_p100_test_pred.min(), y_p100_test_pred.max()], '--',color = 'red', lw=2)
    mrk = 0
    for a in apps_list:
        y_pred = lms[a].predict(x_testdf[a])
        p_app=plt.scatter(y_testdf[a], y_pred, color=next(colors), linewidths = 1, marker =markers[mrk],  edgecolor =None,  s = ms)
        mrk += 1
        plt_apps.append(p_app)
        avg_pred_pwr.append(np.mean(y_pred))
        max_orig_pwr.append(np.mean(y_testdf[a]))
        diff_pwr.append(np.mean(y_pred)-np.mean(y_testdf[a]))
        e = metrics.mean_absolute_percentage_error(y_testdf[a], y_pred)
        err.append(e)

    plt.legend((plt_apps),
            (apps_list),
            scatterpoints=1,
            loc='upper left',
            ncol=7,
            fontsize=10)
    
    plt.xlabel("Measured Power (Watt)",weight='bold',fontsize=ls)
    plt.ylabel("Predicted Power (Watt)",weight='bold',fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.ylim(0,165)
    print ('Mean Percentage Error:',np.mean(err))
    # plt.title("Predictions vs. actual power values in the test set")
    plt.xticks(fontsize=ts)
    plt.grid(True)
    plt.savefig('C:/rf/'+num_features+'p100-bm-prediction.png')

    # Plot original vs predicted powers 
    err = [i*100 for i in err]
    plt.figure()
    plt.style.use('classic')
    index = apps_list
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
