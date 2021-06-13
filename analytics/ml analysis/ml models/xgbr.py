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

from xgboost import XGBRegressor


def Plt_Importance(data, importance, num_features, app, ls, ts, pltPath):
    vals = []
    for i, v in enumerate(importance):
        vals.append(v)
        print('Feature: %0d, Score: %.5f' % (i, v))

    plt.figure()
    labels = list(data.columns)
    plotdata = pd.DataFrame({
        "Features": vals
    },
        index=labels[2:]
    )
    colors = 'rgbkymc'  # red, green, blue, black, etc.
    plotdata = plotdata.sort_values('Features')
    plotdata.plot(kind="bar", color=colors, rot=60, legend=None)
    plt.xlabel("Features", weight='bold', fontsize=ls)
    plt.ylabel('Importance Score', weight='bold', fontsize=ls)
    # plt.margins(0.05)

    plt.xticks(fontsize=ts)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.tight_layout()
    plt.savefig(pltPath+'-'+num_features+app+'-'+'featuresofimportance.png')
    # plt.show()
    print(labels[2:])

def new_app_fit_predict(gpu_data,x_gpu,y_gpu,p100_data,data,x,y,x_p100,y_p100,x_p100_df,y_p100_df,x_v100_df,y_v100_df,apps_list,ls,ts,ms,num_features,size,pltPath,x_dgemm, y_dgemm, x_stream, y_stream, label):
    x_p100_train, x_p100_test, y_p100_train, y_p100_test = train_test_split(
        x_p100, y_p100, test_size=size)

    lm = XGBRegressor()
    lm.fit(x, y)
    # get importance
    importance = lm.feature_importances_
    # Plt_Importance(gpu_data, importance, num_features,
                #    'NEWAPP', ls, ts, pltPath)

    avg_pred_pwr = []
    max_orig_pwr = []
    diff_pwr = []
    plt_apps = []
    err = []
    mse_list = []
    rmse_list = []
    mae_list = []
    evs_list = []
    r2s_list = []
    medae_list = []
    # colors = itertools.cycle(['b', 'c', 'y', 'm', 'r'])
    colors = iter(cm.rainbow(np.linspace(0, 1, 19)))

    markers = ['o', '+', '*', '.', 'x', '_', '|', 's', 'd',
               '^', 'v', '>', '<', 'p', 'h', 'X', '8', '1', '2']
    mrk = 0
    plt.figure(figsize=(12, 11)).add_subplot(111)  # figsize=(10, 10)
    plt.style.use('classic')
    y_dgemm_pred = lm.predict(x_dgemm)
    plt.plot([y_dgemm.min(), y_dgemm.max()], [y_dgemm_pred.min(),
                                            y_dgemm_pred.max()], '--', color='red', lw=2,label = 'DGEMM Regression Line')
    y_stream_pred = lm.predict(x_stream)
    plt.plot([y_stream.min(), y_stream.max()], [y_stream_pred.min(),
                                            y_stream_pred.max()], '--', color='blue', lw=2,label = 'STREAM Regression Line')
    apps_list = []
    # for a in apps_list:
    # y_pred = lm.predict(x_p100_df[a])
    p_app = plt.scatter(y_dgemm, y_dgemm_pred, color=next(
            colors), linewidths=1, marker=markers[mrk],  edgecolor=None,  s=ms)
    mrk += 1
    plt_apps.append(p_app)
    apps_list.append('DGEMM')
    avg_pred_pwr.append(np.mean(y_dgemm_pred))
    max_orig_pwr.append(np.mean(y_dgemm))
    diff_pwr.append(np.mean(y_dgemm_pred)-np.mean(y_dgemm))
    e = metrics.mean_absolute_percentage_error(y_dgemm, y_dgemm_pred)
    err.append(e)
    mse_list.append(mse(y_dgemm, y_dgemm_pred))
    rmse_list.append(rmse(y_dgemm, y_dgemm_pred))
    medae_list.append(medae(y_dgemm, y_dgemm_pred))
    mae_list.append(mae(y_dgemm, y_dgemm_pred))
    evs_list.append(evs(y_dgemm, y_dgemm_pred))
    r2s_list.append(r2s(y_dgemm, y_dgemm_pred))

    p_app = plt.scatter(y_stream, y_stream_pred, color=next(
            colors), linewidths=1, marker=markers[mrk],  edgecolor=None,  s=ms)
    mrk += 1
    plt_apps.append(p_app)
    apps_list.append('STREAM')
    avg_pred_pwr.append(np.mean(y_stream_pred))
    max_orig_pwr.append(np.mean(y_stream))
    diff_pwr.append(np.mean(y_stream_pred)-np.mean(y_stream))
    e = metrics.mean_absolute_percentage_error(y_stream, y_stream_pred)
    err.append(e)
    mse_list.append(mse(y_stream, y_stream_pred))
    rmse_list.append(rmse(y_stream, y_stream_pred))
    medae_list.append(medae(y_stream, y_stream_pred))
    mae_list.append(mae(y_stream, y_stream_pred))
    evs_list.append(evs(y_stream, y_stream_pred))
    r2s_list.append(r2s(y_stream, y_stream_pred))


    plt.legend((plt_apps),
               (apps_list),
               scatterpoints=1,
               loc='upper left',
               ncol=2,
               fontsize=11)

    plt.xlabel("Measured "+label, weight='bold', fontsize=ls)
    plt.ylabel("Predicted "+label, weight='bold', fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.ylim(0,165)
    print('Mean Percentage Error:', np.mean(err))
    # plt.title("Predictions vs. actual power values in the test set")
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')

    plt.grid(True)
    plt.savefig(pltPath+num_features+'-bm-prediction-NEWAPP.png',
                transparent=True, bbox_inches='tight',dpi=600)

    # Plot original vs predicted powers
    err = [i*100 for i in err]

    ax = plt.figure(figsize=(12, 9)).add_subplot(111)
    plt.style.use('classic')
    index = apps_list
    df = pd.DataFrame({'Measured '+label: max_orig_pwr,
                       'Predicted '+label: avg_pred_pwr}, index=index)
    df = df.sort_values('Measured '+label)
    df.plot(ax=ax, kind='bar', color='rbg', rot=70, legend=False)
    bars = ax.patches
    hatches = ''.join(h*len(df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='upper left', ncol=1) #bbox_to_anchor=(1, 1),
    plt.xlabel("Benchmarks", weight='bold', fontsize=ls)
    plt.ylabel(label, weight='bold', fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.grid(True)
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # chr(37)+
    plt.suptitle('MAE - Min: '+str(round(min(err), 2))+',   Mean: '+str(round(np.mean(np.array(err)), 2)) +
                 ',   Max: '+str(round(max(err), 2))+',   Std: '+str(round(np.std(np.array(err)), 2)), fontsize=14, weight='bold', y=0.93)
    plt.savefig(pltPath+num_features+'-bm_orig_measured-NEWAPP.png',
                transparent=True, bbox_inches='tight',dpi=600)
    # plt.show()

    print('avg length:', len(avg_pred_pwr), avg_pred_pwr)
    print('Error:', len(err), err)
    print('### NEW APP ###')
    print('MAE = ', mae_list)
    print('MAPE = ', err)
    print('MSE = ', mse_list)
    print('RMSE = ', rmse_list)
    print('MedAE = ', medae_list)
    print('EVS = ', evs_list)
    print('R2S = ', r2s_list)

def crossarch_app_fit_predict(gpu_data, x_gpu, y_gpu, p100_data, data, x, y, x_p100, y_p100, x_p100_df, y_p100_df, x_v100_df, y_v100_df, apps_list, ls, ts, ms, num_features, size, pltPath):
    x_p100_train, x_p100_test, y_p100_train, y_p100_test = train_test_split(
        x_p100, y_p100, test_size=size)

    lm = XGBRegressor()
    lm.fit(x, y)
    # get importance
    importance = lm.feature_importances_
    Plt_Importance(data, importance, num_features,
                   'CROSSARCH', ls, ts, pltPath)

    avg_pred_pwr = []
    max_orig_pwr = []
    diff_pwr = []
    plt_apps = []
    err = []
    mse_list = []
    rmse_list = []
    mae_list = []
    evs_list = []
    r2s_list = []
    medae_list = []
    # colors = itertools.cycle(['b', 'c', 'y', 'm', 'r'])
    colors = iter(cm.rainbow(np.linspace(0, 1, 19)))

    markers = ['o', '+', '*', '.', 'x', '_', '|', 's', 'd',
               '^', 'v', '>', '<', 'p', 'h', 'X', '8', '1', '2']
    mrk = 0
    plt.figure(figsize=(12, 11)).add_subplot(111)  # figsize=(10, 10)
    plt.style.use('classic')
    y_p100_pred = lm.predict(x_p100)
    plt.plot([y_p100.min(), y_p100.max()], [y_p100_pred.min(),
                                            y_p100_pred.max()], '--', color='red', lw=2)

    for a in apps_list:
        y_pred = lm.predict(x_p100_df[a])
        p_app = plt.scatter(y_p100_df[a], y_pred, color=next(
            colors), linewidths=1, marker=markers[mrk],  edgecolor=None,  s=ms)
        mrk += 1
        plt_apps.append(p_app)
        avg_pred_pwr.append(np.mean(y_pred))
        max_orig_pwr.append(np.mean(y_p100_df[a]))
        diff_pwr.append(np.mean(y_pred)-np.mean(y_p100_df[a]))
        e = metrics.mean_absolute_percentage_error(y_p100_df[a], y_pred)
        err.append(e)
        mse_list.append(mse(y_p100_df[a], y_pred))
        rmse_list.append(rmse(y_p100_df[a], y_pred))
        medae_list.append(medae(y_p100_df[a], y_pred))
        mae_list.append(mae(y_p100_df[a], y_pred))
        evs_list.append(evs(y_p100_df[a], y_pred))
        r2s_list.append(r2s(y_p100_df[a], y_pred))

    plt.legend((plt_apps),
               (apps_list),
               scatterpoints=1,
               loc='upper left',
               ncol=5,
               fontsize=11)

    plt.xlabel("Measured Power (Watt)", weight='bold', fontsize=ls)
    plt.ylabel("Predicted Power (Watt)", weight='bold', fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.ylim(0,165)
    print('Mean Percentage Error:', np.mean(err))
    # plt.title("Predictions vs. actual power values in the test set")
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')

    plt.grid(True)
    plt.savefig(pltPath+num_features+'-bm-prediction-CROSSARCH.png',
                transparent=True, bbox_inches='tight')

    # Plot original vs predicted powers
    err = [i*100 for i in err]

    ax = plt.figure(figsize=(12, 9)).add_subplot(111)
    plt.style.use('classic')
    index = apps_list
    df = pd.DataFrame({'Measured Power (W)': max_orig_pwr,
                       'Predicted Power (W)': avg_pred_pwr}, index=index)
    df = df.sort_values('Measured Power (W)')
    df.plot(ax=ax, kind='bar', color='rbg', rot=70, legend=False)
    bars = ax.patches
    hatches = ''.join(h*len(df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='upper left', ncol=1) #bbox_to_anchor=(1, 1),
    plt.xlabel("Benchmarks", weight='bold', fontsize=ls)
    plt.ylabel('Power (W)', weight='bold', fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.grid(True)
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # chr(37)+
    plt.suptitle('MAE - Min: '+str(round(min(err), 2))+',   Mean: '+str(round(np.mean(np.array(err)), 2)) +
                 ',   Max: '+str(round(max(err), 2))+',   Std: '+str(round(np.std(np.array(err)), 2)), fontsize=14, weight='bold', y=0.93)
    plt.savefig(pltPath+num_features+'-bm_orig_measured-CROSSARCH.png',
                transparent=True, bbox_inches='tight')
    # plt.show()

    print('avg length:', len(avg_pred_pwr), avg_pred_pwr)
    print('Error:', len(err), err)
    print('### CROSS ARCH ###')
    print('MAE = ', mae_list)
    print('MAPE = ', err)
    print('MSE = ', mse_list)
    print('RMSE = ', rmse_list)
    print('MedAE = ', medae_list)
    print('EVS = ', evs_list)
    print('R2S = ', r2s_list)


def global_fit_predict(data, x_p100, y_p100, x_df, y_df, apps_list, ls, ts, ms, num_features, size, pltPath):

    # x_p100_train, x_p100_test, y_p100_train, y_p100_test = train_test_split(x_p100, y_p100, test_size=size)
    # data = p100_data.copy()
    train = data.sample(frac=0.5)
    test = data.drop(train.index)
    # print (train_df.shape[0])
    # print (train_df.head())
    # print (test_df.shape[0])
    # print (test_df.head())
    # import sys
    # sys.exit(0)

    x_train = train.iloc[:, 2:]
    y_train = train.iloc[:, 0]

    x_test = test.iloc[:, 2:]
    y_test = test.iloc[:, 0]

    lm = XGBRegressor()
    lm.fit(x_train, y_train)
    # get importance
    importance = lm.feature_importances_
    Plt_Importance(data, importance, num_features, 'GLOBAL', ls, ts, pltPath)

    x_test_df = {}
    y_test_df = {}

    for app in apps_list:
        df = test[test['app'] == app]
        print('App test size:', df.shape[0])
        x_test_df[app] = df.iloc[:, 2:]
        y_test_df[app] = df.iloc[:, 0]

    avg_pred_pwr = []
    max_orig_pwr = []
    diff_pwr = []
    plt_apps = []
    err = []
    mse_list = []
    rmse_list = []
    mae_list = []
    evs_list = []
    r2s_list = []
    medae_list = []

    # colors = itertools.cycle(['b', 'c', 'y', 'm', 'r'])
    colors = iter(cm.rainbow(np.linspace(0, 1, 19)))
    markers = ['o', '+', '*', '.', 'x', '_', '|', 's', 'd',
               '^', 'v', '>', '<', 'p', 'h', 'X', '8', '1', '2']
    mrk = 0
    plt.figure(figsize=(12, 11)).add_subplot(111)  # figsize=(10, 10)
    plt.style.use('classic')
    y_test_pred = lm.predict(x_test)
    plt.plot([y_test.min(), y_test.max()], [y_test_pred.min(),
                                            y_test_pred.max()], '--', color='red', lw=2)

    for a in apps_list:
        y_pred = lm.predict(x_test_df[a])
        p_app = plt.scatter(y_test_df[a], y_pred, color=next(
            colors), linewidths=1, marker=markers[mrk],  edgecolor=None,  s=ms)
        mrk += 1
        plt_apps.append(p_app)
        avg_pred_pwr.append(np.mean(y_pred))
        max_orig_pwr.append(np.mean(y_test_df[a]))
        diff_pwr.append(np.mean(y_pred)-np.mean(y_test_df[a]))
        e = metrics.mean_absolute_percentage_error(y_test_df[a], y_pred)
        err.append(e)
        mse_list.append(mse(y_test_df[a], y_pred))
        rmse_list.append(rmse(y_test_df[a], y_pred))
        medae_list.append(medae(y_test_df[a], y_pred))
        mae_list.append(mae(y_test_df[a], y_pred))
        evs_list.append(evs(y_test_df[a], y_pred))
        r2s_list.append(r2s(y_test_df[a], y_pred))

    plt.legend((plt_apps),
               (apps_list),
               scatterpoints=1,
               loc='upper left',
               ncol=5,
               fontsize=11)

    plt.xlabel("Measured Power (Watt)", weight='bold', fontsize=ls)
    plt.ylabel("Predicted Power (Watt)", weight='bold', fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.ylim(0,165)
    print('Mean Percentage Error:', np.mean(err))
    # plt.title("Predictions vs. actual power values in the test set")
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.grid(True)
    plt.savefig(pltPath+num_features+'-bm-prediction-GLOBAL.png',
                transparent=True, bbox_inches='tight')

    # Plot original vs predicted powers
    err = [i*100 for i in err]

    ax = plt.figure(figsize=(12, 9)).add_subplot(111)
    plt.style.use('classic')
    index = apps_list
    df = pd.DataFrame({'Measured Power (W)': max_orig_pwr,
                       'Predicted Power (W)': avg_pred_pwr}, index=index)
    df = df.sort_values('Measured Power (W)')
    df.plot(ax=ax, kind='bar', color='rbg', rot=70, legend=False)
    bars = ax.patches
    hatches = ''.join(h*len(df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='upper left', ncol=1)
    plt.xlabel("Benchmarks", weight='bold', fontsize=ls)
    plt.ylabel('Power (W)', weight='bold', fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.grid(True)

    # plt.tick_params(axis='both', which='major', labelsize=12)
    # chr(37)+
    plt.suptitle('MAE - Min: '+str(round(min(mae_list), 2))+',   Mean: '+str(round(np.mean(np.array(mae_list)), 2)) +
                 ',   Max: '+str(round(max(mae_list), 2))+',   Std: '+str(round(np.std(np.array(mae_list)), 2)), fontsize=14,weight='bold', y=0.93)
    
    plt.savefig(pltPath+num_features+'-bm_orig_measured-GLOBAL.png',
                transparent=True, bbox_inches='tight')
    # plt.show()

    print('avg length:', len(avg_pred_pwr), avg_pred_pwr)
    print('Error:', len(err), err)
    print('### CROSS APP ###')
    print('MAE = ', mae_list)
    print('MAPE = ', err)
    print('MSE = ', mse_list)
    print('RMSE = ', rmse_list)
    print('MedAE = ', medae_list)
    print('EVS = ', evs_list)
    print('R2S = ', r2s_list)


def app_fit_predict(p100_data, x_p100, y_p100, x_df, y_df, apps_list, ls, ts, ms, num_features, size, pltPath):

    lms = {}
    x_traindf = {}
    y_traindf = {}
    x_testdf = {}
    y_testdf = {}

    for app in apps_list:
        x_train, x_test, y_train, y_test = train_test_split(
            x_df[app], y_df[app], test_size=size)
        lm = XGBRegressor()
        lm.fit(x_train, y_train)
        lms[app] = lm
        x_traindf[app] = x_train
        x_testdf[app] = x_test
        y_traindf[app] = y_train
        y_testdf[app] = y_test

        # Plt_Importance(p100_data,importance,num_features,app,ls,ts)

    avg_pred_pwr = []
    max_orig_pwr = []
    diff_pwr = []
    plt_apps = []

    err = []
    mse_list = []
    rmse_list = []
    mae_list = []
    evs_list = []
    r2s_list = []
    medae_list = []
    # colors = itertools.cycle(['b', 'c', 'y', 'm', 'r'])
    colors = iter(cm.rainbow(np.linspace(0, 1, 19)))
    markers = ['o', '+', '*', '.', 'x', '_', '|', 's', 'd',
               '^', 'v', '>', '<', 'p', 'h', 'X', '8', '1', '2']

    plt.figure(figsize=(12, 11)).add_subplot(111)  # figsize=(10, 10)
    plt.style.use('classic')
    # y_p100_test_pred = lm.predict(x_p100_test)
    # plt.plot([y_p100_test.min(), y_p100_test.max()], [y_p100_test_pred.min(), y_p100_test_pred.max()], '--',color = 'red', lw=2)
    mrk = 0
    for a in apps_list:
        y_pred = lms[a].predict(x_testdf[a])
        print(y_pred.shape)
        p_app = plt.scatter(y_testdf[a], y_pred, color=next(
            colors), linewidths=1, marker=markers[mrk],  edgecolor=None,  s=ms)
        mrk += 1
        plt_apps.append(p_app)
        avg_pred_pwr.append(np.mean(y_pred))
        max_orig_pwr.append(np.mean(y_testdf[a]))
        diff_pwr.append(np.mean(y_pred)-np.mean(y_testdf[a]))

        e = mape(y_testdf[a], y_pred)
        err.append(e)
        mse_list.append(mse(y_testdf[a], y_pred))
        rmse_list.append(rmse(y_testdf[a], y_pred))
        medae_list.append(medae(y_testdf[a], y_pred))
        mae_list.append(mae(y_testdf[a], y_pred))
        evs_list.append(evs(y_testdf[a], y_pred))
        r2s_list.append(r2s(y_testdf[a], y_pred))

    plt.legend((plt_apps),
               (apps_list),
               scatterpoints=1,
               loc='upper left',
               ncol=5,
               fontsize=11)

    plt.xlabel("Measured Power (Watt)", weight='bold', fontsize=ls)
    plt.ylabel("Predicted Power (Watt)", weight='bold', fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    # plt.ylim(0,165)
    print('Mean Percentage Error:', np.mean(err))
    # plt.title("Predictions vs. actual power values in the test set")
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.grid(True)
    plt.savefig(pltPath+num_features+'-bm-prediction-APP.png',
                transparent=True, bbox_inches='tight')

    # Plot original vs predicted powers
    err = [i*100 for i in err]

    ax = plt.figure(figsize=(12, 9)).add_subplot(111)
    plt.style.use('classic')
    index = apps_list
    df = pd.DataFrame({'Measured Power (W)': max_orig_pwr,
                       'Predicted Power (W)': avg_pred_pwr}, index=index)
    
    # df = df.sort_index(axis=0, ascending=False, kind='quicksort')
    df = df.sort_values('Measured Power (W)')
    df.plot(ax=ax, kind='bar', color='rbg', rot=70, legend=False)
    # plt.gca().invert_xaxis()
    bars = ax.patches
    hatches = ''.join(h*len(df) for h in 'x/O.')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.legend(loc='upper left',  ncol=1)
    plt.xlabel("Benchmarks", weight='bold', fontsize=ls)
    plt.ylabel('Power (W)', weight='bold', fontsize=ls)
    plt.tick_params(axis='both', which='major', labelsize=ts)
    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')
    plt.grid(True)
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # chr(37)+
    plt.suptitle('MAE - Min: '+str(round(min(mae_list), 2))+',   Mean: '+str(round(np.mean(np.array(mae_list)), 2)) +',   Max: '+str(round(max(mae_list), 2))+',   Std: '+str(round(np.std(np.array(mae_list)), 2)), fontsize=14,weight='bold', y=0.93)
    plt.savefig(pltPath+num_features+'-bm_orig_measured-APP.png',
                transparent=True, bbox_inches='tight')
    # plt.show()

    print('avg length:', len(avg_pred_pwr), avg_pred_pwr)
    print('### APP ###')
    print('MAE = ', mae_list)
    print('MAPE = ', err)
    print('MSE = ', mse_list)
    print('RMSE = ', rmse_list)
    print('MedAE = ', medae_list)
    print('EVS = ', evs_list)
    print('R2S = ', r2s_list)


def mae(y_test, y_test_pred):
    return round(metrics.mean_absolute_error(y_test, y_test_pred), 2)


def mape(y_test, y_test_pred):
    return round(metrics.mean_absolute_percentage_error(y_test, y_test_pred), 2)


def mse(y_test, y_test_pred):
    return round(metrics.mean_squared_error(y_test, y_test_pred), 2)


def rmse(y_test, y_test_pred):
    return round(np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)), 2)


def medae(y_test, y_test_pred):
    return round(metrics.median_absolute_error(y_test, y_test_pred), 2)


def evs(y_test, y_test_pred):
    return round(metrics.explained_variance_score(y_test, y_test_pred), 2)


def r2s(y_test, y_test_pred):
    return round(metrics.r2_score(y_test, y_test_pred), 2)
