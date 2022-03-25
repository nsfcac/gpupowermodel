#! /usr/bin/env python
# from numpy.core.fromnumeric import mean
# from numpy.linalg import norm
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

# from xgboost import XGBRegressor
# from sklearn.svm import SVR
import numpy as np
import pandas as pd
# from sklearn import metrics
import matplotlib
import sys

xss = 6
ss  = 8
ms  = 10
ls  = 12
xls = 14
xxls = 16

matplotlib.rc('xtick', labelsize=ss) 
matplotlib.rc('ytick', labelsize=ss) 

figureExtension = '.pdf'
# figureExtension = '.png'
# figureExtension = '.svg'

dvfs_configs=[1380, 1372, 1365, 1357, 1350, 1342, 1335, 1327, 1320, 1312, 1305, 1297, 1290, 1282, 1275, 1267, 1260, 1252, 1245, 1237, 1230, 1222, 1215, 1207, 1200, 1192, 1185, 1177, 1170, 1162, 1155, 1147, 1140, 1132, 1125, 1117, 1110, 1102, 1095, 1087, 1080, 1072, 1065, 1057, 1050, 1042, 1035, 1027, 1020, 1012, 1005, 997, 990, 982, 975, 967, 960, 952, 945, 937, 930, 922, 915, 907, 900, 892, 885, 877, 870, 862, 855, 847, 840, 832, 825, 817, 810, 802, 795, 787, 780, 772, 765, 757, 750, 742, 735, 727, 720, 712, 705, 697, 690, 682, 675, 667, 660, 652, 645, 637, 630, 622, 615, 607, 600, 592, 585, 577, 570, 562, 555, 547, 540, 532, 525, 517, 510]

def metric_avg(app, timeDataFile, n_run):
    print ('\n***** COMPUTING AVG OF '+str(n_run)+' runs for'+str(app)+' *****\n')
    fp64 = []
    fp32 = []
    dram = []
    for i in range (n_run):
        f1=str(timeDataFile)
        # f1 = 'results/GV100-dvfs-'+app+'-'+str(freq_ran)+'-'+str(n_run) # FOR DGEMM FILES
        # reading data
        df1 = pd.read_csv(f1,delim_whitespace=True,error_bad_lines=False) #dgemm_minf_1ms
        # basic data cleaning
        del df1['Entity']
        del df1['#']
        df1.drop(df1[df1['POWER'] == 'POWER'].index, inplace = True)
        df1 = df1.dropna(axis=0) #how='all'
        df1["FP32A"] = pd.to_numeric(df1["FP32A"], downcast="float")
        df1["FP64A"] = pd.to_numeric(df1["FP64A"], downcast="float")
        df1["DRAMA"] = pd.to_numeric(df1["DRAMA"], downcast="float")
        d64 = df1.loc[df1['FP64A'] > 0]
        d32 = df1.loc[df1['FP32A'] > 0]
        ddram = df1.loc[df1['DRAMA'] > 0]
        fp64.append(d64['FP64A'].mean())
        fp32.append(d32['FP32A'].mean())
        dram.append(ddram['DRAMA'].mean())
    # print ('FP64A 3 Runs: ',fp64)
    # print ('FP32A 3 Runs: ',fp32)
    # print ('DRAMA 3 Runs: ',dram)
    fp64_avg = np.mean(fp64)
    fp32_avg = np.mean(fp32)

    if (not np.isnan(fp32_avg).any()) & (not np.isnan(fp64_avg).any()):  
        fp_avg = fp32_avg/2 + fp64_avg
    elif np.isnan(fp64_avg).any():
        fp_avg = fp32_avg/2
    elif np.isnan(fp32_avg).any():
        fp_avg = fp64_avg
    
    dram_avg = np.mean(dram)
    return fp_avg, dram_avg

def predict_power(fp,dram):
    df = pd.DataFrame(dvfs_configs,columns=['sm_app_clock'])
    df['n_sm_app_clock'] = np.log1p(df['sm_app_clock'])
    df['predicted_n_power_usage'] =  -1.0318354343254663 + 0.84864*fp + 0.09749*dram + df["n_sm_app_clock"].mul(0.77006)

    return df

def predict_runtime(app_df,T,fp):
    # dgemm and stream
    T_fmax = np.log1p(T)
    p=[1.43847511, -0.16736726, -0.90400864, 0.48241361, 0.78898516]
    # B0=-0.5395982891830665
    B0=0
    app_df['predicted_n_run_time'] = T_fmax + B0 + p[0]*fp + p[1]*(7.230563 - app_df['n_sm_app_clock']) + p[2]*(fp**2) + p[3]*fp*(7.230563 - app_df['n_sm_app_clock']) + p[4]*((7.230563 - app_df['n_sm_app_clock']).pow(2))

    # predicted n to r power
    app_df['predicted_n_to_r_power_usage'] = np.expm1(app_df['predicted_n_power_usage'])
    # predicted n to r run time
    app_df['predicted_n_to_r_run_time'] = np.expm1(app_df['predicted_n_run_time'])
    # predicted n to r energy
    app_df['predicted_n_to_r_energy'] = app_df['predicted_n_to_r_run_time'].mul(app_df['predicted_n_to_r_power_usage'])
    app_df['predicted_n_energy'] = np.log1p(app_df['predicted_n_to_r_energy'])
    app_df.reset_index(inplace=True)
    app_df.to_csv('results/model_prediction_'+app+'.csv')

    return app_df

def get_def_runtime(app, profiledDataFile):
    f = str(profiledDataFile)
    # f = 'results/GV100-dvfs-'+app+'-perf.csv'  # FOR DGEMM FILES
    # reading data
    df = pd.read_csv(f) 
    df.columns = ['sm_app_clock','perf','runtime']
    runtime_df = df['runtime'].mean()
    if app == 'dgemm':
        runtime_df = runtime_df/1000
    return round(runtime_df,1)

# ============================================================================

def EDP_Optimal(df,ecol,tcol):
    wt = 0.5
    we = 0.5

    df['score'] =  (wt*df[tcol]) * (we*df[ecol])

    optimum_sol = df.loc[df['score'] == min(df['score'])]
    f = int(optimum_sol.iloc[0]['sm_app_clock'])
    x = round(optimum_sol.iloc[0]['predicted_n_to_r_run_time'],2)
    y = round(optimum_sol.iloc[0]['predicted_n_to_r_power_usage'],2)
    z = round(optimum_sol.iloc[0][ecol],2)
    return f,x,y,z

# ============================================================================

def ED2P_Optimal(df,ecol,tcol):
    wt = 0.5
    we = 0.5

    df['score'] =  (wt*(df[tcol]**2)) * (we*df[ecol])

    optimum_sol = df.loc[df['score'] == min(df['score'])]
    f = int(optimum_sol.iloc[0]['sm_app_clock'])
    x = round(optimum_sol.iloc[0]['predicted_n_to_r_run_time'],2)
    y = round(optimum_sol.iloc[0]['predicted_n_to_r_power_usage'],2)
    z = round(optimum_sol.iloc[0][ecol],2)
    return f,x,y,z

# ============================================================================

def optimal_dvfs(df, app, p_ecol, p_tcol, appType):
    
    print ("\n********** STEP 3: Determining Optimal DVFS **********\n")

    s_size = 40
    sub_plts = []
    lbl = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
           'q','r','s','t','u','v','w']

    #fig = plt.figure(figsize=(15, 9)) #7.5, 2  , sharey=True 
#     fig = plt.figure(figsize=(9, 3)) #7.5, 2  , sharey=True 
    plt.style.use('classic')
    sns.set_context('paper', font_scale=1)
    
    ax = plt.figure(figsize=(3.5,3)).add_subplot()

    edp_freq,x,y,z = EDP_Optimal(df,p_ecol,p_tcol)

    g = sns.scatterplot(ax=ax, data=df, x='predicted_n_to_r_run_time', y='predicted_n_to_r_power_usage', s=s_size, 
                        color='blue', marker='d')
    
    myStr = ''
    myStr = app
    if myStr == 'LSTM':
        myStr = 'TensorFlow'

    ax.set_xlabel('Run Time (s)', weight='bold',fontsize=xls)
    ax.set_ylabel('Power (W)', weight='bold',fontsize=xls)
    
    # ax.set_xlim([475, 1420])
    # Reference the optimal energy profile
    label = 'Predicted EDP DVFS: '+str(edp_freq)+'MHz'
    print (label)
    ax.annotate(
        # text='EDP:'+str(freq)+'MHz,'+str(x)+'S',
        text=label,
        xy=(x, y),
        xycoords='data',
        fontsize=ls,#11
        # xytext=(df[m_tcol].min(), df['power_usage'].min()),
        xytext=(df['predicted_n_to_r_run_time'].min(), df['predicted_n_to_r_power_usage'].max()-10),
        #textcoords='offset points',
        bbox=dict(boxstyle='square,pad=0.3', fc='snow', alpha=0.75),
        arrowprops=dict(arrowstyle='-|>', color='red'),  # Use color black
        # horizontalalignment='left',  # Center horizontally
        # verticalalignment='left'
        fontweight='bold'
        )  # Center vertically

    ed2p_freq, x, y, z = ED2P_Optimal(df,p_ecol,p_tcol)
    
    label = 'Predicted ED\u00b2P DVFS: '+str(ed2p_freq)+'MHz'
    print (label)

    plt.tick_params(axis='both', which='major', labelsize=ss, 
            direction = 'in', reset=True, color = 'black')
    
    plt.grid(False)
    plt.tight_layout()
    plt.subplots_adjust(hspace = .25, wspace=.15)

    plt.savefig('figures/' + appType + '_' + app + '_edp_perf_pwr' + figureExtension, 
                transparent=True, bbox_inches = 'tight', pad_inches = 0.1, dpi=400)
    plt.close()
    return edp_freq, ed2p_freq

def storing_optimal_freq(app, edp_freq, ed2p_freq,freq_database):
    print ("\n********** STEP 4: Storing the optimal frequency **********\n")
    # df = pd.read_csv('database.csv')
    df = pd.read_csv(str(freq_database))
    record = {'job':app,'gpu_optimal_freq':ed2p_freq}
    df = df.append(record, ignore_index = True)
    df = df[['job','gpu_optimal_freq']]
    print(df)
    df.to_csv(str(freq_database))
    # df.to_csv('database.csv')

# ============================================================================

if __name__ == '__main__':
    # print(" *** Start of of __main__ for model.py! ***")
    # df = pd.read_csv('database.csv')
    # appName = "ERROR"

    appType = 'hpc_3f'
    p_ecol = 'predicted_n_energy'
    p_tcol = 'predicted_n_run_time'

    if len(sys.argv) <= 5:
        print("Please provide arguments: ")
        print("  [1] application name")
        print("  [2] timeDataFile")
        print("  [3] profiledDataFile")
        print("  [4] frequency")
        print("  [5] number of runs")
        print("  [6] database file")
        opt_freq = 0
    else:
        appName = sys.argv[1]           # application name
        timeDataFile = sys.argv[2]      # e.g. results/GV100-dvfs-DEMO-perf.csv
        profiledDataFile = sys.argv[3]  # e.g. results/GV100-dvfs-DEMO-1234-0
        freq_ran = sys.argv[4]          # frequency monitoring happened
        n_run = sys.argv[5]             # number of runs
        freq_database = sys.argv[6]     # database file

        #   ---   ---   ---   ---   ---   ---   ---   ---

        fp, dram = metric_avg(appName,timeDataFile, n_run)
        df = predict_power(fp, dram)
        def_runtime = get_def_runtime(appName,profiledDataFile)
        df = predict_runtime(df, def_runtime,fp)
        
        #   ---   ---   ---   ---   ---   ---   ---   ---

        edp_freq, ed2p_freq = optimal_dvfs(df, appName, p_ecol, p_tcol, appType)

        storing_optimal_freq(appName, edp_freq, ed2p_freq, freq_database)

        # TODO: Assign opt_freq
        # opt_freq = 
    #   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---
        
    #print("Optimal Frequecy:", str(freq))
    print(str(opt_freq))

    sys.exit(0)
    # print(" *** End of __main__ for model.py! ***")

# ============================================================================
