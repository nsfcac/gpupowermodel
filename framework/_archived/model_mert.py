from numpy.core.fromnumeric import mean
from numpy.linalg import norm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# from xgboost import XGBRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn import metrics
import sys

# ============================================================================
def EDP_Optimal(df,ecol,tcol):
    wt = 0.5
    we = 0.5

    df['score'] =  (wt*df[tcol]) * (we*df[ecol])

    optimum_sol = df.loc[df['score'] == min(df['score'])]
    f = int(optimum_sol.iloc[0]['sm_app_clock'])
    x = round(optimum_sol.iloc[0][tcol],2)
    y = round(optimum_sol.iloc[0]['power_usage'],2)
    z = round(optimum_sol.iloc[0][ecol],2)
    return f,x,y,z

# ============================================================================
def ED2P_Optimal(df,ecol,tcol):
    wt = 0.5
    we = 0.5

    df['score'] =  (wt*(df[tcol]**2)) * (we*df[ecol])

    optimum_sol = df.loc[df['score'] == min(df['score'])]
    f = int(optimum_sol.iloc[0]['sm_app_clock'])
    x = round(optimum_sol.iloc[0][tcol],2)
    y = round(optimum_sol.iloc[0]['power_usage'],2)
    z = round(optimum_sol.iloc[0][ecol],2)
    return f,x,y,z

# ============================================================================

def estimate_pwr_runtime(apps,f,m,features, normalized):
    
    df = pd.read_csv(f)
    '''
    correlation(df['power_usage'].subtract(28),df['fp32_active'])
    correlation(df['power_usage'].subtract(28),df['fp64_active'])
    correlation(df['power_usage'].subtract(28),df['dram_active'])
    correlation(df['power_usage'].subtract(28),df['sm_app_clock'])
    '''
    
    if m == 'power_usage':
        # print ('###REAL###',df["power_usage"])
        if features == 'F3':
            if normalized == True:
                # df['n_power_usage'] = min_max_norm(df['power_usage'])
                # df['n_fp32_active'] = min_max_norm(df['fp32_active'])
                # df['n_fp64_active'] = min_max_norm(df['fp64_active'])
                # df['n_dram_active'] = min_max_norm(df['dram_active'])
                # df['n_sm_app_clock'] = min_max_norm(df['sm_app_clock'])
                df['n_power_usage'] = np.log1p(df['power_usage'])
                # df['n_fp32_active'] = np.log1p(df['fp32_active'])
                # df['n_fp64_active'] = np.log1p(df['fp64_active'])
                # df['n_fp_active'] = np.log1p(df['fp_active'])
                # df['n_dram_active'] = np.log1p(df['dram_active'])
                df['n_sm_app_clock'] = np.log1p(df['sm_app_clock'])

                # data = df[["n_power_usage", "application","n_fp32_active","n_fp64_active", "n_dram_active", "n_sm_app_clock"]]
                # data = df[["n_power_usage", "application","n_fp_active", "n_dram_active", "n_sm_app_clock"]]
            elif normalized == False:
                data = df[["power_usage", "application","fp32_active","fp64_active", "dram_active", "sm_app_clock"]] #
        elif features == 'F2':
            if normalized == True:
                df['n_power_usage'] = np.log1p(df.power_usage)
                data = df[["n_power_usage", "application","fp_active", "sm_app_clock"]]
            elif normalized == False:
                data = df[["power_usage", "application","fp_active", "sm_app_clock"]]    
    # elif m == 'run_time':
    #     # print ('###REAL###',df["run_time"])
    #     if features == 'F3':
    #         if normalized == True:
    #             df['n_run_time'] = np.log1p(df.run_time)
    #             df['n_fp_active'] = np.log1p(df['fp_active'])
    #             df['n_sm_app_clock'] = np.log1p(df['sm_app_clock'])
    #             data = df[["n_run_time", "application","n_fp_active","n_sm_app_clock"]]
    #         elif normalized == False:
    #             data = df[["run_time", "application","fp_active","dram_active","sm_app_clock"]]
    #     elif features == 'F2':
    #         if normalized == True:
    #             df['n_run_time'] = np.log1p(df.run_time)
    #             data = df[["n_run_time", "application","fp_active","sm_app_clock"]]
    #         elif normalized == False:
    #             data = df[["run_time", "application","fp_active","sm_app_clock"]]

    data = df.copy()
    
    if normalized == True:
        if m == 'power_usage':
            # '''
            for app in apps:
                print (app,end=": ")
                app_df = data.loc[data['application'] == app]
                app_df = app_df.copy()
                
                P_fmax = app_df.loc[app_df['sm_app_clock'] == 1380]['n_power_usage'].values[0]
                dram = app_df.loc[app_df['sm_app_clock'] == 1380]['dram_active'].values[0]
                fp_32 = app_df.loc[app_df['sm_app_clock'] == 1380]['fp32_active'].values[0]
                fp_64 = app_df.loc[app_df['sm_app_clock'] == 1380]['fp64_active'].values[0]
                fp = app_df.loc[app_df['sm_app_clock'] == 1380]['fp_active'].values[0]
                # print ('P_fmax',P_fmax)
                # app_df['predicted_n_power_usage'] = P_fmax  - 0.69514 * (7.230563 - app_df['n_sm_app_clock'])
                # app_df['predicted_n_power_usage'] = fp_32 * 1.24535 + fp_64 * 0.96885 + dram_a * 0.95227 + data["n_sm_app_clock"].mul(0.59899)
                # TODO: ghali-mside
                app_df['predicted_n_power_usage'] =  -1.0318354343254663 + 0.84864*fp + 0.09749*dram + app_df["n_sm_app_clock"].mul(0.77006) 

                app_df['n_energy'] = np.log1p(app_df['energy'])
                app_df.reset_index(inplace=True)
                app_df.to_csv('data/am_pwr_'+app+'.csv')
                # print (app_df)
                errors = abs(app_df["n_power_usage"].subtract(app_df['predicted_n_power_usage']))
                mape = round(100 * np.mean(errors / app_df["n_power_usage"]),1)   
                # print ('MAPE:',mape)
                accuracy = 100 - mape
                # print('***Model Performance***')
                # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
                print('Accuracy = {:0.1f}%.'.format(accuracy))

            # '''
            '''
            data['predicted_n_power_usage'] = data["n_fp32_active"].mul(1.24535) + data["n_fp64_active"].mul(0.96885) + data["n_dram_active"].mul(0.95227) + data["n_sm_app_clock"].mul(0.59899)
            data.reset_index(inplace=True,drop=True)
            # print (data)
            data.to_csv('P_hpc_apps.csv')
            errors = abs(data["n_power_usage"].subtract(data['predicted_n_power_usage']))
            mape = round(100 * np.mean(errors / data["n_power_usage"]),1)   
            print ('MAPE:',mape)
            accuracy = 100 - mape
            print('***Model Performance***')
            print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
            print('Accuracy = {:0.2f}%.'.format(accuracy))
            '''
        elif m == 'run_time':
            data['n_run_time'] = np.log1p(data.run_time)
            for app in apps:
                app_df = data.loc[data['application'] == app]
                app_df = app_df.copy()
                app_df.reset_index(inplace=True,drop=True)
                T_fmax = app_df.loc[app_df['sm_app_clock'] == 1380]['n_run_time'].values[0]
                fp = app_df.loc[df['sm_app_clock'] == 1380]['fp_active'].values[0]

                # df['estimated_run_time'] = T_fmax  + fp*0.79523 * (7.230563 - df['n_sm_app_clock'])
                # app_df['predicted_n_run_time'] = T_fmax  + fp*0.79523 * (7.230563 - app_df['n_sm_app_clock'])
                # app_df['predicted_n_run_time'] = T_fmax  + 1.11274 * (7.230563 - app_df['n_sm_app_clock'])
                # app_df['predicted_n_run_time'] = T_fmax  + 0.47994 * (7.230563 - app_df['n_sm_app_clock'])
                # app_df['predicted_n_run_timepredicted_n_run_time'] = T_fmax  + fp*0.47994 * (7.230563 - app_df['n_sm_app_clock'])
                
                 # dgemm and stream
                p=[1.43847511, -0.16736726, -0.90400864, 0.48241361, 0.78898516]
                # B0=-0.5395982891830665
                B0=0
                # TODO: ghali-mside
                app_df['predicted_n_run_time'] = T_fmax + B0 + p[0]*fp + p[1]*(7.230563 - app_df['n_sm_app_clock']) + p[2]*(fp**2) + p[3]*fp*(7.230563 - app_df['n_sm_app_clock']) + p[4]*((7.230563 - app_df['n_sm_app_clock']).pow(2))

                '''
                p = [-0.42927755, 1.57807214, -0.72607066]
                # B0 = 1.466857137666937
                app_df['predicted_n_run_time'] = T_fmax + p[0]*(7.230563 -  app_df['n_sm_app_clock']) + p[1]*((7.230563 -  app_df['n_sm_app_clock']).pow(2)) + p[2]*((7.230563 -  app_df['n_sm_app_clock']).pow(3))
                '''
                # predicted n to r power
                app_df['predicted_n_to_r_power_usage'] = np.expm1(app_df['predicted_n_power_usage'])
                # predicted n to r run time
                app_df['predicted_n_to_r_run_time'] = np.expm1(app_df['predicted_n_run_time'])
                # predicted n to r energy
                app_df['predicted_n_to_r_energy'] = app_df['predicted_n_to_r_run_time'].mul(app_df['predicted_n_to_r_power_usage'])
                app_df['predicted_n_energy'] = np.log1p(app_df['predicted_n_to_r_energy'])
                app_df.reset_index(inplace=True)
                app_df.to_csv('data/am_rt_'+app+'.csv')
                # print (app_df)
                print (app,end=": ")
                errors = abs(app_df["n_run_time"].subtract(app_df['predicted_n_run_time']))
                mape = round(100 * np.mean(errors / app_df["n_run_time"]),1)   
                # print ('MAPE:',mape)
                accuracy = 100 - mape
                # print('***Model Performance***')
                # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
                print('Accuracy = {:0.1f}%.'.format(accuracy))
    
    elif normalized == False:
        data['estimated_power_usage'] = data["fp32_active"].mul(imp[0]) + data["fp64_active"].mul(imp[1]) + data["dram_active"].mul(imp[2]) + data["sm_app_clock"].mul(imp[3])
    
        for i, v in enumerate(imp):
            print('Feature: %0d, Score: %.5f' % (i, v))

        # data["power_usage"] = data["power_usage"].add(28)
        # data["estimated_power_usage"] = data["estimated_power_usage"].add(28)
        errors = abs(data["power_usage"].subtract(data['estimated_power_usage']))
        mape = round(100 * np.mean(errors / data["power_usage"]),1)   
        print (data)
        print ('MAPE:',mape)
        accuracy = 100 - mape
        print('***Model Performance***')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))

if __name__ == '__main__':
    # print(" *** Start of of __main__ for model.py! ***")
    # df = pd.read_csv('database.csv')
    # appName = "ERROR"

    if len(sys.argv) <= 2:
        # print("Missing arguments!")
        print("Please provide arguments: ")
        print("  [1] application script")
        print("  [2] application name")
        print("  [3] number of runs (optional)")
        print("  [4] core frequency (optional)")
        print("  [5] memory frequency (optional)")
        opt_freq = 0
    else:
        appName = sys.argv[1]
        timeDataFile = sys.argv[2]
        profiledDataFile = sys.argv[3]
        #estimate_pwr_runtime(apps,f,m,features, normalized):
        estimate_pwr_runtime(appName, profiledDataFile, 'power_usage', 'F3', True)
        # TODO: Assign opt_freq
        # opt_freq = 
        
    #print("Optimal Frequecy:", str(freq))
    print(str(opt_freq))

    sys.exit(0)
    # print(" *** End of __main__ for model.py! ***")
