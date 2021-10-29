from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

def random_best_params(train_features, train_labels):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    # print('random_grid params:',random_grid)
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(train_features, train_labels)

    print ('rf_random.best_params_:',rf_random.best_params_)
    return rf_random

def model_tuning(f,mode):

    df = pd.read_csv(f)
    print ('***MODEL***',mode)
    if mode == 'power_usage':
        # data = df[["power_usage", "application","fp_active", "dram_active","sm_app_clock"]]
        data = df[["power_usage", "application","fp_active","sm_app_clock"]]
    else:
        # data = df[["run_time", "application","fp_active", "dram_active","sm_app_clock"]]
        data = df[["run_time", "application","fp_active","sm_app_clock"]]
    
    data = data.copy()
    data.reset_index(drop=True, inplace=True)
    x = data.iloc[:, 2:].values
    y = data.iloc[:, 0].values
    from sklearn.model_selection import train_test_split
    train_features, test_features, train_labels, test_labels = train_test_split(
            x, y, random_state=42)

    base_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    base_model.fit(train_features, train_labels)
    base_accuracy = evaluate(base_model, test_features, test_labels)

    rf_random = random_best_params(train_features, train_labels)
    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, test_features, test_labels)
    print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

def combine_pwr_rt():
    df_pwr = pd.read_csv('2f-predicted_power_usage_HPC_apps.csv')
    df_rt = pd.read_csv('2f-predicted_run_time_HPC_apps.csv')
    df_rt['energy'] = df_rt['run_time'] * df_pwr['power_usage']
    df_tmp = df_rt[['predicted_run_time','energy']]
    df_tmp = df_tmp.copy()
    print (df_tmp)
    print (df_pwr)
    df = df_pwr.join(df_tmp)
    df['predicted_energy'] = df['predicted_power_usage'] * df['predicted_run_time']
    print (df)
    df.to_csv('2f-predicted_power_runtime_energy_HPC_apps.csv')


'''
#*** MODEL TUNING ***

# mode = 'power_usage'
mode = 'run_time'
# f = '2f-fill_mean_fp_data_21_apps.csv'
f = 'mean_fp_data_HPC_apps.csv'
model_tuning(f,mode)

## 2-features POWER MODEL:
# 'n_estimators': 1400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 100, 'bootstrap': True
## 2-features PERF MODEL:
# 'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 100, 'bootstrap': True}
#*** MODEL TUNING***
'''

'''
#*** MODEL PREDICTION, OF POWER, RUNTIME AND ENERGY ***
# f = 'mean_fp_data_HPC_apps.csv'
# mode = 'power_usage'
# mode = 'run_time'

# model = 'RFR'
# model = 'SVR'
# model = 'XGBR'

# power_runtime_prediction(f,mode,model,1400) #run time = 
# combine_pwr_rt()
#*** MODEL TUNING, PREDICTION, OF POWER, RUNTIE AND ENERGY ***
'''