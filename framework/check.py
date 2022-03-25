#! /usr/bin/env python
import numpy as np
import pandas as pd
# from sklearn import metrics
import sys

# dict = {
#             'job': ['X'],
#             'gpu_optimal_freq': [1]
#         pip  }
# df = pd.DataFrame(dict)
# print(df)
# df.to_csv('jobs_optimal_freqs.csv', encoding='utf-8')

if __name__ == '__main__':
    # print(" *** Start of of __main__ for check.py! ***")
    df = pd.read_csv('database.csv')
    # appName = "ERROR"

    if len(sys.argv) <= 1:
        # print("Missing argument!")
        freq = 0
    else:
        appName = sys.argv[1]
        #print("App Name is:", str(appName))
        freq = df.loc[df['app_name']==appName]['gpu_opt_freq'].values
        if freq.size == 0:
            # print("Optimal frequency unknown!")
            freq = 0
        else:
            # print("Found optimal frequency.")
            freq = freq[0]
    
    #print("Optimal Frequecy:", str(freq))
    print(str(freq))

    # print(" *** End of __main__ for check.py! ***") 
    sys.exit(0)
