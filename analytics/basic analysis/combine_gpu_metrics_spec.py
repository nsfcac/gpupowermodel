import os
import glob
import pandas as pd
import numpy as np
from typing_extensions import runtime
from statistics import mean
import sys
#path = '/mnt/c/rf/lbnl/data/SPEC/P100/dvfs/dvfs_profile/result/'
p100_spec_runtime_file_path = 'C:/rf/lbnl/data/SPEC/P100/dvfs/dvfs_spec/SPEC-DVFS-RUNTIME_P100.csv'
v100_spec_runtime_file_path = 'C:/rf/lbnl/data/SPEC/V100/dvfs/dvfs_spec/SPEC-DVFS-RUNTIME_V100.csv'
# os.chdir('/mnt/c/rf/lbnl/data/phase2/p100-1/results') #/mnt/c/rf/cs
# path = '/mnt/c/rf/lbnl/data/phase2/p100-1/results/'
# os.chdir('/mnt/c/rf/lbnl/data/phase2/v100-1/results') #/mnt/c/rf/cs
# path = '/mnt/c/rf/lbnl/data/phase2/v100-1/results/'

# os.chdir('/home/ghali/v100-1/results') #/mnt/c/rf/cs
# path = '/home/ghali/v100-1/results/'

p100_freqs = [544, 556, 569, 582, 594, 607, 620, 632, 645, 658, 670, 683, 696, 708, 721, 734, 746, 759, 772, 784, 797, 810, 822, 835, 847, 860, 873, 885, 898, 911, 923, 936, 949, 961, 974, 987, 999, 1012, 1025, 1037, 1050, 1063, 1075, 1088, 1101, 1113, 1126, 1139, 1151, 1164, 1177, 1189, 1202, 1215, 1227, 1240, 1252, 1265, 1278, 1290, 1303, 1316, 1328]
v100_freqs = [135, 142, 150, 157, 165, 172, 180, 187, 195, 202, 210, 217, 225, 232, 240, 247, 255, 262, 270, 277, 285, 292, 300, 307, 315, 322, 330, 337, 345, 352, 360, 367, 375, 382, 390, 397, 405, 412, 420, 427, 435, 442, 450, 457, 465, 472, 480, 487, 495, 502, 510, 517, 525, 532, 540, 547, 555, 562, 570, 577, 585, 592, 600, 607, 615, 622, 630, 637, 645, 652, 660, 667, 675, 682, 690, 697, 705, 712, 720, 727, 735, 742, 750, 757, 765, 772, 780, 787, 795, 802, 810, 817, 825, 832, 840, 847, 855, 862, 870, 877, 885, 892, 900, 907, 915, 922, 930, 937, 945, 952, 960, 967, 975, 982, 990, 997, 1005, 1012, 1020, 1027, 1035, 1042, 1050, 1057, 1065, 1072, 1080, 1087, 1095, 1102, 1110, 1117, 1125, 1132, 1140, 1147, 1155, 1162, 1170, 1177, 1185, 1192, 1200, 1207, 1215, 1222, 1230, 1237, 1245, 1252, 1260, 1267, 1275, 1282, 1290, 1297, 1305, 1312, 1320, 1327, 1335, 1342, 1350, 1357, 1365, 1372, 1380]

# # index P100 frequencies:
# p100_freq_dic = {}
# i=0
# for f in p100_freqs:
#     p100_freq_dic[str(f)] = i
#     i += 1
# app ID mapping
app_id_dic = {"tpacf":'101',"stencil":'103' ,"lbm":'104' ,"fft":'110' ,"spmv":'112' ,"mriq":'114' ,"histo":'116' ,"bfs":'117' ,"cutcp":'118' ,"kmeans":'120' ,"lavamd":'121' ,"cfd":'122' ,"nw":'123' ,"hotspot":'124' ,"lud":'125' ,"ge":'126' ,"srad":'127' ,"heartwall":'128' ,"bplustree":'140'}
app_type_dic = {"tpacf":1,"stencil":1 ,"lbm":1 ,"fft":1 ,"spmv":1 ,"mriq":1 ,"histo":0 ,"bfs":1 ,"cutcp":1 ,"kmeans":0 ,"lavamd":1 ,"cfd":1 ,"nw":1 ,"hotspot":1 ,"lud":1 ,"ge":1 ,"srad":0 ,"heartwall":1 ,"bplustree":1}
# read SPEC run times of all apps in a data frame:
v100_runtime_df = pd.read_csv(v100_spec_runtime_file_path,sep=r'\s*,\s*',engine='python')
p100_runtime_df = pd.read_csv(p100_spec_runtime_file_path,sep=r'\s*,\s*',engine='python')

def app_SPEC_runtime(app,f,run_time_df,loc):
    id = app_id_dic[app]
    return run_time_df[id+'.'+app+'.EstBaseRunTime'].iloc[loc]

def clean_col(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    #.replace(',', '')
    if isinstance(x, str) and (len (x.split(' '))>1):
        return(x.replace(x.split(' ')[1], ''))
    return(x)

def r2rAVG(d1,d2,d3):
    cols = list(d1.columns)
    cols.remove('pstate')
    cols.remove('index')

    df = pd.DataFrame(columns=cols)
    
    for col in cols:
        if col == 'timestamp':
            df[col] = d1[col] 
            continue
        if col == 'temperature.gpu':
            d1[col] = d1[col].astype('int')
            d2[col] = d2[col].astype('int')
            d3[col] = d3[col].astype('int')
            df[col] = ((d1[col] + d2[col] + d3[col])/3).astype('int')
            continue

        datatype = 'int'

        if col == 'power.draw [W]':
            datatype = 'float'
            d1[col] = d1[col].apply(clean_col).astype(datatype)
            d2[col] = d2[col].apply(clean_col).astype(datatype)
            d3[col] = d3[col].apply(clean_col).astype(datatype)
            df[col] = ((d1[col] + d2[col] + d3[col])/3).astype(datatype)
            df[col] = df[col].round(2)

        else:
            datatype = 'int'
            d1[col] = d1[col].apply(clean_col).astype(datatype)
            d2[col] = d2[col].apply(clean_col).astype(datatype)
            d3[col] = d3[col].apply(clean_col).astype(datatype)
            df[col] = ((d1[col] + d2[col] + d3[col])/3).astype(datatype)
    
    df.reset_index(inplace=True,drop=True)
    return df

def readFile(path,file):
    rSize = []
    r1 = pd.read_csv(path+file+'-0',sep=r'\s*,\s*',engine='python')
    r1.fillna(method ='pad')
    r1 = r1[r1['index'] == 0]
    r1.reset_index(inplace=True,drop=True)

    r2 = pd.read_csv(path+file+'-1',sep=r'\s*,\s*',engine='python')
    r2.fillna(method ='pad')
    r2 = r2[r2['index'] == 0]
    r2.reset_index(inplace=True,drop=True)

    r3 = pd.read_csv(path+file+'-2',sep=r'\s*,\s*',engine='python')
    r3.fillna(method ='pad')
    r3 = r3[r3['index'] == 0]
    r3.reset_index(inplace=True,drop=True)

    rSize.append(len(r1.index))
    rSize.append(len(r2.index))
    rSize.append(len(r3.index))
    minSize = min(rSize)-1
    r1=r1[:minSize]
    r2=r2[:minSize]
    r3=r3[:minSize]
    
    # r1.reset_index()
    # r2.reset_index()
    # r3.reset_index()

    return r2rAVG (r1,r2,r3)


def combineCSVData (p100_path, v100_path):
    os.chdir(p100_path)
    all_filenames = [i for i in glob.glob('*')]
    files = []
    count = 0

    for f in all_filenames:
      
        if (f.split('-')[-1] != '0'):
            continue
        count += 1
        sArr = f.split('-')
        freq = sArr[-2]
        app = sArr[-3]
        del sArr[-1]
        fname = '-'.join(sArr)
        file = readFile(p100_path,fname)
        rows = len(file.index)
        print (count,app)
        # add TDP column
        tdp = [250 for i in range(rows)]
        file['TDP'] = tdp

        transistors = 15.3
        cores = 56
        die_size = 610
        arch = 'P100'

        transistors_list = [transistors for j in range(rows)]
        cores_list = [cores for k in range(rows)]
        diesize_list = [die_size for l in range(rows)]
        arch_list = [arch for m in range(rows)]
        
        file['transistors'] = transistors_list
        file['cores'] = cores_list
        file['die_size'] = diesize_list
        file['arch'] = arch_list

        run_time = app_SPEC_runtime(app,freq,p100_runtime_df,p100_freqs.index(int(freq)))
        run_time_arr = [run_time for o in range(rows)]
        file['base_run_time'] = run_time_arr

        app_id = app_type_dic[app]
        app_type = [app_id for q in range(rows)]
        file['app_type'] = app_type

        files.append(file)
    
    # # V100
    os.chdir(v100_path)
    v100_all_filenames = [i for i in glob.glob('*')]
    
    for f in v100_all_filenames:
        
        if (f.split('-')[-1] != '0'):
            continue
        count += 1
        sArr = f.split('-')
        freq = sArr[-2]
        app = sArr[-3]
        del sArr[-1]
        fname = '-'.join(sArr)
        file = readFile(v100_path,fname)
        rows = len(file.index)
        print (count,app)

        # add TDP column
        tdp_arr = [250 for i in range(rows)]
        file['TDP'] = tdp_arr

        transistors = 21.1
        cores = 80
        die_size = 815
        arch = 'V100'

        transistors_list = [transistors for j in range(rows)]
        cores_list = [cores for k in range(rows)]
        diesize_list = [die_size for l in range(rows)]
        arch_list = [arch for m in range(rows)]
        
        file['transistors'] = transistors_list
        file['cores'] = cores_list
        file['die_size'] = diesize_list
        file['arch'] = arch_list

        run_time = app_SPEC_runtime(app,freq,v100_runtime_df,v100_freqs.index(int(freq)))
        run_time_arr = [run_time for o in range(rows)]
        file['base_run_time'] = run_time_arr

        app_id = app_type_dic[app]
        app_type = [app_id for q in range(rows)]
        file['app_type'] = app_type

        files.append(file)
  
    
    # combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    
    combined_csv = pd.concat(files)
    combined_csv.reset_index(inplace=True,drop=True)
    print ('total rows:',len(combined_csv.index))
    #export to csv #/mnt/c/rf /home/ghali 
    combined_csv.to_csv("C:/rf/SPEC_gpu_p100_v100_metrics.csv", encoding='utf-8-sig') #mode='a', header=False,index=False,

# p100_arch = 'P100'
# v100_arch = 'V100'
p100_path = 'C:/rf/lbnl/data/SPEC/P100/dvfs/dvfs_profile/'
v100_path = 'C:/rf/lbnl/data/SPEC/V100/dvfs/dvfs_profile/'
combineCSVData(p100_path, v100_path)
