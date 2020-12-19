import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle, islice

import pwrcommon as pc

def readPerfList(filePath,bDGEMM):
        perfFile = open(filePath)
        file_contents = perfFile.read()
        contents_split = file_contents.splitlines()
        data = []
        for item in contents_split:
                if bDGEMM:
                        d=item.split()[0]
                        data.append(float(d))
                else:
                        data.append(float(item)/1000)
        perfFile.close()
        return data

def pltPerfGroupBar(results,perfApps,bDGEMM,attainableGFs,attainableHBM):
        data = []
        stds = []
        cats = []
        apps = []
        
        for app in perfApps:
                appName = app[0]
                #print("\n"+appName+"\n")
                appCats = app[1]
                for cat in appCats:
                        #print("\n"+cat+"\n")
                        catName = cat[0]
                        fileID = cat[1]
                        d = readPerfList(results+fileID,bDGEMM)
                        p = statistics.mean(d)
                        s = statistics.stdev(d)
                        data.append(p)
                        stds.append(s)
                        cats.append(catName)
                        apps.append(appName)
        if bDGEMM:
                plt.figure()
                plt.style.use('classic')
                df = pd.DataFrame({'DGEMM':cats, 'GFLOPs/s':data})
                ax = sns.barplot(x="DGEMM", y="GFLOPs/s", data=df)
                fig = ax.get_figure()
                pltName = "dgemm_performance_variations.png"
                path = results + pltName
                #attainableGFs = 4800 #P100=4800, V100=7065
                ax.axhline(attainableGFs,linewidth=3, color='black',label="Attainable GFLOPs/s",linestyle='--')
                yX = 8000 #P100=5500
                plt.ylim(0, yX)
                plt.xlabel('DGEMM', weight='bold')  
                plt.ylabel('GFLOPs/s', weight='bold')
                plt.legend(loc='upper center',prop={'size': 9})
                plt.grid(True)
                fig.savefig(path)

        else:                
                df = pd.DataFrame({'STREAM':cats, 'HBM (GB/s)':data})
                plt.figure()
                plt.style.use('classic')
                ax = sns.barplot(x="STREAM", y="HBM (GB/s)", data=df)
                fig = ax.get_figure()
                pltName = "stream_performance_variations.png"
                path = results + pltName
                
                #attainableHBM = 600 #P100=600, V100=900
                ax.axhline(attainableHBM,linewidth=3, color='black',label="Attainable HBM Bandwidth (GB/s)",linestyle='--')
                yX = 1000#P100=700
                plt.ylim(0,  yX)
                plt.xlabel('STREAM', weight='bold')
                plt.ylabel('HBM (GB/s)', weight='bold')
                plt.legend(loc='upper center',prop={'size': 9})
                plt.grid(True)
                fig.savefig(path)

def grouped_barplot(df, cat, subcat, val, err,path ):
        
        u = df[cat].unique()
        x = np.arange(len(u))
        subx = df[subcat].unique()
        offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
        width= np.diff(offsets).mean()

        plt.figure()
        plt.style.use('classic')

        for i,gr in enumerate(subx):
                dfg = df[df[subcat] == gr]
                plt.bar(x+offsets[i], dfg[val].values, width=width,label="{}".format(gr), yerr=dfg[err].values)
        plt.xlabel(cat, weight='bold')
        plt.ylabel(val, weight='bold')
        plt.ylim(0,  300)
        plt.xticks(x, u)
        
        plt.axhline(250,linewidth=3, color='black',label="TDP",linestyle='--')
        
        plt.legend(loc='center left',prop={'size': 9})
        plt.grid(True)
        #plt.show()
        plt.savefig(path)

def pltPwrGroupBar(results,Apps,numRuns,threShold):
        data = []
        stds = []
        cats = []
        apps = []
        j=0
        for app in Apps:
                if j == 0:
                        threShold = 25
                        j += 1
                appName = app[0]
                appCats = app[1]
                i=0
                for cat in appCats:
                        if i == 0:
                                threShold = 25
                        catName = cat[0]
                        fileID = cat[1]
                        d = getPyMMPwr(results,fileID,numRuns,threShold)
                        p = statistics.mean(d)
                        s = statistics.stdev(d)
                        #print (p,s,catName,fileID)
                        if s > 10:
                                print('/n***I am special***/n')
                                print (p,s,catName,fileID)
                                print (d)
                                print('/n***I am special***/n') 

                        data.append(p)
                        stds.append(s)
                        cats.append(catName)
                        apps.append(appName)
                        
        df = pd.DataFrame(columns=['Benchmarks', 'cat', 'Power (W)', 'Std'])
        for n in range(len(data)):
                df.loc[n] = [apps[n]] + [cats[n]] + [data[n]] + [stds[n]]
        
        cat = "Benchmarks"
        subcat = "cat"
        val = "Power (W)"
        err = "Std"
        pltName = "power_variations.png"
        results = results + pltName
        grouped_barplot(df, cat, subcat, val, err,results)
        

def getPyMMPwr(output,fileID,numRuns,pwrThresh = 100):
        pyMMPwrs = []
        for r in numRuns:
                
                '''
                if fileID+r == 'dgemm-fp64-run-250-V100-135-8' or fileID+r == 'dgemm-fp64-run-250-V100-135-9':
                        print ("\n ***SKIPPING*** \n")
                        print (fileID+r)
                        print ("\n ***SKIPPING*** \n")
                        continue
                '''

                file = output+fileID+r
                data = pc.readMetrics(file)
                data,s,e = pc.getComputationStartEnd (data,pwrThresh)
                # striping non-computing power consumption
                df = pc.getStablePwr(data,s,e)
                pyMMPwrs.append(df['pwr'].mean())
        return pyMMPwrs
