import matplotlib
import matplotlib.pyplot as plt

import pwrcommon as pc

# mode = 'run'
# app = 'pymm'
prec = 'fp64'
# pstate = 'p0'
# pl = 'tdp'
# mclk = 'm715'
# pclk = 'p1189'
# fPrefix = prec +

def drawBoxPlt(output,fileID,numRuns):
        pyMMPwrs = getPyMMPwr(output,fileID,numRuns)
        
        box_plot_data=[pyMMPwrs]
        fig = plt.figure()
        box = plt.boxplot(box_plot_data,patch_artist=True,showmeans=True,labels=['PyMM'],)
 
        colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
        for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
 
        #plt.show()
        plt.savefig(output+fileID+'boxplot.png')

def getPyMMPwr(output,fileID,numRuns,pwrThresh = 100):
        pyMMPwrs = []
        for r in numRuns:
                file = output+fileID+r
                data = pc.readMetrics(file)
                data,s,e = pc.getComputationStartEnd (data,pwrThresh)
                # striping non-computing power consumption
                df = pc.getStablePwr(data,s,e)
                pyMMPwrs.append(df['pwr'].mean())
        return pyMMPwrs
