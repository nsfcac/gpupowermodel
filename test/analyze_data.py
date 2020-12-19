import sys
import os

if __name__ == '__main__':
    app = 'dgemm'
    prec='fp64'
    load='load'
    run='run'
    p100='P100'
    v100='V100'
    pwr_limit='250'
    pwr_threshold=240
    numRuns = ['2','3','4','5','6','7','8','9','10','11']
    loadSize = ['16','32','48','64','72']
    #loadSize = ['5000']

    cwd = os.getcwd()
    output = cwd+'/results/'

    runFileID = app+'-'+prec+'-'+run+'-'+pwr_limit+'-'+v100+'-'
    loadFileID = app+'-'+prec+'-'+load+'-'+pwr_limit+'-'+v100+'-'

    #sys.path.insert(1,'../../analysis')
    sys.path.insert(1,'C:/rf/projects/gpupower/analysis')
    
    import pwrloadscaling
    import pwrperf
    import pwrvariations
    import pwrboxplot    
    import pwrbarplot as bp

    '''
    path = 'C:/rf/lbnl/data/phase2/p100-1/results/'
    
    attainableHBM = 600 #P100=600, V100=900
    attainableGFs = 4800 #P100=4800, V100=7065
    
    perfDGEMM = [["DGEMM",[["DVFS(Low)","run_perf_dvfs_544"],["PowerCap(Low)","run_perf_pl_125"],["Performance","run_perf_pl_250"]]]]

    bDGEMM = True
    bp.pltPerfGroupBar(path,perfDGEMM,bDGEMM,attainableGFs,attainableHBM)

    #bDGEMM = False
    #bp.pltPerfGroupBar(path,perfSTREAM,bDGEMM,attainableGFs,attainableHBM)
    '''

    '''
    # plot power usage per workload
    for size in loadSize:
        pwrperf.pltIndividual(output,loadFileID, size)
    '''
    # plot power usage for different matrix sizes
    fileID = 'dgemm-fp64-load-250-V100-P0-'
    path = 'C:/rf/lbnl/data/phase1/v100/results/'
    plotName = "DGEMMLoadScaling.png"
    pwrloadscaling.pltPwrLoadScaling(path,fileID,loadSize,plotName)
    inputSize = ['5000','10000','15000','20000','25000']    
    plotName = "StreamLoadScaling.png"
    fileID = 'stream-fp64-load-250-V100-P0-'
    pwrloadscaling.pltPwrLoadScaling(path,fileID,inputSize,plotName)

    #plot Gflops and Gflops/watt
    #pwrperf.pltPwrPerf(output,loadFileID,loadSize,pwr_threshold)
    
    '''
    # plot variations i.e. powr raming up, power stabilization at peak, and power ramping down
    pwrvariations.pltVarations(output,runFileID,'11')
    
    # draw boxplot
    pwrboxplot.drawBoxPlt(output,runFileID,numRuns)
    '''
