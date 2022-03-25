#! /usr/bin/env python
# Python power utility to measure power consumed by any process
# No root privileges are necessary
# It is recommended that the running time of a process is at least 'X' seconds
# to get acceptable power consumption results

import os
import time
import signal
import subprocess
import sys
import random

#r = str(random.random())
# outputFileDirectory='/home/userName/output/'
# outputFileStartPattern = 'dpgpu_mets_'
# outputFile = outputFileDirectory+outputFileStartPattern+str(size)
outputFile = os.path.join(sys.path[0], "changeme")

outputFormat = "csv"
interval = "1"    # millisecond
initTime = 0

# monitoringCmd = "dcgmi dmon -i 0 -e 1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,203,204,210,211,155,156,110 -d 1 > changeme"
monitoringCmd = "dcgmi dmon -i 0 -e 1001,1002,1003,1005,1006,1007,1008,203,204,210,211,155,156,110 -d 1 > changeme"
# +------+------------------+
# | F_ID | F_Tag            |
# +------+------------------+
# | 1003 | sm_occupancy     |
# | 1002 | sm_active        |
# | 1004 | tensor_active    |
# | 1007 | fp32_active      |
# | 1006 | fp64_active      |
# | 1008 | fp16_active      |
# | 1005 | dram_active      |
# | 1009 | pcie_tx_bytes    |
# | 1010 | pcie_rx_bytes    |
# | 1001 | gr_engine_active |
# | 1011 | nvlink_tx_bytes  |
# | 1012 | nvlink_rx_bytes  |
# +------+------------------+

def get_power_info(cmd):
    global initTime
    global pid
    initTime = time.time()
    print ("profile cmd:",cmd)
    pid = os.fork()
    if pid == 0:
        code = os.system(monitoringCmd)
        print(" === [monitoringCmd] Interval for pid" + str(pid) + ":" + \
            str(round(time.time() - initTime, 2)) + " s")
    else:
        os.system(cmd)
        os.system("killall -9 dcgmi")
        print(" === [cmd] Interval for pid" + str(pid) + ":" + \
            str(round(time.time() - initTime, 2)) + " s")
    return

if __name__ == '__main__':
    print(" *** Start of of __main__ for profile.py! ***")
    if len(sys.argv) < 2:
        get_power_info('')
    else:
        get_power_info(' '.join(str(x) for x in sys.argv[1:]))
    print(" *** End of __main__ for profile.py! ***")
