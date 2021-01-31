# Evaluation of power controls and counters on general-purpose Graphics Processing Units (GPUs)
General-purpose graphics processing units (GPUs) are increasingly becoming vital processing elements in high-end computing systems. A single advanced GPU may consume similar power (300 watts) as that of a high-performance computing (HPC) node.  The next-generation HPC systems could have up to 16 GPUs per node requiring multiple kilowatts per node. Consequently, it is quintessential to study the characteristics of GPU power consumption and control to inform future procurement design and decisions.

This repository is intended to a)launch GPU benchmarkings, b)generate GPU metrics data including power 
consumption for the launched benchmaring, and c) analyze GPU power consumption patterns.

## Getting Started

These instructions will get you a copy of the project up and intended to run on your GPU-enabled HPC node to:
* launch GPU benchmarking/workload
* generate GPU metrics data
* analye GPU power consumption pattern of the benchmaring/workload. 
See deployment for notes on how to deploy the project on a GPU-enabled HPC system.

### Prerequisites
* Install Nvidia CUDA 11 or CUDA 10.1 (This repository has been tested with both CUDA 10.1 and CUDA 11). For details, see [Nvidia CUDA installation](https://docs.nvidia.com/cuda/pdf/CUDA_Installation_Guide_Linux.pdf)
* Python 3 with required modules (pandas, matplotlib, and other Python modules)
* Compiled workload/benchmarking

## Overall Deployment Framework
The following diagram shows GPU Power Control & Counters Collection Framework.

![picture alt](imgs/gpu-fw.png "GPU Power Control & Counters Collection Framework")

The framework consists of three modules:
1.  Test module includes two directories (applications and data), two scripts (app_loadscaling and app_run2run), and analyze_data. 
* applications directory is place holder for executable application of whom power consumption behavior 
is to be analyzed. 
* data directory stores the GPU metrics data collected during the runtime of application. 
* app_loadscaling script enables launching of application with different input i.e. matrix sizes of 5K, 10K, 15K, 20K, and 25K.
* app_run2run enables launching of application with multiple runs to get run-to-run power variations
 of the application.
* analyze_data analyzes the GPU power data by calling the related functions in the analysis module.


2.  Utility module includes control and profile. control enforces the required GPU power control, executes 
profile script, and finally resetting the GPU control parameters to default Configuration. 
profile launches the application and starts nvidia-smi command to start collecting GPU metrics. 
Once application complete its runtime, it terminates the nvidia-smi command. 

3.  Power Analysis Engine analyzes the workloads power consumption from different perspectives 
including group bar plot (of all workloads with different power controls),
box plot (summary of power consumption of different benchmarking), power variations,
power consumption with varying load, and performance analysis (GFLOPS/s, GB/s).


## Running the tests
In order to generate the GPU metrics data and then perform different aspects of power analysis, follow these steps:

### clone the repository on the target GPU-enabled HPC node:
```
git clone https://github.com/nsfcac/gpupowermodel.git
```

### Configuration of Utility Functions

#### GPU Control parameters
* edit control script and enable/disable different control functions (e.g. power limit, frequency)
```
vi gpupowermodel/utility/control
```
#### GPU Profile Parameters
* edit profile script and change profile parameters (e.g. nvidia-smi sampling rate, adding/removing GPU metrics)
```
vi gpupowermodel/utility/profile
```

### Data Generation
```
cd gpupowermodel/test
```
* GPU metrics data collection for run-to-run application
```
./app_run2run.sh
```

* GPU metrics data collection for application with different matrix sizes
```
./app_loadscaling.sh
```

### Data Analysis

* Edit analyze_data.py available at: gpupower/test
* Run analyze_data.py to generate power analysis plots:
```
python3 analyze_data.py
```

## Testing with Historical GPU data

TBD

## Contributing

Further contributions to enhance and extend this work are welcome.


## Technical Support

In case of any technical issue in reproducing these results, you are welcome to contact Texas Tech University (TTU) developer: ghazanfar.ali@ttu.edu  


## Authors

* Mr. Ghazanfar Ali Sanpal, PhD student, Texas Tech University

* Dr. Sridutt Bhalachandra, Lawrence Berkeley National Laboratory 

* Dr. Nicholas Wright, Lawrence Berkeley National Laboratory


##  License

This project is licensed under [BSD 3-Clause License](https://github.com/nsfcac/Nagios-Redfish-API-Integration/blob/master/LICENSE)


## Acknowledgments

The National Energy Research Scientific Computing Center (NERSC) is a U.S. Department of Energy Office of Science User Facility operated under Contract No. DEAC02-05CH11231. Results presented in this paper were obtained using the Chameleon testbed supported by the National Science Foundation.
