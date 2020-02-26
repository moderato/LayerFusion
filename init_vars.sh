#!/bin/bash
export LF_HOME=`pwd`
export DMLC_CORE=${TVM_HOME}/3rdparty/dmlc-core
export VTUNE_HOME=${HOME}/intel/vtune_profiler

# For VTune benchmark
sudo sh -c 'echo 0 >/proc/sys/kernel/perf_event_paranoid' 
sudo sh -c 'echo 0 >/proc/sys/kernel/kptr_restrict'

# For CPU better performance
export KMP_BLOCKTIME=1
export KMP_AFFINITY=verbose,granularity=fine,compact,1,0 # "verbose" for printing
export OMP_NUM_THREADS=4 # = number of cores
export TVM_BIND_THREADS=4
# export TVM_NUM_THREADS=4 # Actual thread num = max(OMP_NUM_THREADS, TVM_NUM_THREADS)

# Disable turbo boost and set scaling_governor to "performance"
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo &> /dev/null
for i in 1 2 3 4
do
  echo performance | sudo tee /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor &> /dev/null
done