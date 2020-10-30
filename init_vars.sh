#!/bin/bash
export LF_HOME=`pwd`
export TVM_HOME=${HOME}/Documents/incubator-tvm
export DMLC_CORE=${TVM_HOME}/3rdparty/dmlc-core
# export VTUNE_HOME=${HOME}/intel/vtune_profiler
export LIBXSMM_HOME=${HOME}/Documents/libxsmm
export MKLDNN_HOME=${HOME}/Documents/mkl-dnn
export PCM_HOME=${HOME}/Documents/pcm
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TVM_HOME}/build"

# icc & icpc
source /opt/intel/oneapi/setvars.sh >& /dev/null

# Make folders
mkdir -p logs/arithmetic_intensity/cpu
mkdir -p logs/arithmetic_intensity/gpu
mkdir -p logs/autotvm/layer/cpu
mkdir -p logs/autotvm/layer/gpu
mkdir -p logs/autotvm/model/cpu
mkdir -p logs/autotvm/model/gpu
mkdir -p logs/gflops/cpu
mkdir -p logs/gflops/gpu
mkdir -p logs/runtime/cpu
mkdir -p logs/runtime/gpu
mkdir -p generated_kernels/cpu/kernel_launch_config
mkdir -p generated_kernels/gpu/kernel_launch_config
mkdir -p npy

# # For VTune benchmark
# sudo ln -sf ${VTUNE_HOME}/bin64/vtune /usr/bin/vtune 
# sudo sh -c 'echo 0 >/proc/sys/kernel/perf_event_paranoid' 
# sudo sh -c 'echo 0 >/proc/sys/kernel/kptr_restrict'
# export INTEL_LIBITTNOTIFY32=${VTUNE_HOME}/lib32/runtime/libittnotify_collector.so
# export INTEL_LIBITTNOTIFY64=${VTUNE_HOME}/lib64/runtime/libittnotify_collector.so

# For SDE benchmark
sudo sh -c 'echo 0 > /proc/sys/kernel/yama/ptrace_scope'

# For PCM
sudo modprobe msr
sudo chown -R $USER /dev/cpu/*/msr
sudo chmod -R go+rw /dev/cpu/*/msr

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

# For Libxsmm wrapper
cd libxsmm_wrapper
make
if [ "$LIBXSMM_PRELOADED" != "1" ];
then
  export LD_PRELOAD="${LD_PRELOAD}:${LF_HOME}/libxsmm_wrapper/libxsmm_wrapper.so"
fi
# export LIBXSMM_PRELOADED=1
cd ..
