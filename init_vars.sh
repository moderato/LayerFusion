#!/bin/bash
export LF_HOME=`pwd`
export DMLC_CORE=${TVM_HOME}/3rdparty/dmlc-core

# For VTune
sudo sh -c 'echo 0 >/proc/sys/kernel/perf_event_paranoid' 
sudo sh -c 'echo 0 >/proc/sys/kernel/kptr_restrict'
