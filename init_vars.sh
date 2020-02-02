#!/bin/bash
export LF_HOME=`pwd`

# For VTune
sudo sh -c 'echo 0 >/proc/sys/kernel/perf_event_paranoid' 
sudo sh -c 'echo 0 >/proc/sys/kernel/kptr_restrict'
