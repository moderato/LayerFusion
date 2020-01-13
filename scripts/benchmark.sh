#!/bin/bash
input="../workloads.csv"
line_count=0
while IFS= read -r line
do
  IFS=',' read -ra PARAM <<< "$line"
  line_count=$((line_count+1))
  if [ $line_count != 1 ]; # Skip first line
  then
    idx=0
    workload_name=${PARAM[0]}
    cd ../benchmark
    nvcc -arch=sm_35 -std=c++11 -O2 \
      -I ${CUDA_HOME}/include \
      -L ${CUDA_HOME}/lib -L/usr/local/lib \
      -D REPEATITION=5 \
      benchmark.cu \
      -o benchmark \
      -lcnpy -lz -lcudnn -include ../generated_kernels/${workload_name}.cuh
    nvprof --metrics flop_count_sp --metrics dram_read_transactions --metrics dram_write_transactions --log-file /tmp/nvprof_generated.txt ./benchmark "$line" 0
    total_dram_read="$(cat /tmp/nvprof_generated.txt | grep dram_read_transactions | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
    total_dram_write="$(cat /tmp/nvprof_generated.txt | grep dram_write_transactions | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
    total_flop="$(cat /tmp/nvprof_generated.txt | grep flop_count_sp | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
    generated_kernel_arithmetic_intensity="$( echo "scale=4; $total_flop * 1.0 / (($total_dram_read + $total_dram_write) * 32)" | bc)"
    nvprof --metrics flop_count_sp --metrics dram_read_transactions --metrics dram_write_transactions --log-file /tmp/nvprof_cudnn.txt ./benchmark "$line" 1
    total_dram_read="$(cat /tmp/nvprof_cudnn.txt | grep dram_read_transactions | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
    total_dram_write="$(cat /tmp/nvprof_cudnn.txt | grep dram_write_transactions | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
    total_flop="$(cat /tmp/nvprof_cudnn.txt | grep flop_count_sp | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
    cudnn_arithmetic_intensity="$( echo "scale=4; $total_flop * 1.0 / (($total_dram_read + $total_dram_write) * 32)" | bc)"
    echo $generated_kernel_arithmetic_intensity
    echo $cudnn_arithmetic_intensity
    cd ../scripts
  fi
done < "$input"

