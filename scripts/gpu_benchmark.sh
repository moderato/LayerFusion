#!/bin/bash
REP_BENCH=20 # By default 20 * 2 = 40 iterations for flops and bandwidth measurements
if [ "$#" == 1 ] ;
then
  REP_BENCH="$1"
fi
echo "$(( ${REP_BENCH} * 2 )) iterations for profiling."

for input in "../workloads/depth_conv_workloads.csv" "../workloads/conv_conv_workloads.csv"
do
  line_count=0
  while IFS= read -r line
  do
    IFS=',' read -ra PARAM <<< "$line"
    line_count=$((line_count+1))
    if [ $line_count != 1 ]; # Skip first line
    then
      idx=0
      workload_name=${PARAM[0]}
      cd ../benchmark/gpu

      # Repeatition=1000 for runtime measurement
      make KERNEL=../../generated_kernels/gpu/fused/${workload_name}.cuh >& /dev/null

      ./gpu_bench "$line" 0 &> /tmp/runtime_generated.txt
      generated_kernel_runtime="$(cat /tmp/runtime_generated.txt | grep Fusion | awk '{ printf  "%10s\n", $4 }')"
      ./gpu_bench "$line" 1 &> /tmp/runtime_cudnn.txt
      cudnn_runtime="$(cat /tmp/runtime_cudnn.txt | grep Fusion | awk '{ printf "%10s\n", $4 }')"

      # Repeatition=20 for arithmetic intensity
      make KERNEL=../../generated_kernels/gpu/fused/${workload_name}.cuh REPEATITION=$(( ${REP_BENCH} * 2 )) >& /dev/null

      nvprof --metrics flop_count_sp \
              --metrics dram_read_transactions \
              --metrics dram_write_transactions \
              --metrics l2_read_transactions \
              --metrics l2_write_transactions \
              --log-file /tmp/nvprof_generated.txt \
              ./gpu_bench "$line" 0 >& /dev/null
      total_dram_read="$(cat /tmp/nvprof_generated.txt | grep dram_read_transactions | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
      total_dram_write="$(cat /tmp/nvprof_generated.txt | grep dram_write_transactions | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
      total_l2_read="$(cat /tmp/nvprof_generated.txt | grep l2_read_transactions | awk '{ x=gensub("\t","","G",$8); printf x "+" } END{ print 0 }' | bc -l)"
      total_l2_write="$(cat /tmp/nvprof_generated.txt | grep l2_write_transactions | awk '{ x=gensub("\t","","G",$8); printf x "+" } END{ print 0 }' | bc -l)"
      generated_kernel_total_flop="$(cat /tmp/nvprof_generated.txt | grep flop_count_sp | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
      generated_kernel_dram_ai="$( echo "scale=4; $generated_kernel_total_flop * 1.0 / (($total_dram_read + $total_dram_write) * 32)" | bc)"
      generated_kernel_l2_ai="$( echo "scale=4; $generated_kernel_total_flop * 1.0 / (($total_l2_read + $total_l2_write) * 32)" | bc)"
      # echo "------ Generated kernel details"
      # echo "    Runtime: $generated_kernel_runtime"
      # echo "    Flop: $generated_kernel_total_flop"
      # echo "    DRAM bytes: $(( ${total_dram_read} + ${total_dram_write} ))"
      # echo "    DRAM AI: $generated_kernel_dram_ai"
      # echo "    L2 bytes: $(( ${total_l2_read} + ${total_l2_write} ))"
      # echo "    L2 AI: $generated_kernel_l2_ai"

      nvprof --metrics flop_count_sp \
              --metrics dram_read_transactions \
              --metrics dram_write_transactions \
              --metrics l2_read_transactions \
              --metrics l2_write_transactions \
              --log-file /tmp/nvprof_cudnn.txt \
              ./gpu_bench "$line" 1 >& /dev/null
      total_dram_read="$(cat /tmp/nvprof_cudnn.txt | grep "$(( ${REP_BENCH} * 2 + 2)).*dram_read_transactions" | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
      total_dram_write="$(cat /tmp/nvprof_cudnn.txt | grep "$(( ${REP_BENCH} * 2 + 2)).*dram_write_transactions" | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
      total_l2_read="$(cat /tmp/nvprof_cudnn.txt | grep "$(( ${REP_BENCH} * 2 + 2)).*l2_read_transactions" | awk '{ x=gensub("\t","","G",$8); printf x "+" } END{ print 0 }' | bc -l)"
      total_l2_write="$(cat /tmp/nvprof_cudnn.txt | grep "$(( ${REP_BENCH} * 2 + 2)).*l2_write_transactions" | awk '{ x=gensub("\t","","G",$8); printf x "+" } END{ print 0 }' | bc -l)"
      cudnn_total_flop="$(cat /tmp/nvprof_cudnn.txt | grep "$(( ${REP_BENCH} * 2 + 2)).*flop_count_sp" | awk '{ x=gensub("\t","","G",$9); printf x "+" } END{ print 0 }' | bc -l)"
      cudnn_dram_ai="$( echo "scale=4; $cudnn_total_flop * 1.0 / (($total_dram_read + $total_dram_write) * 32)" | bc)"
      cudnn_l2_ai="$( echo "scale=4; $cudnn_total_flop * 1.0 / (($total_l2_read + $total_l2_write) * 32)" | bc)"
      # echo "------ CuDNN details"
      # echo "    Runtime: $cudnn_runtime"
      # echo "    Flop: $cudnn_total_flop"
      # echo "    DRAM bytes: $(( ${total_dram_read} + ${total_dram_write} ))"
      # echo "    DRAM AI: $cudnn_dram_ai"
      # echo "    L2 bytes: $(( ${total_l2_read} + ${total_l2_write} ))"
      # echo "    L2 AI: $cudnn_l2_ai"

      # Output results to files
      cd ../../
      mkdir -p logs/runtime/gpu
      echo -e "generated,cudnn\n$generated_kernel_runtime,$cudnn_runtime" > "logs/runtime/gpu/${workload_name}.csv"
      mkdir -p logs/arithmetic_intensity/gpu
      echo -e "generated_dram,cudnn_dram,generated_l2,cudnn_l2\n$generated_kernel_dram_ai,$cudnn_dram_ai,$generated_kernel_l2_ai,$cudnn_l2_ai" > "logs/arithmetic_intensity/gpu/${workload_name}.csv"
      mkdir -p logs/gflops/gpu
      generated_kernel_gflops="$(echo "scale=4; $generated_kernel_total_flop * 1.0 / $generated_kernel_runtime / 1000.0" | bc -l)"
      cudnn_gflops="$(echo "scale=4; $cudnn_total_flop * 1.0 / $cudnn_runtime / 1000.0" | bc -l)"
      echo -e "generated,cudnn\n$generated_kernel_gflops,$cudnn_gflops" > "logs/gflops/gpu/${workload_name}.csv"
      cd scripts

      # Print results
      echo "###################"
      echo "Workload name: ${workload_name}"
      echo "Generated/cuDNN runtime: ${generated_kernel_runtime} us, ${cudnn_runtime} us."
      echo "Generated/cuDNN DRAM AI: ${generated_kernel_dram_ai}, ${cudnn_dram_ai}."
      echo "Generated/cuDNN L2 AI: ${generated_kernel_l2_ai}, ${cudnn_l2_ai}."
      echo "Generated/cuDNN GFLOPS: ${generated_kernel_gflops}, ${cudnn_gflops}."
    fi
  done < "$input"
done
