#!/bin/bash
REP_BENCH=20 # By default 20 * 2 = 40 iterations for flops and bandwidth measurements
if [ "$#" == 1 ] ;
then
  REP_BENCH="$1"
fi
echo "$(( ${REP_BENCH} * 2 )) iterations for profiling."

for input in "../workloads/depth_conv_workloads.csv"
do
  line_count=0
  while IFS= read -r line
  do
    line_count=$((line_count+1))
    if [ $line_count != 1 ]; # Skip first line
    then
      # Get workload layer descriptions
      ./parse_workload_input.sh $line > /tmp/workload_layers.txt
      workload_name="$(cat /tmp/workload_layers.txt | grep -e 'workload_name' | awk '{ printf "%s\n", $2 }')"
      layer_1_desc="$(cat /tmp/workload_layers.txt | grep -e 'layer_1_desc' | awk '{ printf "%s\n", $2 }')"
      layer_2_desc="$(cat /tmp/workload_layers.txt | grep -e 'layer_2_desc' | awk '{ printf "%s\n", $2 }')"
      echo ${workload_name}
      # echo ${layer_1_desc}
      # echo ${layer_2_desc}

      #### Runtime ####
      cd ../benchmark/cpu

      # Repeatition=1000 for runtime measurement
      make KERNEL_FUSED=../../generated_kernels/cpu/fused/${workload_name}.asm \
            KERNEL_UNFUSED_1=../../generated_kernels/cpu/unfused/${workload_name}_1.asm \
            KERNEL_UNFUSED_2=../../generated_kernels/cpu/unfused/${workload_name}_2.asm >& /dev/null
      numactl -l -C 0-3 ./cpu_bench "$line" 0 &> /tmp/runtime_mkldnn.txt
      numactl -l -C 0-3 ./cpu_bench "$line" 1 &> /tmp/runtime_fused.txt
      numactl -l -C 0-3 ./cpu_bench "$line" 2 &> /tmp/runtime_unfused.txt

      mkldnn_runtime_1="$(cat /tmp/runtime_mkldnn.txt | grep 'Stage 1 runtime' | awk '{ printf  "%10s\n", $5 }')"
      mkldnn_runtime_2="$(cat /tmp/runtime_mkldnn.txt | grep 'Stage 2 runtime' | awk '{ printf  "%10s\n", $5 }')"
      mkldnn_total_runtime="$(cat /tmp/runtime_mkldnn.txt | grep 'Total' | awk '{ printf  "%10s\n", $4 }')"
      fused_kernel_runtime="$(cat /tmp/runtime_fused.txt | grep Fusion | awk '{ printf  "%10s\n", $4 }')"
      unfused_kernel_runtime_1="$(cat /tmp/runtime_unfused.txt | grep 'Stage 1 runtime' | awk '{ printf  "%10s\n", $5 }')"
      unfused_kernel_runtime_2="$(cat /tmp/runtime_unfused.txt | grep 'Stage 2 runtime' | awk '{ printf  "%10s\n", $5 }')"
      unfused_total_runtime="$(cat /tmp/runtime_unfused.txt | grep 'Total' | awk '{ printf  "%10s\n", $4 }')"

      mkldnn_dram_bytes_1_th="$(cat /tmp/runtime_mkldnn.txt | grep 'Stage 1 Theoretical DRAM' | awk '{ printf  "%10s\n", $6 }')"
      mkldnn_dram_bytes_2_th="$(cat /tmp/runtime_mkldnn.txt | grep 'Stage 2 Theoretical DRAM' | awk '{ printf  "%10s\n", $6 }')"
      mkldnn_flop_1_th="$(cat /tmp/runtime_mkldnn.txt | grep 'Stage 1 Theoretical FLOP' | awk '{ printf  "%10s\n", $5 }')"
      mkldnn_flop_2_th="$(cat /tmp/runtime_mkldnn.txt | grep 'Stage 2 Theoretical FLOP' | awk '{ printf  "%10s\n", $5 }')"
      fused_kernel_dram_bytes_th="$(cat /tmp/runtime_fused.txt | grep 'Theoretical DRAM' | awk '{ printf  "%10s\n", $4 }')"
      fused_kernel_flop_th="$(cat /tmp/runtime_fused.txt | grep 'Theoretical FLOP' | awk '{ printf  "%10s\n", $3 }')"
      unfused_kernel_dram_bytes_1_th="$(cat /tmp/runtime_unfused.txt | grep 'Stage 1 Theoretical DRAM' | awk '{ printf  "%10s\n", $6 }')"
      unfused_kernel_dram_bytes_2_th="$(cat /tmp/runtime_unfused.txt | grep 'Stage 2 Theoretical DRAM' | awk '{ printf  "%10s\n", $6 }')"
      unfused_kernel_flop_1_th="$(cat /tmp/runtime_unfused.txt | grep 'Stage 1 Theoretical FLOP' | awk '{ printf  "%10s\n", $5 }')"
      unfused_kernel_flop_2_th="$(cat /tmp/runtime_unfused.txt | grep 'Stage 2 Theoretical FLOP' | awk '{ printf  "%10s\n", $5 }')"

      #### Roofline Analysis ####
      # Flexible repeatition for flops and memory transaction measurement
      make KERNEL_FUSED=../../generated_kernels/cpu/fused/${workload_name}.asm \
            KERNEL_UNFUSED_1=../../generated_kernels/cpu/unfused/${workload_name}_1.asm \
            KERNEL_UNFUSED_2=../../generated_kernels/cpu/unfused/${workload_name}_2.asm REPEATITION=${REP_BENCH} ENABLE_PCM=1 LAYER_1=1 >& /dev/null
      setcap cap_sys_rawio=ep cpu_bench

      # MKLDNN 1
      echo "MKLDNN 1"
      numactl -l -C 0-3 ./cpu_bench "$line" 0 >& /tmp/dram_mkldnn.txt
      rm -rf /tmp/sde_mkldnn_1.out*
      sde -knl -d -iform 1 -omix /tmp/sde_mkldnn_1.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- numactl -l -C 0-3 ./cpu_bench "$line" 0 >& /dev/null

      # Fused kernel
      echo "Fused"
      numactl -l -C 0-3 ./cpu_bench "$line" 1 >& /tmp/dram_fused.txt
      rm -rf /tmp/sde_fused.out*
      sde -knl -d -iform 1 -omix /tmp/sde_fused.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- numactl -l -C 0-3 ./cpu_bench "$line" 1 >& /dev/null

      # Unfused kernel 1
      echo "Unfused 1"
      numactl -l -C 0-3 ./cpu_bench "$line" 2 >& /tmp/dram_unfused.txt
      rm -rf /tmp/sde_unfused_1.out*
      sde -knl -d -iform 1 -omix /tmp/sde_unfused_1.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- numactl -l -C 0-3 ./cpu_bench "$line" 2 >& /dev/null

      make KERNEL_FUSED=../../generated_kernels/cpu/fused/${workload_name}.asm \
            KERNEL_UNFUSED_1=../../generated_kernels/cpu/unfused/${workload_name}_1.asm \
            KERNEL_UNFUSED_2=../../generated_kernels/cpu/unfused/${workload_name}_2.asm REPEATITION=${REP_BENCH} ENABLE_PCM=1 LAYER_2=1 >& /dev/null
      setcap cap_sys_rawio=ep cpu_bench

      # MKLDNN 2
      echo "MKLDNN 2"
      rm -rf /tmp/sde_mkldnn_2.out*
      sde -knl -d -iform 1 -omix /tmp/sde_mkldnn_2.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- numactl -l -C 0-3 ./cpu_bench "$line" 0 >& /dev/null

      # Unfused kernel 2
      echo "Unfused 2"
      rm -rf /tmp/sde_unfused_2.out*
      sde -knl -d -iform 1 -omix /tmp/sde_unfused_2.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- numactl -l -C 0-3 ./cpu_bench "$line" 2 >& /dev/null

      #### Parse the above results ####
      cd ../../scripts

      ./parse_sde.sh /tmp/sde_mkldnn_1.out* > /tmp/sde_mkldnn_1.txt
      mkldnn_flop_1="$(cat /tmp/sde_mkldnn_1.txt | grep -e 'Total single-precision FLOPs' | awk '{ printf "%s\n", $5 }')"
      mkldnn_flop_1="$( echo "scale=4; $mkldnn_flop_1 / (${REP_BENCH} * 2)" | bc )"
      mkldnn_l1_bytes_1="$(cat /tmp/sde_mkldnn_1.txt | grep -e 'Total Bytes =' | awk '{ printf "%s\n", $4 }')"
      mkldnn_l1_bytes_1="$( echo "scale=4; $mkldnn_l1_bytes_1 / (${REP_BENCH} * 2)" | bc )" # Per iteration
      mkldnn_dram_bytes_1="$(cat /tmp/dram_mkldnn.txt | grep 'Stage 1 DRAM bytes' | awk '{ printf  "%10s\n", $5 }')"

      ./parse_sde.sh /tmp/sde_mkldnn_2.out* > /tmp/sde_mkldnn_2.txt
      mkldnn_flop_2="$(cat /tmp/sde_mkldnn_2.txt | grep -e 'Total single-precision FLOPs' | awk '{ printf "%s\n", $5 }')"
      mkldnn_flop_2="$( echo "scale=4; $mkldnn_flop_2 / (${REP_BENCH} * 2)" | bc )"
      mkldnn_l1_bytes_2="$(cat /tmp/sde_mkldnn_2.txt | grep -e 'Total Bytes =' | awk '{ printf "%s\n", $4 }')"
      mkldnn_l1_bytes_2="$( echo "scale=4; $mkldnn_l1_bytes_2 / (${REP_BENCH} * 2)" | bc )" # Per iteration
      mkldnn_dram_bytes_2="$(cat /tmp/dram_mkldnn.txt | grep 'Stage 2 DRAM bytes' | awk '{ printf  "%10s\n", $5 }')"

      ./parse_sde.sh /tmp/sde_fused.out* > /tmp/sde_fused.txt
      fused_kernel_flop="$(cat /tmp/sde_fused.txt | grep -e 'Total single-precision FLOPs' | awk '{ printf "%s\n", $5 }')"
      fused_kernel_flop="$( echo "scale=4; $fused_kernel_flop / (${REP_BENCH} * 2)" | bc )"
      fused_l1_bytes="$(cat /tmp/sde_fused.txt | grep -e 'Total Bytes =' | awk '{ printf "%s\n", $4 }')"
      fused_l1_bytes="$( echo "scale=4; $fused_l1_bytes / (${REP_BENCH} * 2)" | bc )" # Per iteration
      fused_dram_bytes="$(cat /tmp/dram_fused.txt | grep 'Total DRAM bytes' | awk '{ printf  "%10s\n", $4 }')"

      ./parse_sde.sh /tmp/sde_unfused_1.out* > /tmp/sde_unfused_1.txt
      unfused_kernel_flop_1="$(cat /tmp/sde_unfused_1.txt | grep -e 'Total single-precision FLOPs' | awk '{ printf "%s\n", $5 }')"
      unfused_kernel_flop_1="$( echo "scale=4; $unfused_kernel_flop_1 / (${REP_BENCH} * 2)" | bc )"
      unfused_l1_bytes_1="$(cat /tmp/sde_unfused_1.txt | grep -e 'Total Bytes =' | awk '{ printf "%s\n", $4 }')"
      unfused_l1_bytes_1="$( echo "scale=4; $unfused_l1_bytes_1 / (${REP_BENCH} * 2)" | bc )" # Per iteration
      unfused_dram_bytes_1="$(cat /tmp/dram_unfused.txt | grep 'Stage 1 DRAM bytes' | awk '{ printf  "%10s\n", $5 }')"

      ./parse_sde.sh /tmp/sde_unfused_2.out* > /tmp/sde_unfused_2.txt
      unfused_kernel_flop_2="$(cat /tmp/sde_unfused_2.txt | grep -e 'Total single-precision FLOPs' | awk '{ printf "%s\n", $5 }')"
      unfused_kernel_flop_2="$( echo "scale=4; $unfused_kernel_flop_2 / (${REP_BENCH} * 2)" | bc )"
      unfused_l1_bytes_2="$(cat /tmp/sde_unfused_2.txt | grep -e 'Total Bytes =' | awk '{ printf "%s\n", $4 }')"
      unfused_l1_bytes_2="$( echo "scale=4; $unfused_l1_bytes_2 / (${REP_BENCH} * 2)" | bc )" # Per iteration
      unfused_dram_bytes_2="$(cat /tmp/dram_unfused.txt | grep 'Stage 2 DRAM bytes' | awk '{ printf  "%10s\n", $5 }')"

      # GFLOPS and AI
      mkldnn_gflops_1="$( echo "scale=4; $mkldnn_flop_1 / $mkldnn_runtime_1 * 1000000 / 1000000000" | bc )"
      mkldnn_l1_ai_1="$( echo "scale=4; $mkldnn_flop_1 / $mkldnn_l1_bytes_1" | bc )"
      mkldnn_dram_ai_1="$( echo "scale=4; $mkldnn_flop_1 / $mkldnn_dram_bytes_1" | bc )"
      # echo "------ MKLDNN kernel 1 details"
      # echo "    Runtime: $mkldnn_runtime_1"
      # echo "    Theoretical Flop: $mkldnn_flop_1_th"
      # echo "    Flop: $mkldnn_flop_1"
      # echo "    GFLOPS: $mkldnn_gflops_1"
      # echo "    Theoretical DRAM bytes: $mkldnn_dram_bytes_1_th"
      # echo "    DRAM bytes: $mkldnn_dram_bytes_1"
      # echo "    DRAM AI: $mkldnn_dram_ai_1"
      # echo "    L1 bytes: $mkldnn_l1_bytes_1"
      # echo "    L1 AI: $mkldnn_l1_ai_1"

      mkldnn_gflops_2="$( echo "scale=4; $mkldnn_flop_2 / $mkldnn_runtime_2 * 1000000 / 1000000000" | bc )"
      mkldnn_l1_ai_2="$( echo "scale=4; $mkldnn_flop_2 / $mkldnn_l1_bytes_2" | bc )"
      mkldnn_dram_ai_2="$( echo "scale=4; $mkldnn_flop_2 / $mkldnn_dram_bytes_2" | bc )"
      # echo "------ MKLDNN kernel 2 details"
      # echo "    Runtime: $mkldnn_runtime_2"
      # echo "    Theoretical Flop: $mkldnn_flop_2_th"
      # echo "    Flop: $mkldnn_flop_2"
      # echo "    GFLOPS: $mkldnn_gflops_2"
      # echo "    Theoretical DRAM bytes: $mkldnn_dram_bytes_2_th"
      # echo "    DRAM bytes: $mkldnn_dram_bytes_2"
      # echo "    DRAM AI: $mkldnn_dram_ai_2"
      # echo "    L1 bytes: $mkldnn_l1_bytes_2"
      # echo "    L1 AI: $mkldnn_l1_ai_2"

      fused_kernel_gflops="$( echo "scale=4; $fused_kernel_flop / $fused_kernel_runtime * 1000000 / 1000000000" | bc )"
      fused_kernel_l1_ai="$( echo "scale=4; $fused_kernel_flop / $fused_l1_bytes" | bc )"
      fused_kernel_dram_ai="$( echo "scale=4; $fused_kernel_flop / $fused_dram_bytes" | bc )"
      # echo "------ Fused kernel details"
      # echo "    Runtime: $fused_kernel_runtime"
      # echo "    Theoretical Flop: $fused_kernel_flop_th"
      # echo "    Flop: $fused_kernel_flop"
      # echo "    GFLOPS: $fused_kernel_gflops"
      # echo "    Theoretical DRAM bytes: $fused_kernel_dram_bytes_th"
      # echo "    DRAM bytes: $fused_dram_bytes"
      # echo "    DRAM AI: $fused_kernel_dram_ai"
      # echo "    L1 bytes: $fused_l1_bytes"
      # echo "    L1 AI: $fused_kernel_l1_ai"

      unfused_kernel_gflops_1="$( echo "scale=4; $unfused_kernel_flop_1 / $unfused_kernel_runtime_1 * 1000000 / 1000000000" | bc )"
      unfused_kernel_l1_ai_1="$( echo "scale=4; $unfused_kernel_flop_1 / $unfused_l1_bytes_1" | bc )"
      unfused_kernel_dram_ai_1="$( echo "scale=4; $unfused_kernel_flop_1 / $unfused_dram_bytes_1" | bc )"
      # echo "------ Unfused kernel 1 details"
      # echo "    Runtime: $unfused_kernel_runtime_1"
      # echo "    Theoretical Flop: $unfused_kernel_flop_1_th"
      # echo "    Flop: $unfused_kernel_flop_1"
      # echo "    GFLOPS: $unfused_kernel_gflops_1"
      # echo "    Theoretical DRAM bytes: $unfused_kernel_dram_bytes_1_th"
      # echo "    DRAM bytes: $unfused_dram_bytes_1"
      # echo "    DRAM AI: $unfused_kernel_dram_ai_1"
      # echo "    L1 bytes: $unfused_l1_bytes_1"
      # echo "    L1 AI: $unfused_kernel_l1_ai_1"

      unfused_kernel_gflops_2="$( echo "scale=4; $unfused_kernel_flop_2 / $unfused_kernel_runtime_2 * 1000000 / 1000000000" | bc )"
      unfused_kernel_l1_ai_2="$( echo "scale=4; $unfused_kernel_flop_2 / $unfused_l1_bytes_2" | bc )"
      unfused_kernel_dram_ai_2="$( echo "scale=4; $unfused_kernel_flop_2 / $unfused_dram_bytes_2" | bc )"
      # echo "------ Unfused kernel 2 details"
      # echo "    Runtime: $unfused_kernel_runtime_2"
      # echo "    Theoretical Flop: $unfused_kernel_flop_2_th"
      # echo "    Flop: $unfused_kernel_flop_2"
      # echo "    GFLOPS: $unfused_kernel_gflops_2"
      # echo "    Theoretical DRAM bytes: $unfused_kernel_dram_bytes_2_th"
      # echo "    DRAM bytes: $unfused_dram_bytes_2"
      # echo "    DRAM AI: $unfused_kernel_dram_ai_2"
      # echo "    L1 bytes: $unfused_l1_bytes_2"
      # echo "    L1 AI: $unfused_kernel_l1_ai_2"

      cd ..
      mkdir -p logs/runtime/cpu/$(( ${REP_BENCH} * 2 ))
      echo -e "fused,unfused_1,unfused_2,mkldnn_1,mkldnn_2\n$fused_kernel_runtime,$unfused_kernel_runtime_1,$unfused_kernel_runtime_2,$mkldnn_runtime_1,$mkldnn_runtime_2" > "logs/runtime/cpu/$(( ${REP_BENCH} * 2 ))/${workload_name}.csv"
      mkdir -p logs/arithmetic_intensity/cpu/$(( ${REP_BENCH} * 2 ))
      echo -e "fused_dram,unfused_1_dram,unfused_2_dram,mkldnn_dram_1,mkldnn_dram_2,fused_l1,unfused_1_l1,unfused_2_l1,mkldnn_l1_1,mkldnn_l1_2\n${fused_kernel_dram_ai},${unfused_kernel_dram_ai_1},${unfused_kernel_dram_ai_2},${mkldnn_dram_ai_1},${mkldnn_dram_ai_2},${fused_kernel_l1_ai},${unfused_kernel_l1_ai_1},${unfused_kernel_l1_ai_2},${mkldnn_l1_ai_1},${mkldnn_l1_ai_2}" > "logs/arithmetic_intensity/cpu/$(( ${REP_BENCH} * 2 ))/${workload_name}.csv"
      mkdir -p logs/gflops/cpu/$(( ${REP_BENCH} * 2 ))
      echo -e "fused,unfused_1,unfused_2,mkldnn_1,mkldnn_2\n$fused_kernel_gflops,$unfused_kernel_gflops_1,$unfused_kernel_gflops_2,$mkldnn_gflops_1,$mkldnn_gflops_2" > "logs/gflops/cpu/$(( ${REP_BENCH} * 2 ))/${workload_name}.csv"
      cd scripts

      # Print results
      echo "|--------------------"
      echo "| Workload name: ${workload_name}"
      echo "|--------------------"
      echo "| Fused/Unfused/MKLDNN runtime: ${fused_kernel_runtime} us, ${unfused_total_runtime} us, ${mkldnn_total_runtime} us."
      echo "| Fused/Unfused_1/Unfused_2/MKLDNN_1/MKLDNN_2 DRAM AI: ${fused_kernel_dram_ai}, ${unfused_kernel_dram_ai_1}, ${unfused_kernel_dram_ai_2}, ${mkldnn_dram_ai_1}, ${mkldnn_dram_ai_2}."
      echo "| Fused/Unfused_1/Unfused_2/MKLDNN_1/MKLDNN_2 L1 AI: ${fused_kernel_l1_ai}, ${unfused_kernel_l1_ai_1}, ${unfused_kernel_l1_ai_2}, ${mkldnn_l1_ai_1}, ${mkldnn_l1_ai_2}."
      echo "| Fused/Unfused_1/Unfused_2/MKLDNN_1/MKLDNN_2 GFLOPS: ${fused_kernel_gflops}, ${unfused_kernel_gflops_1}, ${unfused_kernel_gflops_2}, ${mkldnn_gflops_1}, ${mkldnn_gflops_2}."
      echo "|--------------------"
    fi
  done < "$input"
done