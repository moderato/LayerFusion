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
      # echo ${workload_name}
      echo ${layer_1_desc}
      echo ${layer_2_desc}

      #### Generated kernels ####
      cd ../benchmark/cpu

      # Repeatition=1000 for runtime measurement
      make KERNEL=../../generated_kernels/cpu/${workload_name}.asm >& /dev/null
      numactl -l -C 0-3 ./cpu_bench "$line" 0 &> /tmp/runtime_generated.txt
      generated_kernel_runtime="$(cat /tmp/runtime_generated.txt | grep Fusion | awk '{ printf  "%10s\n", $4 }')"

      # Repeatition=20 for flops and memory transaction measurement
      make KERNEL=../../generated_kernels/cpu/${workload_name}.asm REPEATITION=${REP_BENCH} >& /dev/null
      rm -rf /tmp/sde_generated.out*
      sde -knl -d -iform 1 -omix /tmp/sde_generated.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- numactl -l -C 0-3 ./cpu_bench "$line" 0 >& /dev/null
      rm -rf /tmp/vtune_report
      vtune -r /tmp/vtune_report -q -collect memory-access -finalization-mode=none -data-limit=0 -- numactl -l -C 0-3 ./cpu_bench "$line" 0 >& /dev/null
      vtune -report hw-events -report-knob show-issues=false -r /tmp/vtune_report -q -group-by=package -format=csv -column=UNC_IMC_DRAM_DATA_READS,UNC_IMC_DRAM_DATA_WRITES -csv-delimiter=comma > /tmp/cpu_bench_report.summary
      
      # Parse the above results
      cd ../../scripts

      ./parse_sde.sh /tmp/sde_generated.out* > /tmp/sde_generated.txt
      generated_kernel_flop="$(cat /tmp/sde_generated.txt | grep -e 'Total single-precision FLOPs' | awk '{ printf "%s\n", $5 }')"
      generated_kernel_flop="$( echo "scale=4; $generated_kernel_flop / (${REP_BENCH} * 2)" | bc )"
      generated_l1_bytes="$(cat /tmp/sde_generated.txt | grep -e 'Total Bytes =' | awk '{ printf "%s\n", $4 }')"
      generated_l1_bytes="$( echo "scale=4; $generated_l1_bytes / (${REP_BENCH} * 2)" | bc )" # Per iteration
      ./parse_vtune.sh -v /tmp/cpu_bench_report.summary > /tmp/bw_generated.txt
      generated_dram_transactions="$(cat /tmp/bw_generated.txt | grep -e 'Total transactions' | awk '{ printf "%s\n", $4 }')"
      generated_dram_bytes="$( echo "scale=4; $generated_dram_transactions * 64 / (${REP_BENCH} * 2)" | bc )" # Per iteration

      # GFLOPS and AI
      generated_kernel_gflops="$( echo "scale=4; $generated_kernel_flop / $generated_kernel_runtime * 1000000 / 1000000000" | bc )"
      generated_kernel_l1_ai="$( echo "scale=4; $generated_kernel_flop / $generated_l1_bytes" | bc )"
      generated_kernel_dram_ai="$( echo "scale=4; $generated_kernel_flop / $generated_dram_bytes" | bc )"
      echo "------ Generated kernel details"
      echo "    Runtime: $generated_kernel_runtime"
      echo "    Flop: $generated_kernel_flop"
      echo "    GFLOPS: $generated_kernel_gflops"
      echo "    DRAM bytes: $generated_dram_bytes"
      echo "    DRAM AI: $generated_kernel_dram_ai"
      echo "    L1 bytes: $generated_l1_bytes"
      echo "    L1 AI: $generated_kernel_l1_ai"

      #### MKLDNN ####
      # Runtime and flops measurement
      numactl -l -C 0-3 benchdnn --fix-times-per-prb=1000 --conv --mode=p -v0 --mb=1 --dir=FWD_I --cfg=f32 ${layer_1_desc} ${layer_2_desc} &> /tmp/runtime_mkldnn.txt
      mkldnn_flop=0
      mkldnn_runtime=0
      while IFS= read -r l
      do
        if [ "${l:0:8}" == "perf,cpu" ];
        then
          result="$(echo $l | awk '{ printf "%s\n", $3 }')"
          IFS=',' read -ra PARAM <<< $result
          mkldnn_flop="$( echo "scale=4; $mkldnn_flop + ${PARAM[1]} * 1000000000" | bc)"
          mkldnn_runtime="$( echo "scale=4; $mkldnn_runtime + ${PARAM[5]} * 1000.0" | bc)" # In microseconds
        fi
      done < "/tmp/runtime_mkldnn.txt"

      # BW measurement
      # Relu: --attr="post_ops='relu'"
      rm -rf /tmp/sde_library.out*
      sde -knl -d -iform 1 -omix /tmp/sde_library.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- numactl -l -C 0-3 benchdnn --fix-times-per-prb="$(( ${REP_BENCH} * 2 ))" --conv --mode=p -v0 --mb=1 --dir=FWD_I --cfg=f32 ${layer_1_desc} ${layer_2_desc} >& /dev/null
      rm -rf /tmp/vtune_report
      vtune -r /tmp/vtune_report -q -collect memory-access -finalization-mode=none -data-limit=0 -- numactl -l -C 0-3 benchdnn --fix-times-per-prb="$(( ${REP_BENCH} * 2 ))" --conv --mode=PC -v1 --mb=1 --dir=FWD_I --cfg=f32 --alg=AUTO ${layer_1_desc} ${layer_2_desc} &> /dev/null
      vtune -report hw-events -report-knob show-issues=false -r /tmp/vtune_report -q -group-by=package -format=csv -column=UNC_IMC_DRAM_DATA_READS,UNC_IMC_DRAM_DATA_WRITES -csv-delimiter=comma > /tmp/cpu_bench_report.summary

      # Parse the above results
      ./parse_sde.sh /tmp/sde_library.out* > /tmp/bw_mkldnn.txt
      library_l1_bytes="$(cat /tmp/bw_mkldnn.txt | grep -e 'Total Bytes =' | awk '{ printf "%s\n", $4 }')"
      library_l1_bytes="$( echo "scale=4; $library_l1_bytes / (${REP_BENCH} * 2) " | bc )" # Per iteration
      ./parse_vtune.sh -v /tmp/cpu_bench_report.summary > /tmp/bw_mkldnn.txt
      library_dram_transactions="$(cat /tmp/bw_mkldnn.txt | grep -e 'Total transactions' | awk '{ printf "%s\n", $4 }')"
      library_dram_bytes="$( echo "scale=4; $library_dram_transactions * 64 / (${REP_BENCH} * 2)" | bc )" # Per iteration

      # GFLOPS and AI
      mkldnn_gflops="$( echo "scale=4; $mkldnn_flop / $mkldnn_runtime * 1000000 / 1000000000" | bc )"
      mkldnn_l1_ai="$( echo "scale=4; $mkldnn_flop / $library_l1_bytes" | bc )"
      mkldnn_dram_ai="$( echo "scale=4; $mkldnn_flop / $library_dram_bytes" | bc )"
      echo "--- MKLDNN details"
      echo "    Runtime: $mkldnn_runtime"
      echo "    Flop: $mkldnn_flop"
      echo "    GFLOPS: $mkldnn_gflops"
      echo "    DRAM bytes: $library_dram_bytes"
      echo "    DRAM AI: $mkldnn_dram_ai"
      echo "    L1 bytes: $library_l1_bytes"
      echo "    L1 AI: $mkldnn_l1_ai"

      cd ..
      mkdir -p logs/runtime/cpu/$(( ${REP_BENCH} * 2 ))
      echo -e "generated,mkldnn\n$generated_kernel_runtime,$mkldnn_runtime" > "logs/runtime/cpu/$(( ${REP_BENCH} * 2 ))/${workload_name}.csv"
      mkdir -p logs/arithmetic_intensity/cpu/$(( ${REP_BENCH} * 2 ))
      echo -e "generated_dram,mkldnn_dram,generated_l1,mkldnn_l1\n$generated_kernel_dram_ai,$mkldnn_dram_ai,${generated_kernel_l1_ai},${mkldnn_l1_ai}" > "logs/arithmetic_intensity/cpu/$(( ${REP_BENCH} * 2 ))/${workload_name}.csv"
      mkdir -p logs/gflops/cpu/$(( ${REP_BENCH} * 2 ))
      echo -e "generated,mkldnn\n$generated_kernel_gflops,$mkldnn_gflops" > "logs/gflops/cpu/$(( ${REP_BENCH} * 2 ))/${workload_name}.csv"
      cd scripts

      # Print results
      echo "|--------------------"
      echo "| Workload name: ${workload_name}"
      echo "|--------------------"
      echo "| Generated/mkldnn runtime: ${generated_kernel_runtime} us, ${mkldnn_runtime} us."
      echo "| Generated/mkldnn DRAM AI: ${generated_kernel_dram_ai}, ${mkldnn_dram_ai}."
      echo "| Generated/mkldnn L1 AI: ${generated_kernel_l1_ai}, ${mkldnn_l1_ai}."
      echo "| Generated/mkldnn GFLOPS: ${generated_kernel_gflops}, ${mkldnn_gflops}."
      echo "|--------------------"
    fi
  done < "$input"
done