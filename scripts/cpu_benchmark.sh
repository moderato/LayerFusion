#!/bin/bash
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
      # echo ${layer_1_desc}
      # echo ${layer_2_desc}

      #### Generated kernels ####
      cd ../benchmark/cpu

      # Repeatition=1000 for runtime measurement
      make KERNEL=../../generated_kernels/cpu/${workload_name}.asm >& /dev/null
      ./cpu_bench "$line" 0 &> /tmp/runtime_generated.txt
      generated_kernel_runtime="$(cat /tmp/runtime_generated.txt | grep Fusion | awk '{ printf  "%10s\n", $4 }')"

      # Repeatition=20 for flops measurement
      make KERNEL=../../generated_kernels/cpu/${workload_name}.asm REPEATITION=20 >& /dev/null
      rm -rf /tmp/sde_generated.out*
      sde -knl -d -iform 1 -omix /tmp/sde_generated.out -i -global_region -start_ssc_mark 111:repeat -stop_ssc_mark 222:repeat -- ./cpu_bench "$line" 0 >& /dev/null
      
      cd ../../scripts
      ./parse_sde.sh /tmp/sde_generated.out* > /tmp/sde_generated.txt
      generated_kernel_total_flop="$(cat /tmp/sde_generated.txt | grep -e 'Total single-precision FLOPs' | awk '{ printf "%s\n", $5 }')"
      total_dram_bytes="$(cat /tmp/sde_generated.txt | grep -e 'Total Bytes =' | awk '{ printf "%s\n", $4 }')"
      generated_kernel_gflops="$( echo "scale=4; $generated_kernel_total_flop / $generated_kernel_runtime * 1000000 / 1000000000" | bc)"
      generated_kernel_dram_ai="$( echo "scale=4; $generated_kernel_total_flop / $total_dram_bytes" | bc)"
      # echo $generated_kernel_runtime
      # echo $generated_kernel_total_flop
      # echo $generated_kernel_gflops
      # echo $generated_kernel_dram_ai

      #### MKLDNN ####
      # Runtime and flops measurement
      benchdnn --conv --mode=p -v0 --mb=1 --dir=FWD_I --cfg=f32 ${layer_1_desc} ${layer_2_desc} &> /tmp/runtime_mkldnn.txt
      mkldnn_total_flop=0
      mkldnn_runtime=0
      while IFS= read -r l
      do
        if [ "${l:0:8}" == "perf,cpu" ];
        then
          result="$(echo $l | awk '{ printf "%s\n", $3 }')"
          IFS=',' read -ra PARAM <<< $result
          mkldnn_total_flop="$( echo "scale=4; $mkldnn_total_flop + ${PARAM[1]} * 1000000000" | bc)"
          mkldnn_runtime="$( echo "scale=4; $mkldnn_runtime + ${PARAM[5]} * 1000.0" | bc)" # In microseconds
        fi
      done < "/tmp/runtime_mkldnn.txt"
      mkldnn_gflops="$( echo "scale=4; $mkldnn_total_flop / $mkldnn_runtime * 1000000 / 1000000000" | bc)"
      # echo $mkldnn_total_flop
      # echo $mkldnn_runtime
      # echo $mkldnn_gflops

      # BW measurement
      # Relu: --attr="post_ops='relu'"
      rm -rf /tmp/vtune_report
      vtune -r /tmp/vtune_report -q -collect memory-access -finalization-mode=none -data-limit=0 benchdnn --conv --mode=PC -v1 --mb=1 --dir=FWD_I --cfg=f32 --alg=AUTO ${layer_1_desc} ${layer_2_desc} &> /dev/null
      vtune -report hw-events -report-knob show-issues=false -r /tmp/vtune_report -q -group-by=package -format=csv -column=UNC_IMC_DRAM_DATA_READS,UNC_IMC_DRAM_DATA_WRITES -csv-delimiter=comma > /tmp/cpu_bench_report.summary
      ./parse_vtune.sh -v /tmp/cpu_bench_report.summary > /tmp/bw_mkldnn.txt
      total_dram_transactions="$(cat /tmp/bw_mkldnn.txt | grep -e 'Total transactions' | awk '{ printf "%s\n", $4 }')"
      # echo $total_dram_transactions
      mkldnn_dram_ai="$( echo "scale=4; ($mkldnn_total_flop / ($total_dram_transactions * 64))" | bc)"

      cd ..
      mkdir -p logs/runtime/cpu
      echo -e "generated,mkldnn\n$generated_kernel_runtime,$mkldnn_runtime" > "logs/runtime/cpu/${workload_name}.csv"
      mkdir -p logs/arithmetic_intensity/cpu
      echo -e "generated_dram,mkldnn_dram\n$generated_kernel_dram_ai,$mkldnn_dram_ai" > "logs/arithmetic_intensity/cpu/${workload_name}.csv"
      mkdir -p logs/gflops/cpu
      echo -e "generated,mkldnn\n$generated_kernel_gfloops,$mkldnn_gflops" > "logs/gflops/cpu/${workload_name}.csv"
      cd scripts

      # Print results
      echo "###################"
      echo "Workload name: ${workload_name}"
      echo "Generated/mkldnn runtime: ${generated_kernel_runtime} us, ${mkldnn_runtime} us."
      echo "Generated/mkldnn DRAM AI: ${generated_kernel_dram_ai}, ${mkldnn_dram_ai}."
      echo "Generated/mkldnn GFLOPS: ${generated_kernel_gflops}, ${mkldnn_gflops}."

    fi
  done < "$input"
done