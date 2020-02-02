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
      total_dram_read="$(cat /tmp/bw_mkldnn.txt | grep -e 'Total read transactions' | awk '{ printf "%s\n", $5 }')"
      total_dram_write="$(cat /tmp/bw_mkldnn.txt | grep -e 'Total write transactions' | awk '{ printf "%s\n", $5 }')"
      # echo $total_dram_read
      # echo $total_dram_write
      mkldnn_dram_ai="$( echo "scale=4; ($mkldnn_total_flop / (($total_dram_read + $total_dram_write) * 64))" | bc)"

      cd ..
      mkdir -p logs/runtime/cpu
      echo -e "generated,mkldnn\n0.0,$mkldnn_runtime" > "logs/runtime/cpu/${workload_name}.csv"
      mkdir -p logs/arithmetic_intensity/cpu
      echo -e "generated_dram,mkldnn_dram\n0.0,$mkldnn_dram_ai" > "logs/arithmetic_intensity/cpu/${workload_name}.csv"
      mkdir -p logs/gflops/cpu
      echo -e "generated,mkldnn\n0.0,$mkldnn_gflops" > "logs/gflops/cpu/${workload_name}.csv"
      cd scripts

      # Print results
      echo "###################"
      echo "Workload name: ${workload_name}"
      echo "Generated/mkldnn runtime: 0.0 us, ${mkldnn_runtime} us."
      echo "Generated/mkldnn DRAM AI: 0.0, ${mkldnn_dram_ai}."
      echo "Generated/mkldnn GFLOPS: 0.0, ${mkldnn_gflops}."

    fi
  done < "$input"
done