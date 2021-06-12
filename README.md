# LayerFusion
To initialize:
```bash
source ./init_vars.sh
```

To tune fused kernels:
```bash
# Terminal 1:
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190

# Terminal 2:
python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=<device_name>

# Terminal 3:
# Valid CPU device_name: i7_7700K, Xeon_GCP, EPYC, Xeon_E5
python fused_schedule_test.py -at 4000 <device_name> # -u for unfused kernels
```

To verify and generate code and ref data for fused kernels:
```bash
python fused_schedule_test.py -akncd <device_name> # -u for unfused kernels
```

To tune graph for CPU:
```bash
# Currently NOT support extracting FUSED tasks from model graphs. Solution: tune the unfused graph and fused kernels, and merge the tuning logs.
# Currently supported CPU device_name: i7_7700K, Xeon_GCP, EPYC, Xeon_E5
# Currently supported model_name: mobilenet_v1, mobilenet_v2, mnasnet_a1, resnet_18, resnet_50
python model_test.py -n -v <device_name> -w <model_name> # Tune kernels and graphs for models with all ops unfused

# Duplicate logs for both the cases of w/ and w/o post ops
python duplicate_logs.py

# Assuming fused kernels are tuned.
python -m tvm.autotvm.record --mode pick_batch --batch_size 200 --append --i logs/autotvm/layer/cpu/fused/ --o logs/autotvm/model/cpu/<model_name>/nchwc_fused.log
cat logs/autotvm/model/cpu/<model_name>/nchwc_unfused.log >> logs/autotvm/model/cpu/<model_name>/nchwc_fused.log

# Have kernel tuning logs ready and tune graph for models with fused ops
python model_test.py -k -v <device_name> -w <model_name>

# Inference only
python model_test.py -kpd -v <device_name> -w <model_name> # fused
python model_test.py -kpnd -v <device_name> -w <model_name> # unfused
```

To benchmark and plot rooflines:
```bash
# See the Dockerfile for all dependencies needed to run the benchmark
cd scripts
./cpu_benchmark.sh

python plots/plot_roofline.py
```
