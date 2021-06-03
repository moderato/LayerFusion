# LayerFusion

To tune fused kernels:
```bash
# Terminal 1: 

# Terminal 2:

# Terminal 3:
python fused_schedule_test.py -at 3000 # -u for unfused kernels
```

To verify and generate code for fused kernels:
```bash
python fused_schedule_test.py -akncd # -u for unfused kernels
```

To tune graph for CPU
```bash
# Valid CPU device_name: i7_7700K, Xeon_GCP, EPYC
# Valid model name: mobilenet_v1, mobilenet_v2, mnasnet_a1, resnet_18, resnet_50
python model_test.py -k -v <device_name> -w <model_name> # Have kernel tuning logs ready and tune graph for models with fused ops
python model_test.py -n -v <device_name> -w <model_name> # Tune kernels and graphs for models with all ops unfused
```