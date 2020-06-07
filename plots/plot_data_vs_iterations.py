import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os, json, math
font = {'size' : 15}
plt.rc('font', **font)

def flatten_list(lst):
    return sum(([x] if not isinstance(x, list) else flatten_list(x) for x in lst), [])

markersize = 5
colors = ['b','g','r','y','m','c']
styles = ['o','s','v','^','D','>','<','*','h','H','+','1','2','3','4','8','p','d','|','_','.',',']
devices = ['cpu']
workload_types = ['depth_conv']
iterations = [2, 4, 10, 20, 40, 100, 200, 400, 1000, 2000, 4000, 10000, 20000]
device_empirical = {
    'cpu': '/home/zhongyilin/Documents/experimental/cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0/Results.i7-7700K/Run.001/roofline.json',
    'gpu': '/home/zhongyilin/Documents/experimental/cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0/Results.1050Ti-cuda-fp32.01/Run.001/roofline.json'
}
device_name = {
    'cpu': 'i7_7700K',
    'gpu': '1050Ti'
}
device_theoretical = {
    '1050Ti': {
        'FP32 GFLOPS': 2138.0,
        'DRAM': 112.1,
        'Cache': 1117
    },
    'i7_7700K': {
        'FP32 GFLOPS': 537.6,
        'DRAM': 62.22,
        'Cache': 500
    }
}
HOME = os.getenv('LF_HOME')

for device in devices:
    data = {}
    data_type = ['AI_dram_generated', 'AI_L2_generated', 'FLOPS_generated', 'AI_dram_library', 'AI_L2_library', 'FLOPS_library']
    for t in data_type:
        data[t] = {}
    labels = []
    workloads = []

    has_L2 = False

    for w in workload_types:
        filename = '{}_plot'.format(device_name[device])
        workload_filename = '{}/workloads/{}_workloads.csv'.format(HOME, w)
        with open(workload_filename , 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]): # skip header
                splitted = line.strip().split(',')
                workload_name = splitted[0]
                workloads.append(workload_name)

                data['AI_dram_generated'][workload_name] = []
                data['AI_dram_library'][workload_name] = []
                data['AI_L2_generated'][workload_name] = []
                data['AI_L2_library'][workload_name] = []
                data['FLOPS_generated'][workload_name] = []
                data['FLOPS_library'][workload_name] = []

        for iter in iterations:
            for workload_name in workloads:
                # Read AI
                with open('{}/logs/arithmetic_intensity/{}/{}/{}.csv'.format(HOME, device, iter, workload_name), 'r') as ff:
                    lls = ff.readlines()
                    for l in lls[1:]: # skip header
                        splitted = l.strip().split(',')
                        data['AI_dram_generated'][workload_name].append(float(splitted[0]))
                        data['AI_dram_library'][workload_name].append(float(splitted[1]))
                        if len(splitted) == 4:
                            has_L2 = True
                            data['AI_L2_generated'][workload_name].append(float(splitted[2]))
                            data['AI_L2_library'][workload_name].append(float(splitted[3]))

                # Read FLOPS
                with open('{}/logs/gflops/{}/{}/{}.csv'.format(HOME, device, iter, workload_name), 'r') as ff:
                    lls = ff.readlines()
                    for l in lls[1:]: # skip header
                        splitted = l.strip().split(',')
                        data['FLOPS_generated'][workload_name].append(float(splitted[0]))
                        data['FLOPS_library'][workload_name].append(float(splitted[1]))

            # print(data['AI_dram_generated'], data['AI_dram_library'], data['AI_L2_generated'], data['AI_L2_library'], \
            #         data['FLOPS_generated'], data['FLOPS_library'], labels)

        fig = plt.figure(1, figsize=(22, 14))
        plt.clf()
        ylabels = ['AI_dram',  'AI_cache', 'FLOPS', 'AI_dram', 'AI_cache', 'FLOPS']

        for i in range(0, 6):
            ax = plt.subplot(2, 3, i+1)
            ax.set_xscale('log')
            ax.set_xlabel('Iterations')
            ax.set_ylabel(ylabels[i])
            keys = data[data_type[i]].keys()
            for key in keys:
                ax.plot(iterations, data[data_type[i]][key])
            ax.set_title(data_type[i])

        plt.savefig('{}_{}.png'.format(device, w), bbox_inches="tight")