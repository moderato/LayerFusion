from fusion_composer import *
from helper import get_workloads
import os
from matplotlib import pyplot as plt

if __name__ == '__main__':
    workloads = get_workloads()
    print("L3: 8MB")
    print("L2: 256KB per core")
    print("L1: 32KB per core")

    footprints = []
    fused_times = []
    tvm_times = []
    mkldnn_times = []
    fused_flops = []
    tvm_flops = []
    mkldnn_flops = []
    keys = []

    for workload_type in workloads.keys():
        for w in workloads[workload_type].keys():
            p = workloads[workload_type][w]
            fc = FusionComposer(p, target='llvm')
            fused_mem_bytes_th = fc.get_theoretical_mem_bytes()
            mem_bytes_1_th, mem_bytes_2_th = fc.get_theoretical_mem_bytes_per_layer()
            flop_1_th, flop_2_th = fc.get_FLOP_per_layer()
            
            flop_file = 'logs/flop/cpu/20/{}.csv'.format(w)
            # if not os.path.isfile(flop_file):
            #     continue
            with open(flop_file, 'r') as f:
                lines = f.readlines()
            fused_flop_em = float(lines[1].split(',')[0])
            flop_1_em_tvm = float(lines[1].split(',')[1])
            flop_2_em_tvm = float(lines[1].split(',')[2])
            flop_1_em_mkldnn = float(lines[1].split(',')[3])
            flop_2_em_mkldnn = float(lines[1].split(',')[4])

            ai_file = 'logs/arithmetic_intensity/cpu/20/{}.csv'.format(w)
            with open(ai_file, 'r') as f:
                lines = f.readlines()
            fused_ai_dram_em = float(lines[1].split(',')[0])
            ai_1_dram_em_tvm = float(lines[1].split(',')[1])
            ai_2_dram_em_tvm = float(lines[1].split(',')[2])
            ai_1_dram_em_mkldnn = float(lines[1].split(',')[3])
            ai_2_dram_em_mkldnn = float(lines[1].split(',')[4])
            fused_ai_cache_em = float(lines[1].split(',')[5])
            ai_1_cache_em_tvm = float(lines[1].split(',')[6])
            ai_2_cache_em_tvm = float(lines[1].split(',')[7])
            ai_1_cache_em_mkldnn = float(lines[1].split(',')[8])
            ai_2_cache_em_mkldnn = float(lines[1].split(',')[9])

            fused_mem_bytes_dram_em = int(fused_flop_em / fused_ai_dram_em)
            mem_bytes_1_dram_em_tvm = int(flop_1_em_tvm / ai_1_dram_em_tvm)
            mem_bytes_2_dram_em_tvm = int(flop_2_em_tvm / ai_2_dram_em_tvm)
            mem_bytes_1_dram_em_mkldnn = int(flop_1_em_mkldnn / ai_1_dram_em_mkldnn)
            mem_bytes_2_dram_em_mkldnn = int(flop_2_em_mkldnn / ai_2_dram_em_mkldnn)
            fused_mem_bytes_cache_em = int(fused_flop_em / fused_ai_cache_em)
            mem_bytes_1_cache_em_tvm = int(flop_1_em_tvm / ai_1_cache_em_tvm)
            mem_bytes_2_cache_em_tvm = int(flop_2_em_tvm / ai_2_cache_em_tvm)
            mem_bytes_1_cache_em_mkldnn = int(flop_1_em_mkldnn / ai_1_cache_em_mkldnn)
            mem_bytes_2_cache_em_mkldnn = int(flop_2_em_mkldnn / ai_2_cache_em_mkldnn)

            time_file = 'logs/runtime/cpu/20/{}.csv'.format(w)
            with open(time_file, 'r') as f:
                lines = f.readlines()
            fused_time = float(lines[1].split(',')[0])
            layer_1_time_tvm = float(lines[1].split(',')[1])
            layer_2_time_tvm = float(lines[1].split(',')[2])
            layer_1_time_mkldnn = float(lines[1].split(',')[3])
            layer_2_time_mkldnn = float(lines[1].split(',')[4])

            print("--------", w)
            print("Time:              {:.2f},           {:.2f}, {:.2f},    {:.2f}, {:.2f}".format(fused_time, layer_1_time_tvm, layer_2_time_tvm, layer_1_time_mkldnn, layer_2_time_mkldnn))
            print("Footprint:         {:.2f} MB,         {:.2f} MB, {:.2f} MB".format((mem_bytes_1_th + mem_bytes_2_th) / 1024 / 1024, (mem_bytes_1_th) / 1024 / 1024, (mem_bytes_2_th) / 1024 / 1024))
            print("Theoretical AI:    {:.2f},            {:.2f}, {:.2f}".format((flop_1_th + flop_2_th) / (mem_bytes_1_th + mem_bytes_2_th), flop_1_th / mem_bytes_1_th, flop_2_th / mem_bytes_2_th))
            print("Empirical DRAM AI: {:.2f}".format(fused_ai_dram_em))
            print("Bandwidth:         {:.2f}".format(mem_bytes_1_th / layer_1_time_tvm / 1e3))
            print("fused em/th dram mem bytes", fused_mem_bytes_dram_em / fused_mem_bytes_th)
            print("tvm 1 em/th dram mem bytes", mem_bytes_1_dram_em_tvm / mem_bytes_1_th)
            print("tvm 2 em/th dram mem bytes", mem_bytes_2_dram_em_tvm / mem_bytes_2_th)
            print("mkldnn 1 em/th dram mem bytes", mem_bytes_1_dram_em_mkldnn / mem_bytes_1_th)
            print("mkldnn 2 em/th dram mem bytes", mem_bytes_2_dram_em_mkldnn / mem_bytes_2_th)
            print("fused em cache mem bytes", fused_mem_bytes_cache_em)
            print("tvm 1 em cache mem bytes", mem_bytes_1_cache_em_tvm)
            print("tvm 2 em cache mem bytes", mem_bytes_2_cache_em_tvm)
            print("mkldnn 1 em cache mem bytes", mem_bytes_1_cache_em_mkldnn)
            print("mkldnn 2 em cache mem bytes", mem_bytes_2_cache_em_mkldnn)
            print(flop_1_th, flop_2_th)

            footprints.append((mem_bytes_1_th + mem_bytes_2_th) / 1024 / 1024)
            keys.append(w)
            fused_times.append(fused_time)
            tvm_times.append(layer_1_time_tvm + layer_2_time_tvm)
            mkldnn_times.append(layer_1_time_mkldnn + layer_2_time_mkldnn)
            fused_flops.append(fused_flop_em / fused_time / 1000)
            tvm_flops.append((flop_1_em_tvm + flop_2_em_tvm) / (layer_1_time_tvm + layer_2_time_tvm) / 1000)
            mkldnn_flops.append((flop_1_em_mkldnn + flop_2_em_mkldnn) / (layer_1_time_mkldnn + layer_2_time_mkldnn) / 1000)

    cpus = {
        'names': ['i7_7700K', 'GCP'],
        'L3': [8, 24.75],
        'L2': [1, 4],
        'L1': [0.25, 0.25],
        'peaks': [511, 0]
    }
    fig, ax = plt.subplots()
    ax.bar(keys, footprints, color='lightskyblue')
    ax.set_ylabel('Footprint Size (MB)')
    ax.set_yscale('log')
    line_styles = ['--', ':']
    colors = ['red', 'black']

    lines = []
    for i in range(2):
        L1 = cpus['L1'][i]
        L2 = cpus['L2'][i]
        L3 = cpus['L3'][i]
        h0 = ax.axhline(L3, 0, 1, linestyle=line_styles[i], color=colors[i])
        _ = ax.axhline(L2, 0, 1, linestyle=line_styles[i], color=colors[i])
        _ = ax.axhline(L1, 0, 1, linestyle=line_styles[i], color=colors[i])
        lines.append(h0)
        ax.set_xticklabels(keys, rotation=45, fontsize=8)
    ax.legend(lines, cpus['names'])

    y_ticks = [min(cpus['L1'])]
    tmp = y_ticks[0]
    while tmp < max(cpus['L3']):
        tmp *= 2
        y_ticks.append(tmp)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(t) for t in y_ticks], fontsize=8)
    ax.set_ylim([y_ticks[0] * 0.9, y_ticks[-1] * 1.1])
    plt.tight_layout()
    plt.savefig('footprints.png', bbox_inches='tight')
    plt.savefig('footprints.eps', bbox_inches='tight')
    plt.savefig('footprints.pdf', bbox_inches='tight')

    x_axis = range(0, len(keys))
    width = 0.2

    fig, ax = plt.subplots()
    a = ax.bar([x-width for x in x_axis], fused_times, width=0.2)
    b = ax.bar(x_axis, tvm_times, width=0.2)
    c = ax.bar([x+width for x in x_axis], mkldnn_times, width=0.2)
    ax.legend([a, b, c], ['TVM fused', 'TVM separate', 'MKLDNN'])
    ax.set_xticks(x_axis)
    ax.set_xticklabels(keys, rotation=45, fontsize=12)
    fig.set_size_inches(12, 3.5)
    plt.tight_layout()
    plt.savefig('runtime.png', bbox_inches='tight')
    plt.savefig('runtime.eps', bbox_inches='tight')
    plt.savefig('runtime.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    a = ax.bar([x-width for x in x_axis], fused_flops, width=0.2)
    b = ax.bar(x_axis, tvm_flops, width=0.2)
    c = ax.bar([x+width for x in x_axis], mkldnn_flops, width=0.2)
    ax.legend([a, b, c], ['TVM fused', 'TVM separate', 'MKLDNN'])
    ax.set_xticks(x_axis)
    ax.set_xticklabels(keys, rotation=45, fontsize=12)
    fig.set_size_inches(12, 3.5)
    plt.tight_layout()
    plt.savefig('flops.png', bbox_inches='tight')
    plt.savefig('flops.eps', bbox_inches='tight')
    plt.savefig('flops.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    a = ax.bar([x-width for x in x_axis], [x / y for x, y in zip(fused_flops, mkldnn_flops)], width=0.2)
    b = ax.bar(x_axis, [x / y for x, y in zip(tvm_flops, mkldnn_flops)], width=0.2)
    c = ax.bar([x+width for x in x_axis], [1 for x in mkldnn_flops], width=0.2)
    ax.set_ylim([0.2, 1.5])
    ax.legend([a, b, c], ['TVM fused', 'TVM separate', 'MKLDNN'], fontsize=11)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(keys, rotation=45, fontsize=12)
    fig.set_size_inches(12, 3.5)
    plt.tight_layout()
    plt.savefig('flops_normalized.png', bbox_inches='tight')
    plt.savefig('flops_normalized.eps', bbox_inches='tight')
    plt.savefig('flops_normalized.pdf', bbox_inches='tight')