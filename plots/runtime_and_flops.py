from fusion_composer import *
from helper import get_workloads
from matplotlib import pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-i", "--iterations", type=int, default=20, help="Number of iterations for roofline collection")
    args = parser.parse_args()

    workloads = get_workloads()
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

            flop_file = 'logs/flop/cpu/{}/{}.csv'.format(args.iterations, w)
            with open(flop_file, 'r') as f:
                lines = f.readlines()
            fused_flop_em = float(lines[1].split(',')[0])
            flop_1_em_tvm = float(lines[1].split(',')[1])
            flop_2_em_tvm = float(lines[1].split(',')[2])
            flop_1_em_mkldnn = float(lines[1].split(',')[3])
            flop_2_em_mkldnn = float(lines[1].split(',')[4])

            ai_file = 'logs/arithmetic_intensity/cpu/{}/{}.csv'.format(args.iterations, w)
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

            time_file = 'logs/runtime/cpu/{}/{}.csv'.format(args.iterations, w)
            with open(time_file, 'r') as f:
                lines = f.readlines()
            fused_time = float(lines[1].split(',')[0])
            layer_1_time_tvm = float(lines[1].split(',')[1])
            layer_2_time_tvm = float(lines[1].split(',')[2])
            layer_1_time_mkldnn = float(lines[1].split(',')[3])
            layer_2_time_mkldnn = float(lines[1].split(',')[4])

            # print("--------", w)
            # print("Time:              {:.2f},           {:.2f}, {:.2f},    {:.2f}, {:.2f}".format(fused_time, layer_1_time_tvm, layer_2_time_tvm, layer_1_time_mkldnn, layer_2_time_mkldnn))
            # print("Footprint:         {:.2f} MB,         {:.2f} MB, {:.2f} MB".format((mem_bytes_1_th + mem_bytes_2_th) / 1024 / 1024, (mem_bytes_1_th) / 1024 / 1024, (mem_bytes_2_th) / 1024 / 1024))
            # print("Theoretical AI:    {:.2f},            {:.2f}, {:.2f}".format((flop_1_th + flop_2_th) / (mem_bytes_1_th + mem_bytes_2_th), flop_1_th / mem_bytes_1_th, flop_2_th / mem_bytes_2_th))
            # print("Empirical DRAM AI: {:.2f}".format(fused_ai_dram_em))
            # print("Bandwidth:         {:.2f}".format(mem_bytes_1_th / layer_1_time_tvm / 1e3))
            # print("fused em/th dram mem bytes", fused_mem_bytes_dram_em / fused_mem_bytes_th)
            # print("tvm 1 em/th dram mem bytes", mem_bytes_1_dram_em_tvm / mem_bytes_1_th)
            # print("tvm 2 em/th dram mem bytes", mem_bytes_2_dram_em_tvm / mem_bytes_2_th)
            # print("mkldnn 1 em/th dram mem bytes", mem_bytes_1_dram_em_mkldnn / mem_bytes_1_th)
            # print("mkldnn 2 em/th dram mem bytes", mem_bytes_2_dram_em_mkldnn / mem_bytes_2_th)
            # print("fused em cache mem bytes", fused_mem_bytes_cache_em)
            # print("tvm 1 em cache mem bytes", mem_bytes_1_cache_em_tvm)
            # print("tvm 2 em cache mem bytes", mem_bytes_2_cache_em_tvm)
            # print("mkldnn 1 em cache mem bytes", mem_bytes_1_cache_em_mkldnn)
            # print("mkldnn 2 em cache mem bytes", mem_bytes_2_cache_em_mkldnn)
            # print(flop_1_th, flop_2_th)

            footprints.append((mem_bytes_1_th + mem_bytes_2_th) / 1024 / 1024)
            keys.append(w)
            fused_times.append(fused_time)
            tvm_times.append(layer_1_time_tvm + layer_2_time_tvm)
            mkldnn_times.append(layer_1_time_mkldnn + layer_2_time_mkldnn)
            fused_flops.append((flop_1_th + flop_2_th) / fused_time / 1000) # Use theoretical flop instead of measured one. Same below.
            tvm_flops.append((flop_1_th + flop_2_th) / (layer_1_time_tvm + layer_2_time_tvm) / 1000)
            mkldnn_flops.append((flop_1_th + flop_2_th) / (layer_1_time_mkldnn + layer_2_time_mkldnn) / 1000)

    cpus = {
        'names': ['i7_7700K', 'GCP Intel', 'GCP AMD'],
        'L3': [8, 24.75, 16],
        'L2': [1, 4, 2],
        'L1': [0.25, 0.25, 0.25],
        'peaks': [511, 710, 413]
    }
    fig, ax = plt.subplots()
    ax.bar(keys, footprints, color='lightskyblue')
    ax.set_ylabel('Footprint Size (MB)')
    ax.set_yscale('log')
    line_styles = ['--', ':', '-.']
    colors = ['red', 'black', 'green']

    lines = []
    for i in range(len(cpus['names'])):
        L1 = cpus['L1'][i]
        L2 = cpus['L2'][i]
        L3 = cpus['L3'][i]
        h0 = ax.axhline(L3, 0, 1, linestyle=line_styles[i], color=colors[i])
        ax.text(-1.5, L3*1.1, 'L3')
        _ = ax.axhline(L2, 0, 1, linestyle=line_styles[i], color=colors[i])
        ax.text(-1.5, L2*1.1, 'L2')
        _ = ax.axhline(L1, 0, 1, linestyle=line_styles[i], color=colors[i])
        if i == 0:
            ax.text(-1.5, L1*1.1, 'L1')
        lines.append(h0)
        ax.set_xticklabels(keys, rotation=45, fontsize=8)
    ax.legend(lines, cpus['names'], loc='upper right')
    fig.set_size_inches(12, 5)

    y_ticks = [min(cpus['L1'])]
    tmp = y_ticks[0]
    while tmp < max(cpus['L3']):
        tmp *= 2
        y_ticks.append(tmp)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(t) for t in y_ticks], fontsize=8)
    ax.set_ylim([y_ticks[0] * 0.9, y_ticks[-1] * 1.1])
    plt.tight_layout()
    plt.savefig('plots/footprints.pdf', bbox_inches='tight')

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
    plt.savefig('plots/runtime.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    a = ax.bar([x-width for x in x_axis], fused_flops, width=0.2)
    b = ax.bar(x_axis, tvm_flops, width=0.2)
    c = ax.bar([x+width for x in x_axis], mkldnn_flops, width=0.2)
    ax.legend([a, b, c], ['TVM fused', 'TVM separate', 'MKLDNN'])
    ax.set_xticks(x_axis)
    ax.set_xticklabels(keys, rotation=45, fontsize=12)
    fig.set_size_inches(12, 3.5)
    plt.tight_layout()
    plt.savefig('plots/flops.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    a = ax.bar([x-width for x in x_axis], [x / y for x, y in zip(fused_flops, mkldnn_flops)], width=0.2)
    b = ax.bar(x_axis, [x / y for x, y in zip(tvm_flops, mkldnn_flops)], width=0.2)
    c = ax.bar([x+width for x in x_axis], [1 for x in mkldnn_flops], width=0.2)
    ax.set_ylim([0.15, 1.5])
    ax.legend([a, b, c], ['TVM fused', 'TVM separate', 'MKLDNN'], fontsize=10.5, bbox_to_anchor=(0.855, 0.655))
    ax.set_xticks(x_axis)
    ax.set_xticklabels(keys, rotation=45, fontsize=12)
    yticks = ax.get_yticks()
    for y in yticks:
        ax.axhline(y, 0, 1, linestyle='--', linewidth=0.3, color='black')
    fig.set_size_inches(12, 3.5)
    plt.tight_layout()
    plt.savefig('plots/flops_normalized.pdf', bbox_inches='tight')

    speedup_tvm = 1
    for a in [x / y for x, y in zip(fused_flops, tvm_flops)]:
        speedup_tvm *= a
    speedup_mkldnn = 1
    for a in [x / y for x, y in zip(fused_flops, mkldnn_flops)]:
        speedup_mkldnn *= a
    num_workloads = len([x / y for x, y in zip(fused_flops, mkldnn_flops)])
    print("Geo mean speedup over TVM: {:.3f}".format(speedup_tvm ** (1.0 / num_workloads)))
    print("Max over TVM: {:.3f}".format(max([x / y for x, y in zip(fused_flops, tvm_flops)])))
    print("Min over TVM: {:.3f}".format(min([x / y for x, y in zip(fused_flops, tvm_flops)])))
    print("Geo mean speedup over MKLDNN: {:.3f}".format(speedup_mkldnn ** (1.0 / num_workloads)))
    print("Max over MKLDNN: {:.3f}".format(max([x / y for x, y in zip(fused_flops, mkldnn_flops)])))
    print("Min over MKLDNN: {:.3f}".format(min([x / y for x, y in zip(fused_flops, mkldnn_flops)])))
