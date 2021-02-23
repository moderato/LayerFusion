import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import chain
import sys, os, json, math
font = {'size': 20}
plt.rc('font', **font)

def flatten_list(lst):
    return sum(
        ([x] if not isinstance(x, list) else flatten_list(x) for x in lst), 
        []
    )

def get_FLOP(parameters):
    N, H, W, C = parameters[:4]
    start = 4
    FLOP = 0
    while 1:
        if start >= len(parameters) - 1:
            break
        f = parameters[start]
        oc = parameters[start+1]
        s = parameters[start+2]
        is_depthwise = parameters[start+3]
        post_ops = parameters[start+4]

        OH, OW = H / s, W / s
        OC = oc if not is_depthwise else C * oc
        FLOP += 2 * N * OH * OW * OC * f * f * (oc if is_depthwise else C)

        if post_ops:
            FLOP += N * OH * OW * OC

        N, H, W, C = N, OH, OW, OC
        start += 5
    return FLOP

def get_theoretical_mem_bytes(parameters):
    N, H, W, C = parameters[:4]
    start = 4
    layer = 0
    mem = 4 * (N * H * W * C)
    while 1:
        if start >= len(parameters) - 1:
            break
        f = parameters[start]
        oc = parameters[start+1]
        s = parameters[start+2]
        is_depthwise = parameters[start+3]
        post_ops = parameters[start+4]

        OH, OW = H / s, W / s
        OC = oc if not is_depthwise else C * oc
        mem += 4 * (f * f * C * oc)
        if post_ops:
            mem += 4 * OC
        N, H, W, C = N, OH, OW, OC
        start += 5
        layer += 1

    mem += 4 * (N * H * W * C)
    return mem


markersize = 12
num_layers = 2
colors = ['b','g','r','y','m','c']
styles = ['o','s','v','^','D','>','<','*','h','H','+','1','2','3','4','8','p','d','|','_','.',',']
devices = ['cpu']
workload_types = ['depth_conv']
iterations = [20]
device_empirical = {
    'cpu': '/home/zhongyilin/Documents/experimental/cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0/Results.i7-7700K/Run.001/roofline.json',
    'gpu': '/home/zhongyilin/Documents/experimental/cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0/Results.1050Ti-cuda-fp32.01/Run.001/roofline.json'
}
device_name = {
    'cpu': 'i7_7700K',
    'gpu': '1050Ti'
}
library_name = {
    'cpu': 'MKLDNN',
    'gpu': 'cuDNN'
}
device_theoretical = {
    '1050Ti': {
        'FP32 GFLOPS': 2138.0,
        'DRAM': 112.1,
        'Cache': 1117
    },
    'i7_7700K': {
        'FP32 GFLOPS': 537.6,
        'DRAM': 34.7,
        'Cache': 200
    }
}
arrowprops = dict(arrowstyle="-|>,head_width=0.15,head_length=0.55", linewidth="0.001", shrinkA=0, shrinkB=0, fc="k", ec="k")

HOME = os.getenv('LF_HOME')

merge = True
for device in devices:
    for iter in iterations:
        filename = '{}_{}_plot'.format(device_name[device], iter)

        AI = []
        FLOPS = []
        times = []

        labels = []
        gflops_roof = []
        gflops_roof_name = []
        BW_roof = []
        BW_roof_name = []
        peaks_ai = []

        has_L2 = False
        read_empirical = False # Whether read empirical or theoretical roofs

        for w in workload_types:
            workload_filename = '{}/workloads/{}_workloads.csv'.format(HOME, w)
            with open(workload_filename , 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines[1:]): # skip header
                    splitted = line.strip().split(',')
                    workload_name = splitted[0]
                    # Read times
                    with open('{}/logs/runtime/{}/{}/{}.csv'.format(HOME, device, iter, workload_name), 'r') as ff:
                        lls = ff.readlines()
                        for l in lls[1:]: # skip header
                            sp = [float(s) for s in l.strip().split(',')]
                            times.append(sp)
                    # Read AI
                    with open('{}/logs/arithmetic_intensity/{}/{}/{}.csv'.format(HOME, device, iter, workload_name), 'r') as ff:
                        l = ff.readlines()[1] # skip header
                        sp = [float(s) for s in l.strip().split(',')]
                        AI.append(sp[:5])
                        if len(sp) == 10:
                            has_L2 = True
                            AI[-1].extend(sp[5:])
                    # Read FLOPS
                    with open('{}/logs/gflops/{}/{}/{}.csv'.format(HOME, device, iter, workload_name), 'r') as ff:
                        l = ff.readlines()[1] # skip header
                        sp = [float(s) for s in l.strip().split(',')]
                        FLOPS.append(sp)
                    # Labels
                    labels.append(workload_name)

                    ####### If plot fused AI and FLOPS for every workload
                    # Read FLOP
                    if merge:
                        with open('{}/logs/flop/{}/{}/{}.csv'.format(HOME, device, iter, workload_name), 'r') as ff:
                            l = ff.readlines()[1] # skip header
                            sp = [float(s) for s in l.strip().split(',')]
                            ai = AI[-1]
                            t = times[-1]
                            dram_mem = [f / a for a, f in zip(ai[:5], sp)] # mem = FLOP / AI
                            tmp_ai = [sp[0] / dram_mem[0], (sp[1] + sp[2]) / (dram_mem[1] + dram_mem[2]), (sp[3] + sp[4]) / (dram_mem[3] + dram_mem[4])] # AI = FLOP / mem
                            tmp_flops = [sp[0] / t[0] / 1e3, (sp[1] + sp[2]) / (t[1] + t[2]) / 1e3, (sp[3] + sp[4]) / (t[3] + t[4]) / 1e3] # flops = FLOP / time
                            if has_L2:
                                cache_mem = [f / a for a, f in zip(ai[5:], sp)]
                                tmp_ai.extend([sp[0] / cache_mem[4], (sp[1] + sp[2]) / (cache_mem[3] + cache_mem[4]), (sp[3] + sp[4]) / (cache_mem[3] + cache_mem[4])])
                            AI[-1] = tmp_ai
                            FLOPS[-1] = tmp_flops

                    parameters = []
                    for idx, s in enumerate(splitted[1:]):
                        if idx == 8 or idx == 13:
                            if s == '':
                                parameters.append(None)
                            else:
                                parameters.append(s)
                        else:
                            parameters.append(int(s))
                    flop = get_FLOP(parameters)
                    mem = get_theoretical_mem_bytes(parameters)
                    peaks_ai.append(flop * 1.0 / mem)

            if read_empirical:
                device_info_file_dir = device_empirical[device]
                with open(device_info_file_dir, 'r') as f:
                    device_info = json.load(f)
                    for l in device_info['empirical']['gbytes']['data']:
                        if l[0] == 'DRAM':
                            BW_roof_name.append('DRAM')
                            BW_roof.append(float(l[1]))
                        elif l[0] == 'L1':
                            BW_roof_name.append('L2') # For GPU, ERT recognizes L2 as L1
                            BW_roof.append(float(l[1]))
                    for l in device_info['empirical']['gflops']['data']:
                        if l[0] == 'FP32 GFLOPs':
                            gflops_roof_name.append('FP32')
                            gflops_roof.append(float(l[1]))
            else:
                device_info = device_theoretical[device_name[device]]
                if 'DRAM' in device_info.keys():
                    BW_roof_name.append('DRAM')
                    BW_roof.append(device_info['DRAM'])
                if 'Cache' in device_info.keys():
                    BW_roof_name.append('Cache')
                    BW_roof.append(device_info['Cache'])
                if 'FP32 GFLOPS' in device_info.keys():
                    gflops_roof_name.append('FP32')
                    gflops_roof.append(device_info['FP32 GFLOPS'])

            # print(AI, FLOPS, \
            #         labels, \
            #         gflops_roof, gflops_roof_name, \
            #         BW_roof, BW_roof_name)

            nx = 10000
            if device == "gpu":
                ymin = 70.0
                ymax = 3000.0
                xmin = 0.1
                xmax = 2.3
                text_distance_scale = 1.1
            else:
                ymin = min(flatten_list(FLOPS)) / 1.1 # 160.0
                ymax = max(gflops_roof) * 1.1 # 600.0
                xmin = math.log10(min(flatten_list(AI)) / 1.4) # -0.4
                xmax = math.log10(max(flatten_list(AI)) * 2.1) # 3.1
                text_distance_scale = 1.02
                print(ymin, ymax, xmin, xmax)

            label_count = len(labels)
            cols = 6
            rows = (label_count + cols - 1) // cols
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 4))

            for idx, label in enumerate(labels):
                ax = axes.flat[idx]
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(10**xmin, 10**xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_title(label, x=0.82, y=0.07)

                ixx = int(nx*0.02)
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                scomp_x_elbow = [] 
                scomp_ix_elbow = [] 
                smem_x_elbow = [] 
                smem_ix_elbow = [] 

                x = np.logspace(xmin, xmax, nx)
                for roof in gflops_roof:
                    for ix in range(1, nx):
                        BW_roof_max = max(BW_roof)
                        if BW_roof_max * x[ix] >= roof and BW_roof_max * x[ix-1] < roof:
                            scomp_x_elbow.append(x[ix-1])
                            scomp_ix_elbow.append(ix-1)
                            break

                for roof in BW_roof:
                    for ix in range(1, nx):
                        if (gflops_roof[0] <= roof * x[ix] and gflops_roof[0] > roof * x[ix-1]):
                            smem_x_elbow.append(x[ix-1])
                            smem_ix_elbow.append(ix-1)
                            break

                for i in range(0, len(gflops_roof)):
                    roof = gflops_roof[i]
                    y = np.ones(len(x)) * roof
                    ax.plot(x[scomp_ix_elbow[i]:], y[scomp_ix_elbow[i]:], c='k', ls='-', lw='2')

                for i in range(0, len(BW_roof)):
                    roof = BW_roof[i]
                    y = x * roof
                    ax.plot(x[:smem_ix_elbow[i]+1], y[:smem_ix_elbow[i]+1], c='k', ls='-', lw='2')

                marker_handles = list()

                # workload: marker styles
                # generated/library: colors
                if merge:
                    ax.plot(float(AI[idx][0]), float(FLOPS[idx][0]), c=colors[0], marker='*', linestyle='None', ms=markersize, label=label)
                    marker_handles.append(ax.plot([], [], c='gray', marker='*', linestyle='None', ms=markersize, label=label)[0])

                    ax.plot(float(AI[idx][1]), float(FLOPS[idx][1]), c=colors[1], marker='*', linestyle='None', ms=markersize, label=label)
                    marker_handles.append(ax.plot([], [], c='gray', marker='*', linestyle='None', ms=markersize, label=label)[0])

                    ax.plot(float(AI[idx][2]), float(FLOPS[idx][2]), c=colors[2], marker='*', linestyle='None', ms=markersize, label=label)
                    marker_handles.append(ax.plot([], [], c='gray', marker='*', linestyle='None', ms=markersize, label=label)[0])
                else:
                    ax.plot(float(AI[idx][0]), float(FLOPS[idx][0]), c=colors[0], marker='*', linestyle='None', ms=markersize, label=label)
                    marker_handles.append(ax.plot([], [], c='gray', marker='*', linestyle='None', ms=markersize, label=label)[0])

                    ax.plot(float(AI[idx][1]), float(FLOPS[idx][1]), c=colors[1], marker='*', linestyle='None', ms=markersize, label=label)
                    marker_handles.append(ax.plot([], [], c='gray', marker='*', linestyle='None', ms=markersize, label=label)[0])

                    ax.plot(float(AI[idx][2]), float(FLOPS[idx][2]), c=colors[1], marker='*', linestyle='None', ms=markersize, label=label)
                    marker_handles.append(ax.plot([], [], c='gray', marker='*', linestyle='None', ms=markersize, label=label)[0])

                    ax.plot(float(AI[idx][3]), float(FLOPS[idx][3]), c=colors[2], marker='*', linestyle='None', ms=markersize, label=label)
                    marker_handles.append(ax.plot([], [], c='gray', marker='*', linestyle='None', ms=markersize, label=label)[0])

                    ax.plot(float(AI[idx][4]), float(FLOPS[idx][4]), c=colors[2], marker='*', linestyle='None', ms=markersize, label=label)
                    marker_handles.append(ax.plot([], [], c='gray', marker='*', linestyle='None', ms=markersize, label=label)[0])

                
                # # Mark the roofline shift from unfused to fused
                # for i in range(0, len(AI['dram']['fused'])):
                #     x0 = float(AI['dram'][library_name[device]][i])
                #     y0 = float(FLOPS[library_name[device]][i])
                #     dx = (float(AI['dram']['fused'][i]) - float(AI['dram'][library_name[device]][i]))
                #     dy = (float(FLOPS['fused'][i]) - float(FLOPS[library_name[device]][i]))
                #     dx_dy = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
                #     scaled_dx = dx
                #     scaled_dy = dy
                #     ax.arrow(x0, y0, scaled_dx, scaled_dy, linestyle=(0, (1, 6)))
                #     ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0), arrowprops=arrowprops)
                #     ax.text(x0 * (0.98 if dx > 0 else 1.02), y0 * 0.98, labels[i], fontsize=8)

                if has_L2:
                    if merge:
                        # Merged
                        ax.plot(float(AI[idx][3]),float(FLOPS[idx][0]),c=colors[3],marker='^',linestyle='None',ms=markersize,label=label)
                        ax.plot(float(AI[idx][4]),float(FLOPS[idx][1]),c=colors[4],marker='^',linestyle='None',ms=markersize,label=label)
                        ax.plot(float(AI[idx][5]),float(FLOPS[idx][2]),c=colors[5],marker='^',linestyle='None',ms=markersize,label=label)
                    else:
                        # Separate
                        ax.plot(float(AI[idx][5]),float(FLOPS[idx][0]),c=colors[3],marker='^',linestyle='None',ms=markersize,label=label)
                        ax.plot(float(AI[idx][6]),float(FLOPS[idx][1]),c=colors[4],marker='^',linestyle='None',ms=markersize,label=label)
                        ax.plot(float(AI[idx][7]),float(FLOPS[idx][2]),c=colors[4],marker='^',linestyle='None',ms=markersize,label=label)
                        ax.plot(float(AI[idx][8]),float(FLOPS[idx][3]),c=colors[5],marker='^',linestyle='None',ms=markersize,label=label)
                        ax.plot(float(AI[idx][9]),float(FLOPS[idx][4]),c=colors[5],marker='^',linestyle='None',ms=markersize,label=label)

                    # # Mark the roofline shift from unfused to fused
                    # for i in range(0, len(AI['cache']['fused'])):
                    #     x0 = float(AI['cache'][library_name[device]][i])
                    #     y0 = float(FLOPS[library_name[device]][i])
                    #     dx = (float(AI['cache']['fused'][i]) - float(AI['cache'][library_name[device]][i]))
                    #     dy = (float(FLOPS['fused'][i]) - float(FLOPS[library_name[device]][i]))
                    #     dx_dy = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
                    #     scaled_dx = dx
                    #     scaled_dy = dy
                    #     ax.arrow(x0, y0, scaled_dx, scaled_dy, linestyle=(0, (1, 6)))
                    #     ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0), arrowprops=arrowprops)
                    #     # ax.text(x0 * (0.98 if dx > 0 else 1.02), y0 * 0.98, labels[i], fontsize=8)

                # Peak DRAM AI
                ax.axvline(x=peaks_ai[idx], ymin=0, ymax=ymax, linestyle='--')

                #### Plot roofline
                # for roof in gflops_roof:
                #     ax.text(x[-ixx],roof,
                #             gflops_roof_name[gflops_roof.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GFLOP/s',
                #             horizontalalignment='right',
                #             verticalalignment='bottom')

                for roof in BW_roof:
                    ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0]) 
                                                * fig.get_size_inches()[1]/fig.get_size_inches()[0])
                    if x[ixx]*roof >ymin:
                        pass
                        # ax.text(x[ixx],x[ixx]*roof*(1+0.25*np.sin(ang)**2),
                        #     BW_roof_name[BW_roof.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GB/s',
                        #     horizontalalignment='left',
                        #     verticalalignment='bottom',
                        #     rotation=180/np.pi*ang)
                    else:
                        ymin_ix_elbow=list()
                        ymin_x_elbow=list()
                        for ix in range(1,nx):
                            if (ymin <= roof * x[ix] and ymin > roof * x[ix-1]):
                                ymin_x_elbow.append(x[ix-1])
                                ymin_ix_elbow.append(ix-1)
                                break
                        # ax.text(x[ixx+ymin_ix_elbow[0]]*0.8,\
                        #     x[ixx+ymin_ix_elbow[0]]*roof*(1+0.05*np.sin(ang)**2),
                        #     BW_roof_name[BW_roof.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GB/s',
                        #     horizontalalignment='left',
                        #     verticalalignment='bottom',
                        #     rotation=180/np.pi*ang)

            for i in range(len(labels), cols * rows):
                fig.delaxes(axes.flat[i])

            # # Device name
            # fig.text(0.5, 0.9, device_name[device], ha='center')
            
            # Common axis labels
            fig.text(0.5, 0.01, 'Arithmetic Intensity [FLOPs/Byte]', ha='center', fontsize=24)
            fig.text(0.001, 0.5, 'Performance [GFLOP/sec]', va='center', rotation='vertical', fontsize=24)

            handles = list()
            if merge:
                src_name = ['TVM fused (DRAM)', 'TVM separate (DRAM)', 'MKLDNN (DRAM)', 'TVM fused (L2)', 'TVM separate (L2)', 'MKLDNN (L2)']
            else:
                src_name = ['TVM fused (DRAM)', 'TVM layer_0 (DRAM)', 'TVM layer_1 (DRAM)', 'Library layer_0 (DRAM)', 'Library layer_1 (DRAM)', 'TVM fused (L2)', 'TVM layer_0 (L2)', 'TVM layer_1 (L2)', 'Library layer_0 (L2)', 'Library layer_1 (L2)']
            j = 0
            for i in range(0, len(src_name)):
                handles.append(Line2D([], [], color=colors[j], marker='*' if i < len(src_name) // 2 else '^', linestyle='None', markersize=markersize))
                if merge:
                    j += 1
                else:
                    if not ((i % 5 == 1) or (i % 5 == 3)):
                        j += 1
            leg2 = fig.legend(handles, src_name, loc='lower right', bbox_to_anchor=(0.98, 0.01))
            plt.tight_layout()
            plt.savefig(filename + '.png', bbox_inches='tight')
            plt.savefig(filename + '.eps', bbox_inches='tight')
            plt.savefig(filename + '.pdf', bbox_inches='tight')
