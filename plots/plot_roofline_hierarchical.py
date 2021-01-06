import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import chain
import sys, os, json, math
font = {'size' : 15}
plt.rc('font', **font)

def flatten_list(lst):
    return sum(
        ([x] if not isinstance(x, list) else flatten_list(x) for x in lst), 
        []
    )

def flatten_dict_values(dictionary):
    return chain.from_iterable(dictionary.values())

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
        'DRAM': 62.22,
        'Cache': 500
    }
}
arrowprops = dict(arrowstyle="-|>,head_width=0.15,head_length=0.55", linewidth="0.001", shrinkA=0, shrinkB=0, fc="k", ec="k")

HOME = os.getenv('LF_HOME')

for device in devices:
    for iter in iterations:
        filename = '{}_{}_plot'.format(device_name[device], iter)

        AI = []
        FLOPS = []

        labels = []
        gflops_roof = []
        gflops_roof_name = []
        BW_roof = []
        BW_roof_name = []

        has_L2 = False
        read_empirical = False # Whether read empirical or theoretical roofs

        for w in workload_types:
            workload_filename = '{}/workloads/{}_workloads.csv'.format(HOME, w)
            with open(workload_filename , 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines[1:]): # skip header
                    splitted = line.strip().split(',')
                    workload_name = splitted[0]
                    # Read AI
                    with open('{}/logs/arithmetic_intensity/{}/{}/{}.csv'.format(HOME, device, iter, workload_name), 'r') as ff:
                        lls = ff.readlines()
                        for l in lls[1:]: # skip header
                            splitted = [float(s) for s in l.strip().split(',')]
                            AI.append(splitted[:5])
                            if len(splitted) == 10:
                                has_L2 = True
                                AI[-1].extend(splitted[5:])
                    # Read FLOPS
                    with open('{}/logs/gflops/{}/{}/{}.csv'.format(HOME, device, iter, workload_name), 'r') as ff:
                        lls = ff.readlines()
                        for l in lls[1:]: # skip header
                            splitted = [float(s) for s in l.strip().split(',')]
                            FLOPS.append(splitted)
                            
                    # Labels
                    labels.append(workload_name)

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

            print(AI, FLOPS, \
                    labels, \
                    gflops_roof, gflops_roof_name, \
                    BW_roof, BW_roof_name)

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
            cols = 5
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
                    ax.plot(float(AI[idx][5]),float(FLOPS[idx][0]),c=colors[3],marker='+',linestyle='None',ms=markersize,label=label)
                    ax.plot(float(AI[idx][6]),float(FLOPS[idx][1]),c=colors[4],marker='+',linestyle='None',ms=markersize,label=label)
                    ax.plot(float(AI[idx][7]),float(FLOPS[idx][2]),c=colors[4],marker='+',linestyle='None',ms=markersize,label=label)
                    ax.plot(float(AI[idx][8]),float(FLOPS[idx][3]),c=colors[5],marker='+',linestyle='None',ms=markersize,label=label)
                    ax.plot(float(AI[idx][9]),float(FLOPS[idx][4]),c=colors[5],marker='+',linestyle='None',ms=markersize,label=label)

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

            # Device name
            fig.text(0.5, 0.9, device_name[device], ha='center')
            
            # Common axis labels
            fig.text(0.5, 0.1, 'Arithmetic Intensity [FLOPs/Byte]', ha='center')
            fig.text(0.1, 0.5, 'Performance [GFLOP/sec]', va='center', rotation='vertical')

            handles = list()
            src_name = ['fused (DRAM)', 'TVM layer_0 (DRAM)', 'TVM layer_1 (DRAM)', 'Library layer_0 (DRAM)', 'Library layer_1 (DRAM)', 'fused (Cache)', 'TVM layer_0 (Cache)', 'TVM layer_1 (Cache)', 'Library layer_0 (Cache)', 'Library layer_1 (Cache)']
            j = 0
            for i in range(0, len(src_name)):
                handles.append(Line2D([], [], color=colors[j], marker='*' if i < len(src_name) // 2 else '+', linestyle='None', markersize=markersize))
                if not ((i % 5 == 1) or (i % 5 == 3)):
                    j += 1
            leg2 = fig.legend(handles, src_name, loc='lower right', bbox_to_anchor=(0.80, 0.05))
            plt.savefig(filename + '.png', bbox_inches='tight')
            plt.savefig(filename + '.eps')

            #plt.show()