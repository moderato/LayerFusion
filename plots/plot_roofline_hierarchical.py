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
iterations = [40, 2, 20000]
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
arrowprops = dict(arrowstyle="-|>,head_width=0.15,head_length=0.55", linewidth="0.001", shrinkA=0, shrinkB=0, fc="k", ec="k")

HOME = os.getenv('LF_HOME')

for device in devices:
    for iter in iterations:
        filename = '{}_{}_plot'.format(device_name[device], iter)

        AI_dram_generated = []
        AI_dram_library = []
        AI_L2_generated = []
        AI_L2_library = []
        FLOPS_generated = []
        FLOPS_library = []
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
                            splitted = l.strip().split(',')
                            AI_dram_generated.append(float(splitted[0]))
                            AI_dram_library.append(float(splitted[1]))
                            if len(splitted) == 4:
                                has_L2 = True
                                AI_L2_generated.append(float(splitted[2]))
                                AI_L2_library.append(float(splitted[3]))
                    # Read FLOPS
                    with open('{}/logs/gflops/{}/{}/{}.csv'.format(HOME, device, iter, workload_name), 'r') as ff:
                        lls = ff.readlines()
                        for l in lls[1:]: # skip header
                            splitted = l.strip().split(',')
                            FLOPS_generated.append(float(splitted[0]))
                            FLOPS_library.append(float(splitted[1]))
                    # Labels
                    labels.append(workload_name)

                    # if idx >= 5:
                    #     break

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

            print(AI_dram_generated, AI_dram_library, AI_L2_generated, AI_L2_library, \
                    FLOPS_generated, FLOPS_library, \
                    labels, \
                    gflops_roof, gflops_roof_name, \
                    BW_roof, BW_roof_name)

            fig = plt.figure(1, figsize=(10.67, 6.6))
            plt.clf()
            ax = fig.gca()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
            ax.set_ylabel('Performance [GFLOP/sec]')
            # ax.set_title('Original GPU Performance')

            nx = 10000
            if device == "gpu":
                ymin = 70.0
                ymax = 3000.0
                xmin = 0.1
                xmax = 2.3
                text_distance_scale = 1.1
            else:
                ymin = min(flatten_list([FLOPS_generated, FLOPS_library])) / 1.4 # 160.0
                ymax = max(gflops_roof) * 1.1 # 600.0
                xmin = math.log10(min(flatten_list([AI_dram_generated, AI_dram_library, AI_L2_generated, AI_L2_library])) / 1.4) # -0.4
                xmax = math.log10(max(flatten_list([AI_dram_generated, AI_dram_library, AI_L2_generated, AI_L2_library])) * 2) # 3.1
                text_distance_scale = 1.02

            ax.set_xlim(10**xmin, 10**xmax)
            ax.set_ylim(ymin, ymax)

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
            for i in range(0, len(labels)):
                ax.plot(float(AI_dram_generated[i]), float(FLOPS_generated[i]), c=colors[0], marker=styles[i], linestyle='None', ms=markersize, label=labels[i])
                marker_handles.append(ax.plot([], [], c='gray', marker=styles[i], linestyle='None', ms=markersize, label=labels[i])[0]) 
            for i in range(0, len(labels)):
                ax.plot(float(AI_dram_library[i]),float(FLOPS_library[i]), c=colors[1], marker=styles[i], linestyle='None', ms=markersize, label=labels[i])
                marker_handles.append(ax.plot([], [], c='gray', marker=styles[i], linestyle='None', ms=markersize, label=labels[i])[0]) 
            for i in range(0, len(AI_dram_generated)):
                x0 = float(AI_dram_library[i])
                y0 = float(FLOPS_library[i])
                dx = (float(AI_dram_generated[i]) - float(AI_dram_library[i]))
                dy = (float(FLOPS_generated[i]) - float(FLOPS_library[i]))
                dx_dy = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
                scaled_dx = dx
                scaled_dy = dy
                ax.arrow(x0, y0, scaled_dx, scaled_dy, linestyle=(0, (1, 6)))
                ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0), arrowprops=arrowprops)
                ax.text(x0 * (0.98 if dx > 0 else 1.02), y0 * 0.98, labels[i], fontsize=8)

            if has_L2:
                for i in range(0,len(AI_L2_generated)):
                    ax.plot(float(AI_L2_generated[i]),float(FLOPS_generated[i]),c=colors[2],marker=styles[i],linestyle='None',ms=markersize,label=labels[i])
                for i in range(0,len(AI_L2_library)):
                    ax.plot(float(AI_L2_library[i]),float(FLOPS_library[i]),c=colors[3],marker=styles[i],linestyle='None',ms=markersize,label=labels[i])
                for i in range(0, len(AI_L2_generated)):
                    x0 = float(AI_L2_library[i])
                    y0 = float(FLOPS_library[i])
                    dx = (float(AI_L2_generated[i]) - float(AI_L2_library[i]))
                    dy = (float(FLOPS_generated[i]) - float(FLOPS_library[i]))
                    dx_dy = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
                    scaled_dx = dx
                    scaled_dy = dy
                    ax.arrow(x0, y0, scaled_dx, scaled_dy, linestyle=(0, (1, 6)))
                    ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0), arrowprops=arrowprops)
                    # ax.text(x0 * (0.98 if dx > 0 else 1.02), y0 * 0.98, labels[i], fontsize=8)

            for roof in gflops_roof:
                ax.text(x[-ixx],roof,
                        gflops_roof_name[gflops_roof.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GFLOP/s',
                        horizontalalignment='right',
                        verticalalignment='bottom')

            for roof in BW_roof:
                ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0]) 
                                            * fig.get_size_inches()[1]/fig.get_size_inches()[0])
                if x[ixx]*roof >ymin:
                    ax.text(x[ixx],x[ixx]*roof*(1+0.25*np.sin(ang)**2),
                        BW_roof_name[BW_roof.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GB/s',
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        rotation=180/np.pi*ang)
                else:
                    ymin_ix_elbow=list()
                    ymin_x_elbow=list()
                    for ix in range(1,nx):
                        if (ymin <= roof * x[ix] and ymin > roof * x[ix-1]):
                            ymin_x_elbow.append(x[ix-1])
                            ymin_ix_elbow.append(ix-1)
                            break
                    ax.text(x[ixx+ymin_ix_elbow[0]]*0.8,\
                        x[ixx+ymin_ix_elbow[0]]*roof*(1+0.05*np.sin(ang)**2),
                        BW_roof_name[BW_roof.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GB/s',
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        rotation=180/np.pi*ang)

            # leg1 = plt.legend(handles = marker_handles,loc=4, ncol=2)
            # ax.add_artist(leg1)

            patch_handles = list()
            src_name = ['generated (DRAM)', 'library (DRAM)', 'generated (Cache)', 'library (Cache)']
            # print(BW_roof_name)
            for i in range(0, len(src_name)):
                patch_handles.append(mpatches.Patch(color=colors[i],label=src_name[i]))

            leg2 = plt.legend(handles = patch_handles, loc='lower right', scatterpoints = 1)

            ax.text(xlim[0]*text_distance_scale, ylim[1]/text_distance_scale, device_name[device], horizontalalignment='left', verticalalignment='top')

            plt.savefig(filename + '.png')
            plt.savefig(filename + '.eps')

            #plt.show()
