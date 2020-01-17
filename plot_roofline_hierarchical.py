import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
import matplotlib.patches as mpatches
font = { 'size'   : 15}
plt.rc('font', **font)

filename = "plot"
markersize = 10
colors = ['b','g','r','y','m','c']
styles = ['o','s','v','^','D',">","<","*","h","H","+","1","2","3","4","8","p","d","|","_",".",","]

AI_dram_generated = []
AI_dram_cudnn = []
AI_L2_generated = []
AI_L2_cudnn = []
FLOPS_generated = []
FLOPS_cudnn = []
labels = []
scomproofs = []
scomp_roof_name = []
smemroofs = []
smem_roof_name = []

workload_types = ['depth_conv']
for w in workload_types:
    workload_filename = "workloads/{}_workloads.csv".format(w)
    with open(workload_filename , "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines[1:]): # skip header
            splitted = line.strip().split(',')
            workload_name = splitted[0]
            # Read AI
            with open("logs/arithmetic_intensity/gpu/{}.csv".format(workload_name), "r") as ff:
                lls = ff.readlines()
                for l in lls[1:]: # skip header
                    splitted = l.strip().split(',')
                    AI_dram_generated.append(float(splitted[0]))
                    AI_dram_cudnn.append(float(splitted[1]))
                    AI_L2_generated.append(float(splitted[2]))
                    AI_L2_cudnn.append(float(splitted[3]))
            # Read FLOPS
            with open("logs/flops/gpu/{}.csv".format(workload_name), "r") as ff:
                lls = ff.readlines()
                for l in lls[1:]: # skip header
                    splitted = l.strip().split(',')
                    FLOPS_generated.append(float(splitted[0]))
                    FLOPS_cudnn.append(float(splitted[1]))
            # Labels
            labels.append(workload_name)
            if idx >= 5:
                break

device_info_file_dir = "/home/zhongyilin/Documents/experimental/cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0/Results.1050Ti-cuda-fp32.01/Run.001/roofline.json"
with open(device_info_file_dir, "r") as f:
    device_info = json.load(f)
    for l in device_info["empirical"]["gbytes"]["data"]:
        if l[0] == "DRAM":
            smem_roof_name.append("DRAM")
            smemroofs.append(float(l[1]))
        elif l[0] == "L1":
            smem_roof_name.append("L2") # For GPU, ERT recognizes L2 as L1
            smemroofs.append(float(l[1]))
    for l in device_info["empirical"]["gflops"]["data"]:
        if l[0] == "FP32 GFLOPs":
            scomp_roof_name.append("FP32")
            scomproofs.append(float(l[1]))

# print(AI_dram_generated, AI_dram_cudnn, AI_L2_generated, AI_L2_cudnn, \
#       FLOPS_generated, FLOPS_cudnn, \
#       labels, scomproofs, scomp_roof_name, smemroofs, smem_roof_name)

fig = plt.figure(1,figsize=(10.67,6.6))
plt.clf()
ax = fig.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
ax.set_ylabel('Performance [GFLOP/sec]')
#ax.set_title('Original GPU Performance')

nx = 10000
xmin = -0.15
xmax = 2.3
ymin = 100.0
ymax = 3000

ax.set_xlim(10**xmin, 10**xmax)
ax.set_ylim(ymin, ymax)

ixx = int(nx*0.02)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

scomp_x_elbow = [] 
scomp_ix_elbow = [] 
smem_x_elbow = [] 
smem_ix_elbow = [] 

x = np.logspace(xmin,xmax,nx)
for roof in scomproofs:
    for ix in range(1,nx):
        if smemroofs[0] * x[ix] >= roof and smemroofs[0] * x[ix-1] < roof:
            scomp_x_elbow.append(x[ix-1])
            scomp_ix_elbow.append(ix-1)
            break


for roof in smemroofs:
    for ix in range(1,nx):
        if (scomproofs[0] <= roof * x[ix] and scomproofs[0] > roof * x[ix-1]):
            smem_x_elbow.append(x[ix-1])
            smem_ix_elbow.append(ix-1)
            break

for i in range(0,len(scomproofs)):
    roof = scomproofs[i]
    y = np.ones(len(x)) * roof
    ax.plot(x[scomp_ix_elbow[i]:],y[scomp_ix_elbow[i]:],c='k',ls='-',lw='2')

for i in range(0,len(smemroofs)):
    roof = smemroofs[i]
    y = x * roof
    ax.plot(x[:smem_ix_elbow[i]+1],y[:smem_ix_elbow[i]+1],c='k',ls='-',lw='2')


marker_handles = list()

# workload: marker styles
# generated/cudnn: colors
for i in range(0,len(AI_dram_generated)):
  ax.plot(float(AI_dram_generated[i]),float(FLOPS_generated[i]),c=colors[0],marker=styles[i],linestyle='None',ms=markersize,label=labels[i])
  marker_handles.append(ax.plot([],[],c='gray',marker=styles[i],linestyle='None',ms=markersize,label=labels[i])[0]) 
# ax.plot((AI_dram_generated),(FLOPS_generated),c='k',linestyle='-.')
for i in range(0,len(AI_dram_cudnn)):
  ax.plot(float(AI_dram_cudnn[i]),float(FLOPS_cudnn[i]),c=colors[1],marker=styles[i],linestyle='None',ms=markersize,label=labels[i])
  marker_handles.append(ax.plot([],[],c='gray',marker=styles[i],linestyle='None',ms=markersize,label=labels[i])[0]) 
# ax.plot((AI_dram_cudnn),(FLOPS_cudnn),c='k',linestyle='-.')

# for i in range(0,len(AI_L2_generated)):
#   ax.plot(float(AI_L2_generated[i]),float(FLOPS_generated[i]),c=colors[0],marker=styles[i],linestyle='None',ms=markersize,label=labels[i])
# #ax.plot((AI_L2),(FLOPS_generated),c=colors[1],linestyle='-')
# for i in range(0,len(AI_L2_cudnn)):
#   ax.plot(float(AI_L2_cudnn[i]),float(FLOPS_cudnn[i]),c=colors[1],marker=styles[i],linestyle='None',ms=markersize,label=labels[i])
# #ax.plot((AI_L2),(FLOPS_cudnn),c=colors[1],linestyle='-')

for roof in scomproofs:
    ax.text(x[-ixx],roof,
            scomp_roof_name[scomproofs.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GFLOP/s',
            horizontalalignment='right',
            verticalalignment='bottom')

for roof in smemroofs:
    ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0]) 
                                 * fig.get_size_inches()[1]/fig.get_size_inches()[0] )
    if x[ixx]*roof >ymin:
        ax.text(x[ixx],x[ixx]*roof*(1+0.25*np.sin(ang)**2),
            smem_roof_name[smemroofs.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GB/s',
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
        ax.text(x[ixx+ymin_ix_elbow[0]],x[ixx+ymin_ix_elbow[0]]*roof*(1+0.25*np.sin(ang)**2),
            smem_roof_name[smemroofs.index(roof)] + ': ' + '{0:.1f}'.format(float(roof)) + ' GB/s',
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation=180/np.pi*ang)


# leg1 = plt.legend(handles = marker_handles,loc=4, ncol=2)
# ax.add_artist(leg1)

patch_handles = list()
src_name = ["generated", "cudnn"]
for i in range(0,len(smem_roof_name)):
    patch_handles.append(mpatches.Patch(color=colors[i],label=src_name[i]))

leg2 = plt.legend(handles = patch_handles,loc='lower right',scatterpoints = 1)

ax.text(xlim[0]*1.1,ylim[1]/1.1,'1050Ti',horizontalalignment='left',verticalalignment='top')

plt.savefig(filename+'.png')
plt.savefig(filename+'.eps')

#plt.show()
