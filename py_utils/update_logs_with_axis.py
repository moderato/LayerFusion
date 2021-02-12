from fusion_composer import *
from helper import remove_axis_from_logs, update_fusion_logs_with_axis, get_workloads

if __name__ == '__main__':

    device = 'cpu'
    log_dir = 'logs/autotvm/layer/{}'.format(device)
    workloads = get_workloads()
    axes = ['oc', 'ic', 'h', 'w']

    for axis in axes:
        for workload_type in workloads.keys():
            for w in workloads[workload_type].keys():
                log_name = '{}_fused_{}.log'.format(workload_type, w)
                logfile = '{}/fused_{}/{}'.format(log_dir, axis, log_name)
                remove_axis_from_logs(log_dir, log_name, axis)