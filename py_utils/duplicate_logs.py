from fusion_composer import *
from helper import duplicate_fusion_logs, get_workloads
import argparse

def get_options():
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-n", "--no_axis", action="store_true", help="")
    options = parser.parse_args()
    return options

if __name__ == '__main__':
    options = get_options()
    axes = [''] if options.no_axis else ['oc', 'ic', 'h', 'w']
    device = 'cpu'
    log_dir = 'logs/autotvm/layer/{}'.format(device)
    workloads = get_workloads()
    relus = {
        'mv1': ['relu', 'relu'],
        'mv2': ['relu6', 'bias'],
        'mna1': ['relu', 'bias'],
        'res': ['relu', 'bias']
    }

    for axis in axes:
        for workload_type in workloads.keys():
            for w in workloads[workload_type].keys():
                log_name = '{}_fused_{}.log'.format(workload_type, w)
                logfile = '{}/fused{}/{}'.format(log_dir, (axis if axis == '' else '_' + axis), log_name)
                relu = None
                for k in relus.keys():
                    if k in w:
                        relu = relus[k]
                assert relu is not None
                print(logfile)
                duplicate_fusion_logs(logfile, relu)