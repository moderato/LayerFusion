from fusion_composer import *
from helper import duplicate_fusion_logs, get_workloads
import argparse

def get_options():
    parser = argparse.ArgumentParser(description="Parses command.")
    options = parser.parse_args()
    return options

if __name__ == '__main__':
    options = get_options()
    device = 'cpu'
    log_dir = 'logs/autotvm/layer/{}'.format(device)
    workloads = get_workloads()
    relus = {
        'mv1': ['relu', 'relu'],
        'mv2': ['relu6', 'bias'],
        'mna1': ['relu', 'bias'],
        'res': ['relu', 'bias']
    }

    for workload_type in workloads.keys():
        for w in workloads[workload_type].keys():
            log_name = '{}_fused_{}.log'.format(workload_type, w)
            logfile = '{}/fused/{}'.format(log_dir, log_name)
            relu = None
            for k in relus.keys():
                if k in w:
                    relu = relus[k]
            assert relu is not None
            print(logfile)
            duplicate_fusion_logs(logfile, relu)