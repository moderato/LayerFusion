from helper import get_workloads
import argparse, os
from tvm import autotvm

def get_options():
    parser = argparse.ArgumentParser(description="Parses command.")
    options = parser.parse_args()
    return options

def duplicate_fusion_logs(logfile, post_ops=['relu', 'relu']):
    if not os.path.isfile(logfile):
        return
    records = autotvm.record.load_from_file(logfile)
    d1 = {}
    d2 = {}
    for k, v in records:
        d1[k] = v

    s = set([(k.task.args, k.config.index) for k in d1.keys()])
    for k, v in d1.items():
        tgt, tsk, config = k.target, k.task, k.config
        name, args = tsk.name, tsk.args
        if 'fused' not in name:
            continue
        new_args = list(args[0])
        c = 0
        while 8 + 5*c < len(new_args):
            if args[0][8 + 5*c] is None:
                new_args[8 + 5*c] = post_ops[c]
            else:
                new_args[8 + 5*c] = None
            c += 1
        if len(args) == 1:
            new_args = (tuple(new_args),)
        else:
            new_args = (tuple(new_args), args[1])
        new_tsk = autotvm.task.Task(name, new_args)
        new_mi = autotvm.MeasureInput(tgt, new_tsk, config)

        if (new_args, config.index) not in s:
            d2[new_mi] = v

    # print(len(d1.keys()))
    # print(len(d2.keys()))

    with open(logfile, "a") as f:
        for k, v in d2.items():
            f.write(autotvm.record.encode(k, v) + "\n")

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