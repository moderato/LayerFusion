from utils import get_workloads
import os
from tvm import autotvm
from tvm.topi.fusion_composer import FusionComposer
from tvm.topi.x86.fused_conv2d import *

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
        fc = FusionComposer(args[0], target=tgt)
        workload = list(fc.make_params().values())
        if workload[8] is [None, None]:
            workload[8] = post_ops
        else:
            workload[8] = [None, None]

        if len(args) == 1:
            new_args = (tuple(workload),)
        else:
            new_args = (tuple(workload), args[1])
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
    device = 'cpu'
    log_dir = 'logs/autotvm/layer/{}'.format(device)
    workloads = get_workloads()
    post_ops_dict = {
        'mv1': ['relu', 'relu'],
        'mv2': ['relu6', 'bias'],
        'mna1': ['relu', 'bias'],
        'res': ['relu', 'bias']
    }

    for workload_type in workloads.keys():
        for w in workloads[workload_type].keys():
            log_name = '{}_fused_{}.log'.format(workload_type, w)
            logfile = '{}/fused/{}'.format(log_dir, log_name)
            post_ops = None
            for k in post_ops_dict.keys():
                if k in w:
                    post_ops = post_ops_dict[k]
            assert post_ops is not None
            print(logfile)
            duplicate_fusion_logs(logfile, post_ops)
