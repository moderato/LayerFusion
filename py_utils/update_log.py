import os, argparse
import tvm
from tvm.topi.fusion_composer import FusionComposer
from tvm.autotvm.record import load_from_file, encode

def convert(in_file, layout='NHWC'):
    if os.path.isfile(in_file):
        in_file = [in_file]
    elif os.path.isdir(in_file):
        tmp = []
        for f in os.listdir(in_file):
            if f.endswith('.log'):
                tmp.append(os.path.join(in_file, f))
        in_file = tmp
    else:
        raise Exception('Unrecognized path!')

    for f in in_file:
        print(f)
        records = load_from_file(f)
        d = {}
        for k, v in records:
            if 'fused_conv2d' in k.task.workload[0] and k.task.workload[1][0] != 'TENSOR': # Not updated
                tgt, config = k.target, k.config
                name, workload = k.task.workload
                fc = FusionComposer(workload, use_autotvm=True, target=tgt, workload_name="xx", workspace='.')
                sargs = tvm.autotvm.task.topi_integration.serialize_args(list(fc.make_params(layout=layout).values()))
                new_tsk = tvm.autotvm.task.Task(name, sargs)
                new_mi = tvm.autotvm.MeasureInput(tgt, new_tsk, config)
                # print(sargs)
                d[new_mi] = v
            else: # Updated
                d[k] = v

        write_mode = 'w'
        with open(f, write_mode) as f:
            for (k, v) in d.items():
                f.write(encode(k, v) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Update fused conv2d workload in old logs.")
    parser.add_argument("-p", "--path", type=str, help="Log file path or folder path")
    parser.add_argument("-l", "--layout", type=str, default='NHWC', help="Log file path or folder path")
    options = parser.parse_args()
    convert(options.path, options.layout)
