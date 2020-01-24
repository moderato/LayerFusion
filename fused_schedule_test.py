import tvm, topi
import topi.testing
import os, logging, sys, argparse
import topi.tag as tag
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from tvm import autotvm

from helper import *

np.random.seed(42)
targets = {
    # "cuda": {
    #     "key": "1050ti",
    #     "host": None,
    #     "timeout": {
    #         "depth_conv": 10,
    #         "conv_conv": 500
    #     }
    # },
    "llvm -mcpu=core-avx2": {
        "key": "i7_7700K",
        "host": "llvm -target=x86_64-linux-gnu",
        "timeout": {
            "depth_conv": 200,
            "conv_conv": 10000
        }
    }
}

def verify_fused(workload_name,
                    parameters,
                    dtype="float32", 
                    layout="NHWC", 
                    no_print_ir=False, 
                    print_src=False, 
                    save_data=False, 
                    export_code=False, 
                    auto_tvm=False, 
                    auto_tvm_skip_training=False, 
                    auto_tvm_trials=20, 
                    name='depth_conv'):
    assert layout in ["NHWC", "NCHW", "NCHWc16", "NCHWc4"]

    ref_data = get_ref_data(workload_name, parameters, dtype=dtype, layout=layout, save_data=save_data, name=name)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        if "llvm" in device:
            ctx = tvm.cpu()
            device_name = "cpu"
        else: # cuda
            ctx = tvm.context(device, 0)
            device_name = "gpu"

        if auto_tvm:
            log_name = 'logs/autotvm/{}/{}_fused_{}.log'.format(device_name, name, workload_name)
            print(log_name)

            # logging
            logging.getLogger('autotvm').setLevel(logging.DEBUG)
            logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

            # fused schedule auto
            sargs = autotvm.task.topi_integration.serialize_args([parameters, auto_tvm, device, name])
            task = autotvm.task.create(get_schedule, args=sargs, target=device, target_host=targets[device]["host"])
            print(task.config_space)
            print(task.target)
            print(task.workload)

            if not auto_tvm_skip_training:
                # autotvm setting
                measure_option = autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.RPCRunner(
                        targets[device]["key"], '0.0.0.0', 9190,
                        number=10, repeat=1, timeout=targets[device]["timeout"][name], min_repeat_ms=500)
                )
                tuner = autotvm.tuner.XGBTuner(task)
                tuner.tune(n_trial=auto_tvm_trials,
                        measure_option=measure_option,
                        callbacks=[autotvm.callback.log_to_file(log_name)])

            # inspect the best config
            dispatch_context = autotvm.apply_history_best(log_name)
            best_config = dispatch_context.query(task.target, task.workload)
            print("\nBest config:")
            print(best_config)

            # export kernel launch config only for gpus, e.g. thxyz, blxy
            output_shape = ref_data[-1].shape
            if device == "cuda":
                export_kernel_launch_config(workload_name, output_shape, best_config)

            # apply history best from log file
            with autotvm.apply_history_best(log_name):
                with tvm.target.create(device):
                    s, flatten_params = get_schedule(parameters, auto_tvm, device, name)
        else:
            with tvm.target.create(device):
                s, flatten_params = get_schedule(parameters, auto_tvm, device, name)

        if not no_print_ir:
            print(tvm.lower(s, flatten_params, simple_mode=True))
        func = tvm.build(s, flatten_params, device, name="fused_2")
        if print_src:
            if device == "cuda":
                print(func.imported_modules[0].get_source())
            else:
                print(func.get_source("asm")) # assembly code
        if export_code:
            if device == "cuda": # Only support cuda for now
                code = func.imported_modules[0].get_source()
                write_code(code, "generated_kernels/{}.cuh".format(workload_name))

        # Prepare data
        nd_arrays = []
        for idx, array in enumerate(ref_data):
            if idx != len(ref_data) - 1: # Append data to nd_arrays
                nd_arrays.append(tvm.nd.array(array, ctx))
            else: # Leave the last nd_array as all-zero
                nd_arrays.append(tvm.nd.array(np.zeros(get_const_tuple(array.shape), dtype=dtype), ctx)) # Append 0 output data

        timer_1 = func.time_evaluator(func.entry_name, ctx, number=10)
        tcost_1 = timer_1(*nd_arrays).mean
        # np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        d = ~np.isclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        if (np.sum(d) > 0):
            print("# of incorrect numbers: {}".format(len(ref_data[-1][d])))
            print(nd_arrays[-1].asnumpy()[d])
            print(ref_data[-1][d])
            print(np.where(d))
        # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[-1]) * 100)))
        print("{}_fused of 2 layers ({}): average running time is {:.2f} us.".format(name, layout, tcost_1 * 1e6))

    for device in targets.keys():
        check_device(device)
    print("############################################")

if __name__ == "__main__":
    # For AutoTVM:
    # terminal 1: python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
    # terminal 2: python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=1050ti
    # terminal 3: run this code

    def get_options():
        parser = argparse.ArgumentParser(description="Parses command.")
        parser.add_argument("-n", "--no_print_ir", action="store_true", help="Don't print IR code.")
        parser.add_argument("-s", "--print_src", action="store_true", help="Print source code.")
        parser.add_argument("-d", "--save_data", action="store_true", help="Save numpy data as npy files.")
        parser.add_argument("-c", "--export_code", action="store_true", help="Export generated kernel code.")
        parser.add_argument("-a", "--auto_tvm", action="store_true", help="AutoTVM for auto tuning.")
        parser.add_argument("-k", "--auto_tvm_skip_training", action="store_true", help="Run AutoTVM tuned kernel.")
        parser.add_argument("-t", "--auto_tvm_trials", type=int, default=20, help="Number of AutoTVM trials")
        options = parser.parse_args()
        return options

    options = get_options()
    # workloads = get_workloads()
    workloads = get_workloads_from_file()

    for t, workload in workloads.items():
        for workload_name, parameters in workload.items():
            print(workload_name, parameters)
            verify_fused(workload_name,
                            parameters,
                            no_print_ir=options.no_print_ir,
                            print_src=options.print_src,
                            save_data=options.save_data,
                            export_code=options.export_code,
                            auto_tvm=options.auto_tvm,
                            auto_tvm_skip_training=options.auto_tvm_skip_training,
                            auto_tvm_trials=options.auto_tvm_trials,
                            name=t)
