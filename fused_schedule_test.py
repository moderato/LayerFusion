import tvm, topi
import topi.testing
import os, logging, sys, argparse
import topi.tag as tag
from scipy import signal
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from tvm import autotvm
from general_fused_compute import get_schedule

from helper import *

np.random.seed(42)
targets = {
    "cuda": {
        "key": "1050ti",
        "config_params": {
            "number": 100, # Number of runs for runtime averaging
            "repeat": 3, # (number of runs) = 1 repeats
            # Suggested min_repeat_ms = 150 on GPUs
            "min_repeat_ms": 300, # Dynamically adjust number of runs, i.e. time of one repeat = min(min_repeat_ms, number * kernel_runtime)
            "timeout": { # Timeout of a compilation
                "depth_conv": 10,
                "conv_conv": 500
            }
        }
    },
    "llvm -mcpu=core-avx2": {
        "key": "i7_7700K",
        "config_params": {
            "number": 200,
            "repeat": 3,
            "min_repeat_ms": 0,
            "timeout": {
                "depth_conv": 500,
                "conv_conv": 10000
            }
        }
    }
}

def verify_fused(workload_name,
                    parameters,
                    dtype="float32", 
                    layout="NHWC", 
                    no_print_ir=False, 
                    print_src=False, 
                    dry_run=False,
                    save_data=False, 
                    export_code=False, 
                    auto_tvm=False, 
                    auto_tvm_skip_training=False, 
                    auto_tvm_trials=20, 
                    name='depth_conv'):
    assert layout in ["NHWC", "NCHW", "NCHWc16", "NCHWc4"]

    def check_target(target):
        if not tvm.runtime.enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        if "llvm" in target:
            ctx = tvm.cpu()
            device = "cpu"
        else: # cuda
            ctx = tvm.gpu()
            device = "gpu"

        if auto_tvm:
            log_name = 'logs/autotvm/{}/{}_fused_{}.log'.format(device, name, workload_name)
            print(log_name)

            # logging
            logging.getLogger('autotvm').setLevel(logging.DEBUG)
            logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

            # fused schedule auto
            sargs = autotvm.task.topi_integration.serialize_args([parameters, auto_tvm, target, name])
            task = autotvm.task.create("fused", args=sargs, target=target)
            print(task.config_space)
            print(task.target)
            print(task.workload)

            if not auto_tvm_skip_training:
                # autotvm setting
                measure_option = autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.RPCRunner(
                        targets[target]["key"], '0.0.0.0', 9190,
                        number=targets[target]["config_params"]["number"],
                        repeat=targets[target]["config_params"]["repeat"],
                        timeout=targets[target]["config_params"]["timeout"][name],
                        min_repeat_ms=targets[target]["config_params"]["min_repeat_ms"])
                )
                tuner = autotvm.tuner.XGBTuner(task, feature_type="curve")

                # Transfer learning if the training log exists
                if os.path.isfile(log_name):
                    tuner.load_history(autotvm.record.load_from_file(log_name))

                tuner.tune(n_trial=auto_tvm_trials,
                            measure_option=measure_option,
                            callbacks=[autotvm.callback.progress_bar(auto_tvm_trials),
                                        autotvm.callback.log_to_file(log_name)])

            # inspect the best config
            dispatch_context = autotvm.apply_history_best(log_name)
            best_config = dispatch_context.query(task.target, task.workload)
            print("\nBest config:")
            print(best_config)

            # apply history best from log file
            with dispatch_context:
                with tvm.target.create(target):
                    s, flatten_params = get_schedule(parameters, auto_tvm, target, name)
        else:
            with tvm.target.create(target):
                s, flatten_params = get_schedule(parameters, auto_tvm, target, name)

        if not no_print_ir:
            print(tvm.lower(s, flatten_params, simple_mode=True))
        func = tvm.build(s, flatten_params, target, name="fused_2")
        if print_src:
            if target == "cuda":
                print(func.imported_modules[0].get_source())
            else:
                print(func.get_source("asm")) # assembly code
        if dry_run: # Only print IR and/or source
            return
        if export_code:
            if target == "cuda":
                code = func.imported_modules[0].get_source()
                write_code(code, "generated_kernels/gpu/{}.cuh".format(workload_name))
            else: # CPU
                code = func.get_source("asm")
                write_code(code, "generated_kernels/cpu/{}.asm".format(workload_name))

                # func.export_library("benchmark/cpu/kernel.so")
                # func_sys = tvm.build(s, flatten_params, target + " --system-lib", name="fused_2_sys")
                # func_sys.save("benchmark/cpu/kernel_sys.o")

        # Prepare data
        ref_data = get_ref_data(workload_name, parameters, dtype=dtype, layout=layout, save_data=save_data, name=name)

        # export kernel launch config ONLY FOR GPUS, e.g. thxyz, blxy
        output_shape = ref_data[-1].shape
        if target == "cuda" and auto_tvm:
            assert (best_config is not None)
            export_kernel_launch_config(workload_name, output_shape, best_config)

        nd_arrays = []
        for idx, array in enumerate(ref_data):
            if idx != len(ref_data) - 1: # Append data to nd_arrays
                nd_arrays.append(tvm.nd.array(array, ctx))
            else: # Leave the last nd_array as all-zero
                nd_arrays.append(tvm.nd.array(np.zeros(get_const_tuple(array.shape), dtype=dtype), ctx)) # Append 0 output data

        # Measure a 'delta' time
        run_number = 1000
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=run_number)
        tcost_1 = timer_1(*nd_arrays).mean * run_number
        timer_2 = func.time_evaluator(func.entry_name, ctx, number=run_number*2)
        tcost_2 = timer_2(*nd_arrays).mean * run_number * 3
        tcost_d = (tcost_2 - tcost_1) / (run_number * 2)

        # np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        d = ~np.isclose(nd_arrays[-1].asnumpy(), ref_data[-1], rtol=1e-3)
        if (np.sum(d) > 0):
            print("# of incorrect numbers: {}".format(len(ref_data[-1][d])))
            print(nd_arrays[-1].asnumpy()[d])
            print(ref_data[-1][d])
            print(np.where(d))
        # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[-1]) * 100)))
        print("{}_fused of {} ({}): average running time is {:.2f} us.".format(name, workload_name, layout, tcost_d * 1e6))
        FLOP = autotvm.task.task.compute_flop(s)
        print("FLOP: {}, GFLOPS: {:.2f}.".format(FLOP, FLOP / tcost_d / 1e9))

    for target in ["llvm -mcpu=core-avx2"]:
        check_target(target)
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
        parser.add_argument("-y", "--dry_run", action="store_true", help="Dry run.")
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
                            dry_run=options.dry_run,
                            export_code=options.export_code,
                            auto_tvm=options.auto_tvm,
                            auto_tvm_skip_training=options.auto_tvm_skip_training,
                            auto_tvm_trials=options.auto_tvm_trials,
                            name=t)
