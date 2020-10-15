import tvm, topi
import topi.testing
import os, logging, sys, argparse
import tvm.topi.tag as tag
from scipy import signal
from tvm.topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize
from tvm import autotvm

from helper import *
from fusion_composer import *

def verify_fused(workload_name,
                    parameters,
                    tuning_opt,
                    name='depth_conv',
                    dtype="float32",):

    no_print_ir = tuning_opt.no_print_ir
    print_src = tuning_opt.print_src
    save_data = tuning_opt.save_data
    dry_run = tuning_opt.dry_run
    export_code = tuning_opt.export_code
    auto_tvm = tuning_opt.auto_tvm
    auto_tvm_skip_training = tuning_opt.auto_tvm_skip_training
    auto_tvm_transfer_learning = tuning_opt.auto_tvm_transfer_learning
    auto_tvm_trials = tuning_opt.auto_tvm_trials

    def check_target(target_str):
        if not tvm.runtime.enabled(target_str):
            print("Skip because %s is not enabled" % target_str)
            return
        print("Running on target: %s" % target_str)
        if "llvm" in target_str:
            ctx = tvm.cpu()
            target = tvm.target.create(target_str)
            device = 'cpu'
        else: # cuda
            ctx = tvm.gpu()
            target = tvm.target.create(target_str)
            device = 'gpu'

        fc = FusionComposer(parameters, auto_tvm=auto_tvm, target=target)
        if auto_tvm:
            log_name = 'logs/autotvm/layer/{}/{}_fused_{}.log'.format(device, name, workload_name)
            print(log_name)

            # logging
            logging.getLogger('autotvm').setLevel(logging.DEBUG)
            logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

            # fused schedule auto
            sargs = autotvm.task.topi_integration.serialize_args([parameters, auto_tvm, target_str, name])
            task = autotvm.task.create("fused", args=sargs, target=target_str)
            print(task.config_space)
            print(task.target)
            print(task.workload)

            if not auto_tvm_skip_training:
                # autotvm setting
                measure_option = autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.RPCRunner(
                        TARGETS[target_str]["key"], '0.0.0.0', 9190,
                        number=TARGETS[target_str]["config_params"]["number"],
                        repeat=TARGETS[target_str]["config_params"]["repeat"],
                        timeout=TARGETS[target_str]["config_params"]["timeout"][name],
                        min_repeat_ms=TARGETS[target_str]["config_params"]["min_repeat_ms"])
                )
                tuner = autotvm.tuner.XGBTuner(task, feature_type="curve")

                # Transfer learning if the training log exists
                if auto_tvm_transfer_learning and os.path.isfile(log_name):
                    tuner.load_history(autotvm.record.load_from_file(log_name))

                tuner.tune(n_trial=auto_tvm_trials,
                            measure_option=measure_option,
                            callbacks=[autotvm.callback.progress_bar(auto_tvm_trials),
                                        autotvm.callback.log_to_file(log_name)])

            # inspect the best config
            # autotvm.record.pick_best(log_name, "logs/autotvm/model/gpu/test.log")
            dispatch_context = autotvm.apply_history_best(log_name)
            best_config = dispatch_context.query(task.target, task.workload)
            print("\nBest config:")
            print(best_config)

            # apply history best from log file
            with dispatch_context:
                with tvm.target.create(target_str):
                    s, flatten_params = get_schedule_results(parameters, auto_tvm, target_str, name)
        else:
            best_config = None
            with tvm.target.create(target_str):
                s, flatten_params = get_schedule_results(parameters, auto_tvm, target_str, name)

        if not no_print_ir:
            print(tvm.lower(s, flatten_params, simple_mode=True))
        func = tvm.build(s, flatten_params, target_str, name="fused_2")
        if print_src:
            if target_str == "cuda":
                print(func.imported_modules[0].get_source())
            else:
                print(func.get_source("asm")) # assembly code
        if dry_run: # Only print IR and/or source
            return
        if export_code:
            if target_str == "cuda":
                code = func.imported_modules[0].get_source()
                write_code(code, "generated_kernels/gpu/{}.cuh".format(workload_name))
            else: # CPU
                code = func.get_source("asm")
                write_code(code, "generated_kernels/cpu/{}.asm".format(workload_name))

                # func.export_library("benchmark/cpu/kernel.so")
                # func_sys = tvm.build(s, flatten_params, target_str + " --system-lib", name="fused_2_sys")
                # func_sys.save("benchmark/cpu/kernel_sys.o")

        # Prepare data
        ref_data = fc.get_ref_data(workload_name, best_config, save_data=save_data)

        # export kernel launch config, e.g. thxyz, blxy, vlen, etc
        output_shape = ref_data[-1].shape
        if auto_tvm:
            assert (best_config is not None)
            export_kernel_launch_config(workload_name, output_shape, best_config, target_str)

        nd_arrays = []
        for idx, array in enumerate(ref_data):
            # if idx == 2 or idx == 3:
            #     print(array)
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
        print("{}_fused of {} ({}): average running time is {:.2f} us.".format(name, workload_name, "NHWC" if target_str == "cuda" else "NCHWc", tcost_d * 1e6))
        FLOP = fc.get_FLOP()
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
        parser.add_argument("-l", "--auto_tvm_transfer_learning", action="store_true", help="Load existing tuning log.")
        parser.add_argument("-t", "--auto_tvm_trials", type=int, default=32, help="Number of AutoTVM trials.")
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
                            options,
                            name=t)
