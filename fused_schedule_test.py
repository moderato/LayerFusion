import tvm, os, logging, sys, argparse
from tvm.topi import testing
from tvm.topi.fusion_composer import FusionComposer
from tvm import autotvm, te, auto_scheduler
from tvm.topi.utils import get_const_tuple, export_kernel_launch_config
from utils import *

@auto_scheduler.register_workload
def get_auto_scheduler_task_x86(parameters):
    target = tvm.target.Target('llvm')
    fc = FusionComposer(parameters, use_autotvm=False, use_auto_scheduler=True, target=target)

    # Get compute
    compute = fc.get_compute(raw_compute=True)
    input_tensors = fc.make_placeholders()
    output_tensor = compute(input_tensors)
    all_tensors = input_tensors + [output_tensor]

    return all_tensors


def verify_tuning(workload_name,
                    workload_type,
                    parameters,
                    tuning_opt,
                    dtype='float32'):

    no_print_ir = tuning_opt.no_print_ir
    print_src = tuning_opt.print_src
    save_data = tuning_opt.save_data
    dry_run = tuning_opt.dry_run
    export_code = tuning_opt.export_code
    use_autotvm = tuning_opt.use_autotvm
    use_auto_scheduler = tuning_opt.use_auto_scheduler
    skip_training = tuning_opt.skip_training
    use_autotvm_transfer_learning = tuning_opt.use_autotvm_transfer_learning
    device_name = tuning_opt.device
    tuning_trials = tuning_opt.tuning_trials if workload_type == 'depth_conv' or workload_type == 'conv_depth' else 2 * tuning_opt.tuning_trials
    unfused = tuning_opt.unfused
    assert (device_name in DEVICES.keys()) # 'TITAN_xp', '1050Ti', '1080', 'Xeon_GCP', 'i7_7700K', 'Xeon_E5', 'EPYC'

    def check_device(device_name):
        target_str = DEVICES[device_name]['target']
        if not tvm.runtime.enabled(target_str):
            print('Skip because {} is not enabled'.format(target_str))
            return
        print('Running on target: {}'.format(target_str))
        if 'llvm' in target_str:
            ctx = tvm.cpu()
            target = tvm.target.Target(target_str)
            device = 'cpu'
        else: # cuda
            ctx = tvm.gpu()
            target = tvm.target.Target(target_str)
            device = 'gpu'

        fc = FusionComposer(parameters, use_autotvm=use_autotvm, use_auto_scheduler=use_auto_scheduler, target=target, workload_name=workload_name, workspace='.')
        if unfused:
            log_names = []
            tasks = []
            depthwises = []
            params = []
            for l in range(2):
                # 
                input_cfg = fc.get_input_cfg(l)
                filter_cfg= fc.get_filter_cfg(l)

                is_depthwise = filter_cfg.depthwise
                task_name = '{}conv2d_{}'.format('depthwise_' if is_depthwise else '', 'NHWC.cuda' if target_str == 'cuda' else 'NCHWc.x86')
                N, H, W, C = input_cfg.get_shape()
                data = te.placeholder((N, H, W, C) if target_str == 'cuda' else (N, C, H, W))
                if is_depthwise:
                    H, W, O, I = filter_cfg.get_shape()
                else:
                    H, W, I, O = filter_cfg.get_shape()
                kernel = te.placeholder(filter_cfg.get_shape() if target_str == 'cuda' else (O, I, H, W))
                stride = (filter_cfg.get_stride())
                padding = (filter_cfg.get_padding_shape())
                dilation = (filter_cfg.get_dilation())
                layout = 'NHWC' if target_str == 'cuda' else 'NCHW'
                out_layout = 'NHWC' if target_str == 'cuda' else 'NCHW'
                out_dtype = dtype
                params.append([data, kernel, stride, padding, dilation, layout, out_layout, out_dtype])
                depthwises.append(is_depthwise)

                if use_autotvm:
                    log_name = 'logs/autotvm/layer/{}/unfused/{}_{}.log'.format(device, workload_name, l+1)
                    log_names.append(log_name)
                    print(log_name)

                    # # logging
                    # logging.getLogger('autotvm').setLevel(logging.DEBUG)
                    # logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

                    sargs = autotvm.task.topi_integration.serialize_args([data, kernel, stride, padding, dilation, layout, out_layout, out_dtype])
                    task = autotvm.task.create(task_name, args=sargs, target=target)
                    print(task.config_space)
                    print(task.target)
                    print(task.workload)
                    tasks.append(task)

                    if not skip_training:
                        # autotvm setting
                        measure_option = autotvm.measure_option(
                            builder=autotvm.LocalBuilder(),
                            runner=autotvm.RPCRunner(
                                device_name, '0.0.0.0', 9190,
                                number=DEVICES[device_name]["config_params"]["number"],
                                repeat=DEVICES[device_name]["config_params"]["repeat"],
                                timeout=DEVICES[device_name]["config_params"]["timeout"][workload_type],
                                min_repeat_ms=DEVICES[device_name]["config_params"]["min_repeat_ms"],
                                enable_cpu_cache_flush=(True if device == 'cpu' else False)
                            )
                        )
                        tuner = autotvm.tuner.XGBTuner(task, feature_type="curve")

                        # Transfer learning if the training log exists
                        if use_autotvm_transfer_learning and os.path.isfile(log_name):
                            tuner.load_history(autotvm.record.load_from_file(log_name))

                        tuner.tune(n_trial=tuning_trials,
                                    measure_option=measure_option,
                                    callbacks=[autotvm.callback.progress_bar(min(tuning_trials, len(task.config_space))),
                                                autotvm.callback.log_to_file(log_name)])
                else:
                    raise Exception("Not accepting unfused kernels without AutoTVM")

            best_1 = autotvm.record.pick_best_batch(log_names[0], batch=100)
            best_2 = autotvm.record.pick_best_batch(log_names[1], batch=100)
            best_pair = None
            best_workloads = None
            best_cost = 1e10
            for b in best_1:
                inp_1, res_1 = b
                config_1 = inp_1.config
                config_dict = config_1.to_json_dict()
                vlen_oc = -1
                for e in config_dict['entity']:
                    if e[0] == 'tile_oc':
                        vlen_oc = int(e[2][-1])
                        break
                assert vlen_oc != -1
                cost_1 = config_1.cost
                for bb in best_2:
                    inp_2, res_2 = bb
                    config_2 = inp_2.config
                    config_dict = config_2.to_json_dict()
                    vlen_ic = -1
                    for e in config_dict['entity']:
                        if e[0] == 'tile_ic':
                            vlen_ic = int(e[2][-1])
                            break
                    if vlen_ic != vlen_oc:
                        continue
                    cost_2 = config_2.cost
                    if cost_1 + cost_2 < best_cost:
                        new_pair_1, new_workload_1 = create_nchwc_config(inp_1, res_1)
                        new_pair_2, new_workload_2 = create_nchwc_config(inp_2, res_2)
                        best_pair = (new_pair_1, new_pair_2)
                        best_workloads = (new_workload_1, new_workload_2)
                        best_cost = cost_1 + cost_2
            assert best_pair is not None

            prev_out = None
            for l in range(2):
                # inspect the best config
                # autotvm.record.pick_best(log_name, "logs/autotvm/model/{}/test.log".format(device))
                dispatch_context = autotvm.apply_history_best(log_names[l])
                new_key = (target.keys[0], best_workloads[l])
                dispatch_context.best_by_targetkey[new_key] = best_pair[l]
                best_config = dispatch_context.query(target, best_workloads[l])
                print('\nBest config:')
                print(best_config)

                # apply history best from log file
                with dispatch_context:
                    with target:
                        vlen_ic, vlen_oc = -1, -1
                        config_dict = best_config.to_json_dict()
                        for e in config_dict['entity']:
                            if e[0] == 'tile_ic':
                                vlen_ic = e[2][-1]
                            if e[0] == 'tile_oc':
                                vlen_oc = e[2][-1]
                        assert vlen_ic != -1 and vlen_oc != -1
                        n, c, h, w = params[l][0].shape
                        data_5D = te.placeholder((n, c//vlen_ic, h, w, vlen_ic))
                        o, i, h, w = params[l][1].shape
                        kernel_6D = te.placeholder((o//vlen_oc, 1, h, w, 1, vlen_oc) if depthwises[l] else (o//vlen_oc, i//vlen_ic, h, w, vlen_ic, vlen_oc))
                        s, flatten_params = tasks[l].func(data_5D, kernel_6D, *params[l][2:])

                if not no_print_ir:
                    print(tvm.lower(s, flatten_params, simple_mode=True))
                func = tvm.build(s, flatten_params, target_str, name='layer_{}'.format(l+1))
                if print_src:
                    if target_str == 'cuda':
                        print(func.imported_modules[0].get_source())
                    else:
                        print(func.get_source('asm')) # assembly code
                if dry_run: # Only print IR and/or source
                    return
                if export_code:
                    if target_str == 'cuda':
                        code = func.imported_modules[0].get_source()
                        write_code(code, 'generated_kernels/gpu/unfused/{}_{}.cuh'.format(workload_name, l+1))
                    else: # CPU
                        code = func.get_source("asm")
                        write_code(code, 'generated_kernels/cpu/unfused/{}_{}.asm'.format(workload_name, l+1))

                        # func.export_library("benchmark/cpu/kernel.so")
                        # func_sys = tvm.build(s, flatten_params, target_str + " --system-lib", name="fused_2_sys")
                        # func_sys.save("benchmark/cpu/kernel_sys.o")

                ref_data = []
                nd_arrays = []

                # Ridiculous return orders: h = 3 (2, 1, 0), h = 1 (2, 0, 1)
                if prev_out is None:
                    data_np = np.random.uniform(0.0, 0.1, size=get_const_tuple(params[l][0].shape)).astype(dtype)
                else:
                    data_np = prev_out
                n, c, h, w = data_np.shape
                data_np_5D = np.array(data_np.reshape((n, c//vlen_ic, vlen_ic, h, w)).transpose(0, 1, 3, 4, 2), order='C')
                ref_data.append(data_np_5D)
                nd_arrays.append(tvm.nd.array(data_np_5D, ctx))
                kernel_np = np.random.uniform(0.0, 0.1, size=get_const_tuple(params[l][1].shape)).astype(dtype)
                o, i, fh, fw = kernel_np.shape
                kernel_np_6D = np.array(kernel_np.reshape((o//vlen_oc, vlen_oc, 1, 1, fh, fw) if depthwises[l] else (o//vlen_oc, vlen_oc, i//vlen_ic, vlen_ic, fh, fw)).transpose(0, 2, 4, 5, 3, 1), order='C')
                if fh == 3:
                    ref_data = [kernel_np_6D] + ref_data
                    nd_arrays = [tvm.nd.array(kernel_np_6D, ctx)] + nd_arrays
                else:
                    ref_data.append(kernel_np_6D)
                    nd_arrays.append(tvm.nd.array(kernel_np_6D, ctx))
                if depthwises[l]:
                    output_np = testing.depthwise_conv2d_python_nchw(data_np, kernel_np, stride=params[l][2], padding='SAME').astype(dtype)
                else:
                    output_np = testing.conv2d_nchw_python(data_np, kernel_np, stride=params[l][2], padding='SAME').astype(dtype)
                prev_out = output_np

                n, c, h, w = output_np.shape
                output_np_5D = np.array(output_np.reshape((n, c//vlen_oc, vlen_oc, h, w)).transpose((0, 1, 3, 4, 2)), order='C')
                ref_data = [output_np_5D] + ref_data
                nd_arrays = [tvm.nd.array(np.zeros(get_const_tuple(output_np_5D.shape), dtype=dtype), ctx)] + nd_arrays

                output_shape = output_np.shape
                if use_autotvm:
                    assert (best_config is not None)
                    export_kernel_launch_config("{}_{}".format(workload_name, l+1), output_shape, best_config, target_str, unfused=unfused)

                if save_data:
                    folder_name = 'npy/unfused/{}_{}/'.format(workload_name, l+1)
                    if not os.path.exists(folder_name):
                        os.mkdir(folder_name)
                    np.save('{}/input.npy'.format(folder_name), ref_data[2] if fh == 3 else ref_data[1])
                    np.save('{}/filter.npy'.format(folder_name), ref_data[1] if fh == 3 else ref_data[2])
                    np.save('{}/output.npy'.format(folder_name), ref_data[0])

                # Measure a 'delta' time
                run_number = 1000
                timer_1 = func.time_evaluator(func.entry_name, ctx, number=run_number)
                tcost_1 = timer_1(*nd_arrays).mean * run_number
                timer_2 = func.time_evaluator(func.entry_name, ctx, number=run_number*2)
                tcost_2 = timer_2(*nd_arrays).mean * run_number * 3
                tcost_d = (tcost_2 - tcost_1) / (run_number * 2)

                # np.testing.assert_allclose(nd_arrays[-1].asnumpy(), ref_data[0], rtol=1e-3) # Output at ref_data[0]
                d = ~np.isclose(nd_arrays[0].asnumpy(), ref_data[0], rtol=1e-3)
                if (np.sum(d) > 0):
                    print('# of incorrect numbers: {}'.format(len(ref_data[-1][d])))
                    print(nd_arrays[0].asnumpy()[d])
                    print(ref_data[0][d])
                    print(np.where(d))
                # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[0]) * 100)))
                print('{} {}_{} ({}): average running time is {:.2f} us.'.format('Depthwise conv2d' if is_depthwise else 'Conv2d', workload_name, l+1, 'NCHW', tcost_d * 1e6))
                FLOP = tasks[l].flop
                print('FLOP: {}, GFLOPS: {:.2f}.'.format(FLOP, FLOP / tcost_d / 1e9))
        else: # Fused
            if use_autotvm:
                log_name = 'logs/autotvm/layer/{}/fused/{}_fused_{}.log'.format(device, workload_type, workload_name)
                print(log_name)

                # # logging
                # logging.getLogger('autotvm').setLevel(logging.DEBUG)
                # logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

                # fused schedule auto
                sargs = autotvm.task.topi_integration.serialize_args(list(fc.make_params(layout=('NHWC' if target_str == 'cuda' else 'NCHW')).values()))
                task = autotvm.task.create('fused_conv2d.{}'.format('cuda' if target_str == 'cuda' else 'x86'), args=sargs, target=target)
                print(task.config_space)
                print(task.target)
                print(task.workload)

                if not skip_training:
                    # autotvm setting
                    measure_option = autotvm.measure_option(
                        builder=autotvm.LocalBuilder(),
                        runner=autotvm.RPCRunner(
                            device_name, '0.0.0.0', 9190,
                            number=DEVICES[device_name]["config_params"]["number"],
                            repeat=DEVICES[device_name]["config_params"]["repeat"],
                            timeout=DEVICES[device_name]["config_params"]["timeout"][workload_type],
                            min_repeat_ms=DEVICES[device_name]["config_params"]["min_repeat_ms"],
                            enable_cpu_cache_flush=(True if device == 'cpu' else False)
                        )
                    )
                    tuner = autotvm.tuner.XGBTuner(task, feature_type="curve")

                    # Transfer learning if the training log exists
                    if use_autotvm_transfer_learning and os.path.isfile(log_name):
                        tuner.load_history(autotvm.record.load_from_file(log_name))

                    tuner.tune(n_trial=tuning_trials,
                                measure_option=measure_option,
                                callbacks=[autotvm.callback.progress_bar(min(tuning_trials, len(task.config_space))),
                                            autotvm.callback.log_to_file(log_name)])

                # inspect the best config
                # autotvm.record.pick_best(log_name, "logs/autotvm/model/{}/test.log".format(device))
                dispatch_context = autotvm.apply_history_best(log_name)
                best_config = dispatch_context.query(task.target, task.workload)
                print('\nBest config:')
                print(best_config)
                fc.update_all_shapes_from_best_cfg(best_config)
                task.args = autotvm.task.topi_integration.serialize_args(list(fc.make_params(raw=False).values())) # Update task.args (workload) with new shapes
                
                with target:
                    s, arg_bufs = task.instantiate(best_config)
            elif use_auto_scheduler:
                log_name = 'logs/auto_scheduler/layer/{}/{}_fused_{}.json'.format(device, workload_type, workload_name)
                print(log_name)

                # logging
                logging.getLogger('auto_scheduler').setLevel(logging.DEBUG)
                logging.getLogger('auto_scheduler').addHandler(logging.StreamHandler(sys.stdout))

                task = tvm.auto_scheduler.SearchTask(func=get_auto_scheduler_task_x86, args=[parameters], target=target)
                print(task.compute_dag)
                print(task.target)
                print(task.workload_key)

                if not skip_training:
                    # auto_scheduler setting
                    tune_option = auto_scheduler.TuningOptions(
                        num_measure_trials=tuning_trials,
                        measure_callbacks=[auto_scheduler.RecordToFile(log_name)],
                        verbose=2,
                        builder=auto_scheduler.LocalBuilder(),
                        runner=auto_scheduler.RPCRunner(
                            device_name, '0.0.0.0', 9190,
                            number=DEVICES[device_name]["config_params"]["number"],
                            repeat=DEVICES[device_name]["config_params"]["repeat"],
                            timeout=DEVICES[device_name]["config_params"]["timeout"][workload_type],
                            min_repeat_ms=DEVICES[device_name]["config_params"]["min_repeat_ms"],
                            enable_cpu_cache_flush=(True if device == 'cpu' else False)
                        )
                    )
                    task.tune(tune_option)

                best_config = None
                s, arg_bufs = task.apply_best(log_name)
            else:
                raise Exception("Unrecognized option!")

            if not no_print_ir:
                print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs, target_str, name='fused_2')
            if print_src:
                if target_str == 'cuda':
                    print(func.imported_modules[0].get_source())
                else:
                    print(func.get_source('asm')) # assembly code
            if dry_run: # Only print IR and/or source
                return

            # Prepare data
            ref_data = testing.get_fused_conv2d_ref_data(fc, workload_name, best_config, save_data=save_data)

            if export_code:
                # export kernel launch config, e.g. thxyz, blxy, vlen, etc
                output_shape = ref_data[-1].shape
                if use_autotvm:
                    assert (best_config is not None)
                    export_kernel_launch_config(workload_name, output_shape, best_config, target_str)
                if target_str == 'cuda':
                    code = func.imported_modules[0].get_source()
                    write_code(code, 'generated_kernels/gpu/fused/{}.cuh'.format(workload_name))
                else: # CPU
                    code = func.get_source("asm")
                    write_code(code, 'generated_kernels/cpu/fused/{}.asm'.format(workload_name))

                    # func.export_library("benchmark/cpu/kernel.so")
                    # func_sys = tvm.build(s, flatten_params, target_str + " --system-lib", name="fused_2_sys")
                    # func_sys.save("benchmark/cpu/kernel_sys.o")

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
                print('# of incorrect numbers: {}'.format(len(ref_data[-1][d])))
                print(nd_arrays[-1].asnumpy()[d])
                print(ref_data[-1][d])
                print(np.where(d))
            # print("Error rate: {:.2f}%".format((len(d) / len(ref_data[-1]) * 100)))
            print('{}_fused of {} ({}): average running time is {:.2f} us.'.format(workload_type, workload_name, 'NHWC' if target_str == 'cuda' else 'NCHWc', tcost_d * 1e6))
            FLOP = fc.get_FLOP()
            print('FLOP: {}, GFLOPS: {:.2f}.'.format(FLOP, FLOP / tcost_d / 1e9))

    check_device(device_name)
    print("############################################")

if __name__ == '__main__':
    def get_options():
        parser = argparse.ArgumentParser(description="Parses command.")
        parser.add_argument("-n", "--no_print_ir", action="store_true", help="Don't print IR code.")
        parser.add_argument("-s", "--print_src", action="store_true", help="Print source code.")
        parser.add_argument("-y", "--dry_run", action="store_true", help="Dry run.")
        parser.add_argument("-d", "--save_data", action="store_true", help="Save numpy data as npy files.")
        parser.add_argument("-c", "--export_code", action="store_true", help="Export generated kernel code.")
        parser.add_argument("-a", "--use_autotvm", action="store_true", help="AutoTVM for auto tuning.")
        parser.add_argument("-r", "--use_auto_scheduler", action="store_true", help="Use Auto Scheduler.")
        parser.add_argument("-k", "--skip_training", action="store_true", help="Run AutoTVM tuned kernel.")
        parser.add_argument("-l", "--use_autotvm_transfer_learning", action="store_true", help="Load existing tuning log.")
        parser.add_argument("-v", "--device", type=str, default="i7_7700K", help="Device name.")
        parser.add_argument("-t", "--tuning_trials", type=int, default=32, help="Number of AutoTVM/AutoScheduler trials.")
        parser.add_argument("-u", "--unfused", action="store_true", help="Tune separate tasks.")
        options = parser.parse_args()
        return options

    options = get_options()
    workloads = get_workloads()
    # workloads = get_workloads_from_file()

    for workload_type, workload in workloads.items():
        for workload_name, parameters in workload.items():
            print(workload_name, parameters)
            verify_tuning(workload_name,
                            workload_type,
                            parameters,
                            options)
