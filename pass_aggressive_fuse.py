import numpy as np
import tvm
from tvm import te, autotvm
from tvm.contrib.util import tempdir
from pprint import pprint
from tvm.relay.testing.mobilenet import separable_conv_block, get_workload
import tvm.relay as relay
import tvm.relay.testing.layers as layers
import tvm.contrib.graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime
from fusion_composer import FusionComposer
from schedules.schedule_utils import gpu_schedules as sch

# target = "llvm -mcpu=core-avx2"
target = tvm.target.cuda()
log_filename='logs/autotvm/model/gpu/test.log'

def print_ir(mod, info, is_before=True):
    """Print the name of the pass, the IR, only before passes execute."""
    if is_before:
        pass

def example():
    data = relay.var("data", shape=(1, 112, 112, 32))
    body = separable_conv_block(data, 'separable_conv_block_1', 32, 64, layout="NHWC")
    return relay.Function(relay.analysis.free_vars(body), body)

model_mod, model_params = get_workload(batch_size=1, dtype="float32", image_shape=(224, 224, 3), layout='NHWC')
params = {}
for k, v in model_params.items():
    if 'separable_conv_block_1' in k:
        params[k] = v

f = example()
mod = tvm.IRModule.from_expr(f)
mod = relay.transform.InferType()(mod)

tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
pprint(tasks)

# compile kernels with history best records
with autotvm.apply_history_best(log_filename):
    print("############### Compile... ###############")
    # disabled_pass: to prevent 'multiply' from being eliminated
    # TODO: Fix this
    with tvm.transform.PassContext(opt_level=5, trace=print_ir, disabled_pass=['ForwardFoldScaleAxis','BackwardFoldScaleAxis']):
        # """
        # build = optimize + generate_code
        # build / generate_code: return mod
        # optimize: return mod and params
        # """

        # # Merged
        # graph_factory = relay.build_module.build(mod, target=target, params=params)

        # Fused
        mod, params = relay.build_module.optimize(mod, target=target, params=params) # This step finish processing the relay graph
        # print(mod)
        graph_factory = relay.build_module.generate_code(mod, target=target, params=params)

    graph, lib, params = graph_factory.graph_json, graph_factory.lib, graph_factory.params
    # print(graph)

    # export library
    tmp = tempdir()
    filename = "net.tar"
    lib.export_library(tmp.relpath(filename))

    # load parameters
    ctx = tvm.context(str(target), 0)
    # module = runtime.create(graph, lib, ctx)
    module = debug_runtime.create(graph, lib, ctx, dump_root="/tmp/tvmdbg")
    data_tvm = tvm.nd.array((np.random.uniform(size=(1, 32, 112, 112))).astype("float32"))
    module.set_input('data', data_tvm)
    module.set_input(**params)
    module.run()

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))

# @relay.transform.function_pass(opt_level=1)
# class CustomPipeline:
#     """Simple test function to replace one argument to another."""

#     def __init__(self, multiplier):
#         self.multiplier = multiplier

#     # This function can define a pass.
#     def transform_function(self, func, mod, ctx):
#         class ReplaceConstant(tvm.relay.ExprMutator):
#             def visit_constant(self, c):
#                 # return relay.multiply(obj.multiplier, c)
#                 return c
#             # def visit_op(self, op):
#             #     # print(dir(op))
#             #     return op
#             # def visit_function(self, f):
#             #     print("&&&&&&")
#             #     pprint(f.body.attrs)
#             #     print("******************************")
#             #     pprint(f.body.op)
#             #     print("******************************")
#             #     pprint(f.body.span)
#             #     print("******************************")
#             #     pprint(f.body.type_args)
#             #     return f
#         return ReplaceConstant().visit(func)