import tvm
from tvm import autotvm

def schedule_conv_conv_fused_nhwc_auto(outs, stages, params, device="cuda", bn_relu1=None, bn_relu2=None):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    PaddedInput = stages[1][0]
    if bn_relu1 is not None:
        Inter, InterScaleShift, InterReLU = stages[2]
        IntermediateStage = InterReLU
        F_1, Scale_1, Shift_1 = params[1]
    else:
        Inter = stages[2][0]
        IntermediateStage = Inter
        F_1 = params[1][0]

    hasPaddedInter = False
    if bn_relu2 is not None:
        if 'Padded' in stages[3][0].op.name:
            hasPaddedInter = True
            PaddedInter = stages[3][0]
            Out, OutScaleShift, OutReLU = stages[4]
        else:
            Out, OutScaleShift, OutReLU = stages[3]
        OutputStage = OutReLU
        F_2, Scale_2, Shift_2 = params[2]
    else:
        if 'Padded' in stages[3][0].op.name:
            hasPaddedInter = True
            PaddedInter = stages[3][0]
            Out = stages[4][0]
        else:
            Out = stages[3][0]
        OutputStage = Out
        F_2 = params[2][0]

    # AutoTVM config
    cfg = autotvm.get_config()

    # ######## Input data, weights, BN, etc
    s[PaddedInput].compute_inline()
    PaddedSharedInput = s.cache_read(PaddedInput, "shared", [Inter])
    FL_1 = s.cache_read(F_1, "local", [Inter])
    FS_2 = s.cache_read(F_2, "shared", [Out])
    s[IntermediateStage].set_scope("shared")

    if bn_relu1 is not None:
        s[InterScaleShift].compute_inline()
        s[Inter].set_scope("local")
        ScaleL_1 = s.cache_read(Scale_1, "local", [InterScaleShift])
        ShiftL_1 = s.cache_read(Shift_1, "local", [InterScaleShift])
        DepthwiseLocalAccumulator = Inter
    else:
        DepthwiseLocalAccumulator = s.cache_write(IntermediateStage, "local")

    if bn_relu2 is not None:
        s[OutScaleShift].compute_inline()
        s[Out].set_scope("local")
        ScaleL_2 = s.cache_read(Scale_2, "local", [OutScaleShift])
        ShiftL_2 = s.cache_read(Shift_2, "local", [OutScaleShift])
        OL = Out
    else:
        OL = s.cache_write(OutputStage, "local")

    # ######## Blocks, threads and vthreads
    if device == "cuda":
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        thread_y = tvm.thread_axis("threadIdx.y")
        thread_z = tvm.thread_axis("threadIdx.z")
        vthread_x = tvm.thread_axis("vthread", name="vthread_x")
        vthread_y = tvm.thread_axis("vthread", name="vthread_y")
        vthread_z = tvm.thread_axis("vthread", name="vthread_z")

    # Vectorization
    vec = [4] if device == "cuda" else [2, 4, 8, 16, 32, 64]
    cfg.define_knob("vectorization", candidate=vec)

    return s
