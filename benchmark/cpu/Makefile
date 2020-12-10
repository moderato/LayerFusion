CXX := icpc
ASXX := as
TARGET := cpu_bench
CXXFLAGS := -std=c++14 -O2 -fPIC \
	-I${TVM_HOME}/include \
	-I${TVM_HOME}/3rdparty/dlpack/include \
	-I${DMLC_CORE}/include \
	-I${PCM_HOME} \
	-I${MKLDNN_HOME}/include
LDFLAGS := \
	-L${TVM_HOME}/build \
	${PCM_HOME}/libPCM.a \
	-L${MKLDNN_HOME}/build \
	-L/usr/local/lib \
	-ltvm_runtime -ldl -lpthread -lcnpy -lz -lmkldnn
# -I${VTUNE_HOME}/include ${VTUNE_HOME}/lib64/libittnotify.a -L${VTUNE_HOME}/lib64 -littnotify 
ASFLAGS ?= -march=corei7+fma+avx+avx2+sse3+avx512f+avx512cd+avx512vl+avx512dq # Put all potentially needed extensions here
KERNEL_FUSED ?= ../../generated_kernels/cpu/fused/mv1_3.asm
KERNEL_UNFUSED_1 ?= ../../generated_kernels/cpu/unfused/mv1_3_1.asm
KERNEL_UNFUSED_2 ?= ../../generated_kernels/cpu/unfused/mv1_3_2.asm
REPEATITION ?= 1000
ENABLE_PCM ?= 0
LAYER_1 ?= 0
LAYER_2 ?= 0

all: $(TARGET).cpp
	$(ASXX) $(ASFLAGS) $(KERNEL_FUSED) -o kernel_fused.o
	$(ASXX) $(ASFLAGS) $(KERNEL_UNFUSED_1) -o kernel_unfused_1.o
	$(ASXX) $(ASFLAGS) $(KERNEL_UNFUSED_2) -o kernel_unfused_2.o
	$(CXX) kernel_fused.o kernel_unfused_1.o kernel_unfused_2.o -shared -fPIC -o kernel.so
	$(CXX) $(CXXFLAGS) \
		-D REPEATITION=$(REPEATITION) \
		-D ENABLE_PCM=$(ENABLE_PCM) \
		-D LAYER_1=$(LAYER_1) \
		-D LAYER_2=$(LAYER_2) \
		$(TARGET).cpp \
		-o $(TARGET) \
		$(LDFLAGS)

.phony: clean

clean:
	rm $(TARGET) kernel_fused.o kernel_unfused_1.o kernel_unfused_2.o kernel.so || echo -n ""