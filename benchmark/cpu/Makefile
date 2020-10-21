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
ASFLAGS ?= -march=corei7+fma+avx+avx2+sse3
KERNEL ?= ../../generated_kernels/cpu/mv1_3.asm
REPEATITION ?= 1000
ENABLE_PCM ?= 0

all: $(TARGET).cpp
	$(ASXX) $(ASFLAGS) $(KERNEL) -o kernel.o
	$(CXX) kernel.o -shared -fPIC -o kernel.so
	$(CXX) $(CXXFLAGS) \
		-D REPEATITION=$(REPEATITION) \
		-D ENABLE_PCM=$(ENABLE_PCM) \
		$(TARGET).cpp \
		-o $(TARGET) \
		$(LDFLAGS)

.phony: clean

clean:
	rm $(TARGET) kernel.o kernel.so || echo -n ""