CXX := g++
ASXX := as
TARGET := cpu_bench
CXXFLAGS := -std=c++11 -O2 -fPIC \
	-I${TVM_HOME}/include \
	-I${DMLC_CORE}/include \
	-I${MKLDNN_HOME}/include \
	-I${TVM_HOME}/3rdparty/dlpack/include \
	-I${VTUNE_HOME}/include
LDFLAGS := \
	-L${TVM_HOME}/build -ltvm_runtime \
	-L${MKLDNN_HOME}/build \
	-L/usr/local/lib \
	-L${VTUNE_HOME}/lib64 \
	-littnotify -ldl -pthread -lcnpy -lz -lmkldnn
ASFLAGS := -march=corei7+fma+avx+avx2+sse3
KERNEL ?= ../../generated_kernels/cpu/mv1_3.asm
REPEATITION ?= 1000

all: $(TARGET).cpp
	$(ASXX) $(ASFLAGS) $(KERNEL) -o kernel.o
	$(CXX) kernel.o -shared -fPIC -o kernel.so
	$(CXX) $(CXXFLAGS) \
		-D REPEATITION=$(REPEATITION) \
		$(TARGET).cpp \
		-o $(TARGET) \
		$(LDFLAGS)

.phony: clean

clean:
	rm $(TARGET) kernel.o kernel.so || echo -n ""