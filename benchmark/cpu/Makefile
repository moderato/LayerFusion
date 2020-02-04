CXX := g++
ASXX := as
TARGET := cpu_bench
CXXFLAGS := -std=c++11 -O2 -fPIC \
	-I${TVM_HOME}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_HOME}/3rdparty/dlpack/include
LDFLAGS := -L${TVM_HOME}/build
ASFLAGS := -march=core-avx2
KERNEL ?= ../../generated_kernels/cpu/mv1_2.asm
REPEATITION ?= 1000

all: $(TARGET).cpp
	$(ASXX) $(KERNEL) -o kernel.o
	$(CXX) kernel.o -shared -fPIC -m64 -o kernel.so
	$(CXX) $(CXXFLAGS) \
		$(LDFLAGS) \
		-D REPEATITION=$(REPEATITION) \
		$(TARGET).cpp \
		-o $(TARGET) \
		-ldl -pthread -lcnpy -lz -ltvm_runtime

.phony: clean

clean:
	rm $(TARGET) kernel.o kernel.so || echo -n ""