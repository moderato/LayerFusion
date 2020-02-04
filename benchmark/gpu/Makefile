CXX := nvcc
TARGET := gpu_bench
CUDA_PATH := /usr/local/cuda
CUDNN_PATH := $(CUDA_PATH)
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_35 -std=c++11 -O2
KERNEL ?= ../../generated_kernels/gpu/mv1_1.cuh
REPEATITION ?= 1000

all: $(TARGET).cu
	$(CXX) $(CXXFLAGS) \
		$(HEADERS) \
		$(LIBS) \
		-include $(KERNEL) \
		-D REPEATITION=$(REPEATITION) \
		$(TARGET).cu \
		-o $(TARGET) \
		-lcnpy -lz -lcudnn

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
