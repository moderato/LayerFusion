CXX := clang++
ASXX := as
TARGET := cpu_bench
CXXFLAGS := -std=c++11 -O0 -g
ASFLAGS := -march=core-avx2

all: $(TARGET).cpp
	$(ASXX) ../../generated_kernels/cpu/mv1_6.s -o kernel.o
	$(CXX) $(CXXFLAGS) $(TARGET).cpp kernel.o -o $(TARGET) -lcnpy -lz

.phony: clean

clean:
	rm $(TARGET) || echo -n ""