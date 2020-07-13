CC= gcc
CFLAGS= -O3 -fPIC -std=c++11 -fopenmp
LDFLAGS= -shared
SOURCES = batch_reduce_plus_init.cc

INC=-I$(LIBXSMM_HOME)/include
LIBS = $(LIBXSMM_HOME)/lib/libxsmm.a $(LIBXSMM_HOME)/lib/libxsmmext.a \
       $(LIBXSMM_HOME)/lib/libxsmmnoblas.a $(LIBXSMM_HOME)/lib/libxsmmgen.a \
       $(LIBXSMM_HOME)/lib/libxsmmf.a
TARGET= libxsmm_wrapper.so

all:
	$(CC) $(INC) $(CFLAGS) -fPIC $(SOURCES) $(LIBS) -o $(TARGET) $(LDFLAGS)
clean:
	rm $(TARGET)
