FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
RUN apt-get update

# General
RUN apt-get install -y wget nano gcc make git lsb-release software-properties-common
RUN apt-get install -y python3 python3-dev python3-setuptools python3-pip
RUN apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev tmux numactl bc vim

# Tmux
RUN echo 'set -g mouse on\n' \
            'set-option -g history-limit 50000\n' >> ${HOME}/.tmux.conf

# ICC & MKL
RUN cd /tmp && \
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
RUN sh -c 'echo deb https://apt.repos.intel.com/oneapi all main > /etc/apt/sources.list.d/intel-oneapi.list' && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --allow-unauthenticated intel-oneapi-icc
RUN echo 'export LD_LIBRARY_PATH=/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH\n' \
            'source /opt/intel/bin/compilervars.sh intel64\n' >> ${HOME}/.bashrc
RUN sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && \
    apt-get update && \
    apt-get install -y --allow-unauthenticated intel-mkl-64bit-2020.3-111
RUN echo 'source /opt/intel/oneapi/setvars.sh >& /dev/null\n' >> ${HOME}/.bashrc

# CMake
RUN mkdir -p ${HOME}/Documents && \
    cd ${HOME}/Documents && \
    wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4.tar.gz && \
    tar -xzvf cmake-3.18.4.tar.gz && \
    cd cmake-3.18.4 && \
    ./bootstrap -- -DCMAKE_USE_OPENSSL=OFF && \
    make -j16 && \
    make install && \
    rm ../cmake-3.18.4.tar.gz

# LLVM
RUN cd ${HOME}/Documents && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 9
RUN ln -sf /usr/bin/llvm-config-9 /usr/bin/llvm-config

# CUDA
RUN echo 'export PATH="${PATH}:/usr/local/cuda/bin"\n' \
            'if [[ "${LD_LIBRARY_PATH}" != "" ]];\n' \
            'then\n' \
            '  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}\n' \
            'else\n' \
            '  export LD_LIBRARY_PATH=/usr/local/cuda/lib64\n' \
            'fi\n' \
            'export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda\n' \
            'export CUDA_HOME=/usr/local/cuda\n' \
            'export CUDA_PATH=/usr/local/cuda\n' \
            'export PATH="${PATH}:/usr/local/cuda/include"\n' >> ${HOME}/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# SDE
RUN cd ${HOME}/Documents && \
    wget -q https://software.intel.com/content/dam/develop/external/us/en/documents/sde-external-8.56.0-2020-07-05-lin.tar.bz2 && \
    tar -xf sde-external-8.56.0-2020-07-05-lin.tar.bz2 && \
    mv sde-external-8.56.0-2020-07-05-lin sde && \
    export PATH=${PATH}:"${HOME}/sde" >> ${HOME}/.bashrc && \
    rm sde-external-8.56.0-2020-07-05-lin.tar.bz2

# PCM
RUN cd ${HOME}/Documents && \
    git clone https://github.com/opcm/pcm.git && \
    cd pcm && \
    make -j16 && \
    make install

# LIBXSMM
RUN export avx_option="$( avx=`grep -c avx /proc/cpuinfo`; \
                          avx2=`grep -c avx2 /proc/cpuinfo`; \
                          avx512=`grep -c avx512 /proc/cpuinfo`; \
                          if [ $avx512 -gt 0 ]; \
                          then \
                            echo 3; \
                          elif [ $avx2 -gt 0 ]; \
                          then \
                            echo 2; \
                          elif [ $avx -gt 0 ]; \
                          then \
                            echo 1; \
                          else \
                            echo 0; \
                          fi )"
RUN cd ${HOME}/Documents && \
    git clone https://github.com/hfp/libxsmm.git && \
    cd libxsmm && \
    make -j16 INTRINSICS=1 AVX=$avx_option

# MKLDNN
RUN cd ${HOME}/Documents && \
    git clone https://github.com/oneapi-src/oneDNN.git mkl-dnn && \
    mkdir -p mkl-dnn/build && \
    cd mkl-dnn/build && \
    cmake .. && \
    make -j16 && \
    make install

# cnpy
RUN cd ${HOME}/Documents && \
    git clone https://github.com/rogersce/cnpy.git && \
    mkdir -p cnpy/build && \
    cd cnpy/build && \
    cmake .. && \
    make -j16 && \
    make install

# TVM
RUN pip3 install --upgrade pip && \
    pip3 install --user numpy decorator attrs tornado psutil xgboost cython typed_ast pytest
RUN cd ${HOME}/Documents && \
    git clone --recursive https://github.com/moderato/incubator-tvm tvm && \
    mkdir -p tvm/build && \
    cd tvm/build && \
    cp ../cmake/config.cmake . && \
    cmake .. && \
    make -j16 && \
    cd .. && \
    make cython3
RUN echo 'export TVM_HOME=${HOME}/Documents/tvm' >> ${HOME}/.bashrc && \
    echo 'export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}' >> ${HOME}/.bashrc && \
    echo 'alias initialize="cd ~/Documents/LayerFusion && source ./init_vars.sh"' >> ${HOME}/.bashrc && \
    echo 'alias attach="tmux attach -t"' >> ${HOME}/.bashrc
