#!/usr/bin/env bash
pushd /home/ahnyoung/cldnn.dynamic/
DEBUG=1 source ./set.sh
popd


export LD_LIBRARY_PATH=$PATH_OPENVINO_BIN:$PATH_OPENVINO_SRC/temp/tbb/lib
source $PATH_OPENVINO_SRC/temp/opencv_4.5.2_ubuntu20/opencv/setupvars.sh



rm -rf build/CMakeCache.txt

mkdir -p build

cmake -DCMAKE_BUILD_TYPE=Debug -B $PWD/build -S $PWD
cmake --build $PWD/build --config build --parallel 16

