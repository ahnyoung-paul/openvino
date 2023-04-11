#!/usr/bin/env bash

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

usage() {
    echo "Build OpenVINO Runtime samples"
    echo
    echo "Options:"
    echo "  -h                       Print the help message"
    echo "  -b SAMPLE_BUILD_DIR      Specify the sample build directory"
    echo "  -i SAMPLE_INSTALL_DIR    Specify the sample install directory"
    echo
    exit 1
}

samples_type="$(basename "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
build_dir="$HOME/openvino_${samples_type}_samples_build"
sample_install_dir=""

# parse command line options
while [[ $# -gt 0 ]]
do
case "$1" in
    -b | --build_dir)
    build_dir="$2"
    shift
    ;;
    -i | --install_dir)
    sample_install_dir="$2"
    shift
    ;;
    -h | --help)
    usage
    ;;
    *)
    echo "Unrecognized option specified $1"
    usage
    ;;
esac
shift
done

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ${LINENO}' ERR

SAMPLES_PATH="$( cd "$( dirname "$(realpath "${BASH_SOURCE[0]}")" )" && pwd )"
printf "\nSetting environment variables for building samples...\n"

export OpenVINO_DIR=/home/ahnyoung/cldnn.dynamic/openvino/build_Debug
export OpenCV_DIR=/home/ahnyoung/cldnn.dynamic/openvino/temp/opencv_4.5.2_ubuntu20/opencv/cmake
export LD_LIBRARY_PATH=$PATH_OPENVINO_BIN:$PATH_OPENVINO_SRC/temp/tbb/lib
source $PATH_OPENVINO_SRC/temp/opencv_4.5.2_ubuntu20/opencv/setupvars.sh

# CentOS 7 has two packages: cmake of version 2.8 and cmake3. install_openvino_dependencies.sh installs cmake3
if command -v cmake3 &>/dev/null; then
    CMAKE_EXEC=cmake3
elif command -v cmake &>/dev/null; then
    CMAKE_EXEC=cmake
else
    printf "\n\nCMAKE is not installed. It is required to build OpenVINO Runtime samples. Please install it. \n\n"
    exit 1
fi

OS_PATH=$(uname -m)
NUM_THREADS=2

if [ "$OS_PATH" == "x86_64" ]; then
  OS_PATH="intel64"
  NUM_THREADS=8
fi

if [ -e "$build_dir/CMakeCache.txt" ]; then
  rm -rf "$build_dir/CMakeCache.txt"
fi

echo "SAMPLES_PATH: $SAMPLES_PATH"

mkdir -p "$build_dir"
$CMAKE_EXEC -DCMAKE_BUILD_TYPE=Debug -S "/home/ahnyoung/cldnn.dynamic/openvino/samples/cpp/" -B "$build_dir" -DOpenCV_DIR=$PATH_OPENCV/cmake
# $CMAKE_EXEC --build "$build_dir" --config Debug --parallel $NUM_THREADS
pushd $build_dir
make debug_CVS_107969 -j
popd

# ./build_samples.sh -b ./debug_CVS_107969/build/