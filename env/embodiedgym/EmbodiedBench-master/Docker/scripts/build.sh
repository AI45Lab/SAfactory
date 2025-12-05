#!/bin/bash
if [ ! -e /proc/driver/nvidia/version ]; then
    echo "Error: Nvidia driver not found at /proc/driver/nvidia/version; Please ensure you have an Nvidia GPU device and appropriate drivers are installed."
    exit 1;
fi;

if  ! type "docker" 2> /dev/null > /dev/null ; then
    echo "Error: docker not found. Please install docker to complete the build. "
    exit 1
fi;

NVIDIA_VERSION=`cat /proc/driver/nvidia/version | grep 'NVRM version:'| grep -oE "Kernel Module\s+[0-9.]+"| awk {'print $3'}` 
NVIDIA_MAJOR=`echo $NVIDIA_VERSION | tr "." "\n" | head -1  | tr -d "\n"`
NVIDIA_MINOR=`echo $NVIDIA_VERSION | tr "." "\n" | head -2  | tail -1| tr -d "\n"`

# https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
if (( $NVIDIA_MAJOR >= 470 && $NVIDIA_MINOR >= 42 )); then
    CUDA_VERSION=11.4.2
elif (( $NVIDIA_MAJOR >= 465 && $NVIDIA_MINOR >= 19 )); then
    CUDA_VERSION=11.3.1
elif (( $NVIDIA_MAJOR >= 460 && $NVIDIA_MINOR >= 27 )); then
    CUDA_VERSION=11.2.2
elif (( $NVIDIA_MAJOR >= 450 && $NVIDIA_MINOR >= 80 )); then
    CUDA_VERSION=11.1
elif (( $NVIDIA_MAJOR >= 450 && $NVIDIA_MINOR >= 36 )); then
    CUDA_VERSION=11.0
elif (( $NVIDIA_MAJOR >= 440 && $NVIDIA_MINOR >= 33 )); then
    CUDA_VERSION=10.2
elif (( $NVIDIA_MAJOR >= 418 && $NVIDIA_MINOR >= 39 )); then
    CUDA_VERSION=10.1
elif (( $NVIDIA_MAJOR >= 410 && $NVIDIA_MINOR >= 48 )); then
    CUDA_VERSION=10.0
elif (( $NVIDIA_MAJOR >= 396 && $NVIDIA_MINOR >= 26 )); then
    CUDA_VERSION=9.2
elif (( $NVIDIA_MAJOR >= 390 && $NVIDIA_MINOR >= 46 )); then
    CUDA_VERSION=9.1
elif (( $NVIDIA_MAJOR >= 384 && $NVIDIA_MINOR >= 81 )); then
    CUDA_VERSION=9.0
else
    echo "No valid CUDA version found for nvidia driver $NVIDIA_VERSION"
    exit 1
fi

echo $NVIDIA_VERSION
echo $CUDA_VERSION

# if (id -nG | grep -qw "docker") || [ "$USER" == "root" ]; then
#     echo "Building Docker container with CUDA Version: $CUDA_VERSION, NVIDIA Driver: $NVIDIA_VERSION"
#     docker build  --build-arg NVIDIA_VERSION=$NVIDIA_VERSION --build-arg CUDA_VERSION=$CUDA_VERSION  -t ai2thor-docker:latest .
# else
#     echo "Error: Unable to run build.sh. Please use sudo to run build.sh or add $USER to the docker group."
#     exit 1
# fi
