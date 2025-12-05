#!/bin/bash
nvidia_version=`cat /proc/driver/nvidia/version |grep 'NVRM version:'| grep -oE "Kernel Module\s+[0-9.]+"| awk {'print $3'}`
nvidia_major_version=`echo $nvidia_version |sed "s/\..*//"`
driver_filename="NVIDIA-Linux-x86_64-$nvidia_version.run"
driver_url="http://us.download.nvidia.com/XFree86/Linux-x86_64/$nvidia_version/$driver_filename"
tesla_driver_url="http://us.download.nvidia.com/tesla/$nvidia_version/$driver_filename"
s3_driver_url="http://ai2-vision-nvidia.s3-us-west-2.amazonaws.com/$driver_filename"
wget $driver_url -P /tmp/ || wget $tesla_driver_url -P /tmp/ || wget $s3_driver_url -P /tmp/

if [ $?  -eq 0 ]; then
    sh /tmp/$driver_filename -s --no-kernel-module
else
    echo "Error trying to install nvidia driver for $nvidia_version"
    exit 1
fi;