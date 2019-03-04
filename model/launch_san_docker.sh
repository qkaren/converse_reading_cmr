#!/usr/bin/env bash
## This is a script to launch docker

##
SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
IMAGE=allenlao/pytorchv2 # docker image

echo $SCRIPTPATH
export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

nvidia-docker run \
--net host \
--volume $SCRIPTPATH:/san \
--interactive --tty $IMAGE /bin/bash

