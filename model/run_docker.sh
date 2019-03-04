SCRIPTPATH="./"

IMAGE=allenlao/pytorch-mt-dnn:v0.41

 

 

docker run \
-it --rm \
--net host \
--volume $SCRIPTPATH:/san \
--interactive --tty $IMAGE /bin/bash
