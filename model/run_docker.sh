CRIPTPATH=$( cd $(dirname $0) ; pwd -P )
IMAGE=allenlao/pytorch-allennlp-rt # docker image

echo $SCRIPTPATH

# start docker
docker run \
-it --rm \
--net host \
--volume $CRIPTPATH:/san \
--interactive --tty $IMAGE /bin/bash
