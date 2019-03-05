CRIPTPATH=$( cd $(dirname $0) ; pwd -P )
IMAGE=allenlao/pytorch-allennlp-rtd # docker image
#IMAGE=allenlao/pytorch-allennlp-rt # docker image
#IMAGE=allenlao/pytorch-allennlp-v2 # docker image
#IMAGE=allenlao/pytorch-allennlp # docker image
#IMAGE=allenlao/pytorchv4 # docker image


echo $SCRIPTPATH

# start docker
docker run \
-it --rm \
--net host \
--volume $CRIPTPATH:/san_mrc \
--interactive --tty $IMAGE /bin/bash
# --volume /home/data/:/cs \
# --runtime=nvidia \