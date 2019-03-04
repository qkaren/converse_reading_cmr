#!/usr/bin/env bash
## This is a script to download data

##
DATA_DIR=$(pwd)/data
echo $DATA_DIR

# Download GloVe (Run this when you need to process raw date from scratch.)
# wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $DATA_DIR/glove.840B.300d.zip
# unzip $DATA_DIR/glove.840B.300d.zip -d $DATA_DIR
# rm $DATA_DIR/glove.840B.300d.zip

# Download CoVe
wget https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth -O $DATA_DIR/MT-LSTM.pt

