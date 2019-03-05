#!/bin/bash

mkdir data
mkdir log

# process raw data to json file
CUDA_VISIBLE_DEVICES=0 \
python3 write_raw_to_json.py \
    --data_dir './data/processed/toy' --train_data train.json \
    --dev_data dev.json --test_data test.json\
    --raw_data_dir './data/raw/toy/' \
    --meta './data/reddit_meta.pick'
