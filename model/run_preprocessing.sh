#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python3 write_raw_to_json.py \
    --data_dir './data' --train_data train.json \
    --dev_data dev.json --test_data test.json\
    --raw_data_dir './data_processing/raw_data/' \
    --log_file './log/data_processing'

