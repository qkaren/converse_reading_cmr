#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python3  train.py \
--data_dir './data/' --train_data train.json --dev_data dev.json \
--dev_full dev.full --test_data test.json  --test_full test.full \
--covec_path 'data/MT-LSTM.pt' \
--batch_size 32 \
--eval_step 5000 \
--optimizer adam  --learning_rate 0.0005 \
--temperature 0.8 \
--if_train 0 \
--contextual_cell_type gru \
--contextual_num_layers 1 \
--msum_cell_type gru \
--msum_num_layers 1 \
--deep_att_lexicon_input_on \
--no_pos --no_ner --no_feat \
--pwnn_on \
--no_lr_scheduler  \
--scheduler_type rop \
--lr_gamma 0.5 \
--max_doc 501 \
--log_per_updates 50 \
--decoding_bleu_normalize \
--decoding weight \
--decoding_topk 20 \
--weight_type 'nist' \
--decoding_bleu_lambda 0  \
--test_output "nist_san_20_0.8" \
--resume './checkpoint/san_checkpoint.pt'

