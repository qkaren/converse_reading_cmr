import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import msgpack
import numpy as np
from train_util import *
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from config import set_args
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger

args = set_args()
# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set environment
set_environment(args.seed, args.cuda)
# setup logger
logger =  create_logger(__name__, to_disk=True, log_file=args.log_file)


def main():
    logger.info('Launching the SAN')
    opt = vars(args)
    logger.info(opt)
    embedding, opt, vocab = load_meta(
        opt, os.path.join(args.data_dir, args.meta))
    max_doc = opt['max_doc']
    smooth = opt['smooth']
    is_rep = opt['is_rep']
    eval_step = opt['eval_step']
    curve_file = opt['curve_file']

    training_step = 0
    cur_eval_step = 1

    checkpoint_path = args.resume
    if checkpoint_path == '':
        if not args.if_train:
            print ('checkpoint path can not be empty during testing...')
            exit()
        model = DocReaderModel(opt, embedding)
    else:
        state_dict = torch.load(checkpoint_path)["state_dict"]
        model = DocReaderModel(opt, embedding, state_dict)
    model.setup_eval_embed(embedding)
    logger.info("Total number of params: {}".format(model.total_param))
    
    if args.cuda:
        model.cuda()

    pred_output_path = os.path.join(model_dir, 'pred_output')
    if not os.path.exists(pred_output_path):
        os.makedirs(pred_output_path)
    full_output_path = os.path.join(model_dir, 'full_output_path')
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)


    if args.if_train:
        logger.info('Loading training data')
        train_data = BatchGen(os.path.join(args.data_dir, args.train_data),
                              batch_size=args.batch_size,
                              gpu=args.cuda, doc_maxlen=max_doc)
        logger.info('Loading dev data')
        dev_data = BatchGen(os.path.join(args.data_dir, args.dev_data),
                            batch_size=8,
                            gpu=args.cuda, is_train=False, doc_maxlen=max_doc)
        curve_file = os.path.join(model_dir, curve_file)
        full_path = os.path.join(args.data_dir, args.dev_full)
        pred_output = os.path.join(pred_output_path, str(model.updates)) + '.txt'
        full_output = os.path.join(full_output_path, str(model.updates)) + '_full.txt'


        for epoch in range(0, args.epoches):
            logger.warning('At epoch {}'.format(epoch))
            train_data.reset()
            start = datetime.now()
            for i, batch in enumerate(train_data):
                training_step += 1
                model.update(batch, smooth, is_rep)
                if (i + 1) % args.log_per_updates == 0:
                    logger.info('updates[{0:6}] train: loss[{1:.5f}]'
                                ' ppl[{2:.5f}] remaining[{3}]'.format(
                       model.updates,
                       model.train_loss.avg,
                       np.exp(model.train_loss.avg),
                       str((datetime.now() - start) / (i + 1) * (len(train_data) - i - 1)).split('.')[0]))

                    # setting up scheduler
                    if model.scheduler is not None:
                        if opt['scheduler_type'] == 'rop':
                            model.scheduler.step(model.train_loss.avg, epoch=epoch)
                        else:
                            model.scheduler.step()

                dev_loss = 0.0
                if (training_step) == cur_eval_step:
                    print('evaluating_step is {} ....'.format(training_step))
                    bleu, bleu_fact, diver_uni, diver_bi = check(
                        model, dev_data, vocab, full_path,
                        pred_output, full_output)
                    dev_loss = eval_test_loss(model, dev_data).data.cpu().numpy()[0]
                    # dev_loss = dev_loss.data.cpu().numpy()[0]
                    logger.info('updates[{0:6}] train: loss[{1:.5f}] ppl[{2:.5f}]\n'
                                'dev: loss[{3:.5f}] ppl[{4:.5f}]'.format(
                      model.updates,
                      model.train_loss.avg,
                      np.exp(model.train_loss.avg),
                      dev_loss,
                      np.exp(dev_loss)))
                    print('{0},{1:.5f},{2:.5f},{3:.5f},{4:.5f},'
                          '{5:.5f},{6:.5f},{7:.5f},{8:.5f}\n'.format(
                        model.updates, model.train_loss.avg, np.exp(model.train_loss.avg),
                        dev_loss, np.exp(dev_loss), float(bleu), float(diver_uni),
                        float(diver_bi), float(bleu_fact)))
                    with open(curve_file, 'a+') as fout_dev:
                        fout_dev.write('{0},{1:.5f},{2:.5f},{3:.5f},{4:.5f},'
                                       '{5:.5f},{6:.5f},{7:.5f},{8:.5f}\n'.format(
                        model.updates, model.train_loss.avg,
                        np.exp(model.train_loss.avg), dev_loss, np.exp(dev_loss),
                        float(bleu), float(diver_uni), float(diver_bi), float(bleu_fact)))

                    if cur_eval_step == 1:
                        cur_eval_step = cur_eval_step -1
                    cur_eval_step += eval_step

                if (i + 1) % (args.log_per_updates * 50) == 0:
                    logger.info('have saved model as checkpoint_step_{0}_{1:.5f}.pt'
                                .format(model.updates, np.exp(dev_loss)))
                    model_file = os.path.join(model_dir, 'checkpoint_step_{0}_{1:.5f}.pt'
                                              .format(model.updates, np.exp(dev_loss)))
                    model.save(model_file, epoch)

            #save
            dev_loss = eval_test_loss(model, dev_data)
            dev_loss = dev_loss.data.cpu().numpy()[0]
            logger.info('have saved model as checkpoint_epoch_{0}_{1}_{2:.5f}.pt'
                        .format(epoch, args.learning_rate, np.exp(dev_loss)))
            model_file = os.path.join(model_dir, 'checkpoint_epoch_{0}_{1}_{2:.5f}.pt'
                                      .format(epoch, args.learning_rate,np.exp(dev_loss)))
            model.save(model_file, epoch)

    else:
        logger.info('Loading evaluation data')
        checkpoint_path = args.resume
        state_dict = torch.load(checkpoint_path)["state_dict"]
        model = DocReaderModel(opt, embedding, state_dict)
        model.setup_eval_embed(embedding)
        logger.info("Total number of params: {}".format(model.total_param))
        if args.cuda:
            model.cuda()

        def _eval_output(file_path=args.dev_data, full_path=args.dev_full, test_type='dev'):
            data = BatchGen(os.path.join(args.data_dir, file_path),
                                batch_size=args.batch_size,
                                gpu=args.cuda, is_train=False)
            print(len(data))
            full_path = os.path.join(args.data_dir, full_path)
            pred_output_path = os.path.join('./output/', test_type) + '/'
            full_output_path = os.path.join('./full_output/', test_type) + '/'
            if not os.path.exists(pred_output_path):
                os.makedirs(pred_output_path)
            if not os.path.exists(full_output_path):
                os.makedirs(full_output_path)
            t = args.test_output
            pred_output = pred_output_path + t + '.txt'
            full_output = full_output_path + t + '_full.txt'
            bleu, bleu_fact, diver_uni, diver_bi = \
            check(model, data, vocab, full_path, pred_output, full_output)
            _loss = eval_test_loss(model, data)
            _loss = _loss.data.cpu().numpy()[0]
            logger.info('dev loss[{0:.5f}] ppl[{1:.5f}]'.format(
               _loss,
               np.exp(_loss)))
            print('{0},{1:.5f},{2:.5f},{3:.5f},{4:.5f},{5:.5f},'
                  '{6:.5f},{7:.5f},{8:.5f}\n'.format(
                   model.updates, model.train_loss.avg, np.exp(model.train_loss.avg), _loss, np.exp(_loss),
                   float(bleu), float(diver_uni), float(diver_bi), float(bleu_fact)))

        print('test result is:')
        _eval_output(args.test_data, args.test_full, 'test')

if __name__ == '__main__':
    main()

