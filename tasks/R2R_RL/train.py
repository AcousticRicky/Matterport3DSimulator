
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince
from env import R2RBatch
from model import EncoderLSTM, AttnDecoderLSTM
from agent import ActorCriticAgent
from test import Evaluation


TRAIN_VOCAB = 'tasks/R2R_RL/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R_RL/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/R2R_RL/results/'
SNAPSHOT_DIR = 'tasks/R2R_RL/snapshots/'
PLOT_DIR = 'tasks/R2R_RL/plots/'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
MAX_INPUT_LENGTH = 80

features = IMAGENET_FEATURES
batch_size = 1
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = 'sample' # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
n_iters = 100000 #if feedback_method == 'teacher' else 20000
model_prefix = 'actercritic_%s_imagenet' % (feedback_method)


def train(train_env, vocab_size, n_iters, log_every=1000, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    agent = ActorCriticAgent(train_env, vocab_size, "", batch_size, max_episode_len)

    data_log = defaultdict(list)
    start = time.time()
    guide_prob = 0.7
    for idx in range(0, n_iters, log_every):
        interval = min(log_every,n_iters-idx)
        iter = idx + interval

        agent.train(interval, guide_prob)


        train_losses = np.array(agent.losses)
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = ''#'guide prob: %.2f' % guide_prob
        #loss_str += ', train loss: %.4f' % train_loss_avg
        # Run validation
        for env_name, (env, evaluator) in val_envs.iteritems():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
            agent.test(0.0)#guide_prob)

            #val_losses = np.array(agent.losses)
            #val_loss_avg = np.average(val_losses)
            #data_log['%s loss' % env_name].append(val_loss_avg)

            agent.write_results()

            score_summary, _ = evaluator.score(agent.results_path)
            #loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            loss_str += ', %s' % (env_name)
            for metric,val in score_summary.iteritems():
                data_log['%s %s' % (env_name,metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ' success: %.2f' % (val)

        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters), iter, float(iter)/n_iters*100, loss_str))
        guide_prob -= 0.01
        guide_prob = max(guide_prob, 0.0)

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def test_submission():
    ''' Train on combined training and validation sets, and generate test submission. '''
  
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train', 'val_seen', 'val_unseen'], tokenizer=tok)
    
    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx, 
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters)

    # Generate test submission
    test_env = R2RBatch(features, batch_size=batch_size, splits=['test'], tokenizer=tok)
    agent = Seq2SeqAgent(test_env, "", encoder, decoder, max_episode_len)
    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, 'test', 20000)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
  
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok)

    # Creat validation environments
    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split], 
                tokenizer=tok), Evaluation([split])) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    train(train_env, len(vocab), n_iters, val_envs=val_envs)


if __name__ == "__main__":
    train_val()
    #test_submission()

