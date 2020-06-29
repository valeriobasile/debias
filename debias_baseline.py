#!/usr/bin/env python

import os
import numpy as np
import codecs
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tqdm import tqdm
import pickle
import csv
import keras
from keras_radam import RAdam
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
from tensorflow.python.keras import backend as K
import tensorflow as tf
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.matutils import sparse2full
from pprint import pprint
from keras_self_attention import SeqSelfAttention
from numpy.random import seed
import matplotlib.pyplot as plt
from keras.models import load_model
from keras_bert import get_custom_objects
import sys

SEQ_LEN = 100
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-6
SEED = eval(sys.argv[1])
TRAIN_SET = sys.argv[2]
LANGUAGE = sys.argv[3]
DATASETS ["abuseval","ami_evalita","ami_ibereval","davidson","founta","hateval","hateval_mig","hateval_mis","irony","offenseval","waseem"]
seed(SEED)

bert_models = {
    'en': {
        'model': 'uncased_L-12_H-768_A-12',
        'config': 'bert_config.json',
        'checkpoint': 'bert_model.ckpt',
        'vocab': 'vocab.txt'
    },
    'it': {
        'model': 'alberto_uncased_L-12_H-768_A-12_italian_ckpt',
        'config': 'config.json',
        'checkpoint': 'alberto_model.ckpt',
        'vocab': 'vocab.txt'
    },
    'es': {
        'model': 'bert-base-spanish-wwm-uncased',
        'config': 'config.json',
        'checkpoint': 'model.ckpt-2000000',
        'vocab': 'vocab.txt'
    }
}

bert_model = bert_models[LANGUAGE]

base_path = '.'
data_path = os.path.join(base_path, 'data/multilingual_binary/{0}'.format(language))
model_path = os.path.join(base_path, 'bert-models')
output_path = os.path.join(base_path, 'results.csv')
pretrained_path = os.path.join(model_path, 'uncased_L-12_H-768_A-12')
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

train_path = os.path.join(data_path, '{0}/train.csv'.format(TRAIN_SET))
dev_path = os.path.join(data_path, '{0}/dev.csv'.format(TRAIN_SET))

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)

def load_data(path):
    global tokenizer
    indices, sentiments = [], []
    sentences = []
    instance_ids = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids, segments = tokenizer.encode(row['text'], max_len=SEQ_LEN)
            indices.append(ids)
            sentiments.append(eval(row['label']))
            tokens = tokenizer.tokenize(row['text'])
            sentences.append(tokens[1:-1])
            instance_ids.append(row['id'])

    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(sentiments), instance_ids, sentences

train_x, train_y, train_ids, train_sentences = load_data(train_path)
dev_x, dev_y, dev_ids, dev_sentences = load_data(dev_path)

def evaluate(predicts, gold):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        hits = 0
        for c in range(len(np.unique(gold))):
            for g, p in zip(gold, predicts):
                if g == p:
                    hits += 1
                    if p==c:
                        tp+=1
                    else:
                        tn+=1
                else:
                    if p==c:
                        fp+=1
                    else:
                        fn+=1
        if tp+fp == 0:
            microp = 0.0
        else:
            microp = tp/(tp+fp)
        if tp+fn == 0:
            micror = 0.0
        else:
            micror = tp/(tp+fn)
        if microp+micror == 0.0:
            microf1 = 0.0
        else:
            microf1 = (2*microp*micror)/(microp+micror)
        accuracy = (hits/2)/gold.shape[0]
        report = classification_report(gold, predicts, output_dict=True)
        df = pd.DataFrame(report)
        eval_values = []
        for l in [str(x) for x in range(len(np.unique(gold)))]:
            if not l in df:
                eval_values.extend([0,0,0])
            else:
                eval_values.extend([
                    df[l]['precision'],
                    df[l]['recall'],
                    df[l]['f1-score']
                ])
        eval_values.extend([
                    df['macro avg']['precision'],
                    df['macro avg']['recall'],
                    df['macro avg']['f1-score']
        ])
        #eval_values.extend([microp, micror, microf1])
        eval_values.extend([accuracy])
        return eval_values



model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)

inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(1, activation="sigmoid")(dense)

model = keras.models.Model(inputs, outputs)

model.compile(
    Adam(lr=LR),
    loss="binary_crossentropy",
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit(
    train_x,
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=True,
    callbacks=[es, mc],
    validation_data=(dev_x, dev_y)
    )

obj = get_custom_objects()
model = load_model('best_model.h5', custom_objects=obj)

predicts = model.predict(dev_x, verbose=False)
pred = predicts
pred[pred>=.5]=1
pred[pred<.5]=0
scores = evaluate(pred, dev_y)

output = ",".join(['baseline', LANGUAGE, TRAIN_SET, TRAIN_SET, 'dev', str(LR), "-", "-", str(SEED)]+['{0:.4f}'.format(s) for s in scores])

with open(output_path, 'a') as fo:
    fo.write("{0}\n".format(output))

for test_dataset in DATASETS:
    test_path = os.path.join(data_path, '{0}/test.csv'.format(test_dataset))
    test_x, test_y, test_ids, test_sentences = load_data(test_path)

    predicts = model.predict(test_x, verbose=False)
    pred = predicts
    pred[pred>=.5]=1
    pred[pred<.5]=0
    scores = evaluate(pred, test_y)
    output = ",".join(['baseline', LANGUAGE, TRAIN_SET, test_dataset, 'test', str(LR), "-", "-", str(SEED)]+['{0:.4f}'.format(s) for s in scores])
    with open(output_path, 'a') as fo:
        fo.write("{0}\n".format(output))
