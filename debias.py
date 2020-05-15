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
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
from tensorflow.python.keras import backend as K
import tensorflow as tf
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.matutils import sparse2full
from pprint import pprint
from numpy.random import seed
import sys

SEQ_LEN = 100
BATCH_SIZE = eval(sys.argv[1])
EPOCHS = eval(sys.argv[2])
LR = eval(sys.argv[3])
N_TOPICS = eval(sys.argv[4])
ALPHA = eval(sys.argv[5])
SEED = eval(sys.argv[6])
TRAIN_SET = sys.argv[7].split("-")
TEST_SET = sys.argv[8].split("-")

bert_model = {
        'model': 'uncased_L-12_H-768_A-12',
        'config': 'bert_config.json',
        'checkpoint': 'bert_model.ckpt',
        'vocab': 'vocab.txt'
}

base_path = '.'
data_path = os.path.join(base_path, 'data')
model_path = os.path.join(base_path, 'bert-models')
pretrained_path = os.path.join(model_path, 'uncased_L-12_H-768_A-12')
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

fulldata_path = os.path.join(data_path, 'hateval2019_target.csv')
topics_train_path = os.path.join(data_path, 'topics_train_{0}.csv'.format(N_TOPICS))
topics_dev_path = os.path.join(data_path, 'topics_dev_{0}.csv'.format(N_TOPICS))
topics_test_path = os.path.join(data_path, 'topics_test_{0}.csv'.format(N_TOPICS))

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

with open(fulldata_path, 'r') as f:
    reader = csv.DictReader(f)

tokenizer = Tokenizer(token_dict)

def load_data(path, target, language, split):
    global tokenizer
    indices, sentiments = [], []
    sentences = []
    instance_ids = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['set'] == split and row['language'] == language and row['target'] in target:
                ids, segments = tokenizer.encode(row['text'], max_len=SEQ_LEN)
                indices.append(ids)
                sentiments.append(eval(row['HS']))
                tokens = tokenizer.tokenize(row['text'])
                sentences.append(tokens[1:-1])
                instance_ids.append(row['id'])

    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    mod = indices.shape[0] % BATCH_SIZE
    if mod > 0:
        indices, sentiments = indices[:-mod], sentiments[:-mod]
    return [indices, np.zeros_like(indices)], np.array(sentiments), instance_ids, sentences

train_x, train_y, train_ids, train_sentences = load_data(fulldata_path, TRAIN_SET, 'en', 'train')
dev_x, dev_y, dev_ids, dev_sentences = load_data(fulldata_path, TRAIN_SET, 'en', 'dev')
test_x, test_y, test_ids, test_sentences = load_data(fulldata_path, TEST_SET, 'en', 'test')

if not os.path.isfile(topics_train_path):
    id2word = Dictionary(train_sentences)
    corpus_train = [id2word.doc2bow(text) for text in train_sentences]
    lda_model = LdaModel(corpus=corpus_train,
                                        id2word=id2word,
                                        num_topics=N_TOPICS,
                                        random_state=100,
                                        passes=50,
                                        alpha='auto')

    topics_train = lda_model.get_document_topics(corpus_train)
    corpus_dev = [id2word.doc2bow(text) for text in dev_sentences]
    topics_dev = lda_model.get_document_topics(corpus_dev)
    corpus_test = [id2word.doc2bow(text) for text in test_sentences]
    topics_test = lda_model.get_document_topics(corpus_test)

    with open(topics_train_path, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(['id']+['topic_{0}'.format(i) for i in range(N_TOPICS)])
        for i, vector in enumerate(topics_train):
            row = [train_ids[i]] + [str(x) for x in sparse2full(vector, N_TOPICS)]
            writer.writerow(row)
    with open(topics_dev_path, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(['id']+['topic_{0}'.format(i) for i in range(N_TOPICS)])
        for i, vector in enumerate(topics_dev):
            row = [dev_ids[i]] + [str(x) for x in sparse2full(vector, N_TOPICS)]
            writer.writerow(row)
    with open(topics_test_path, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(['id']+['topic_{0}'.format(i) for i in range(N_TOPICS)])
        for i, vector in enumerate(topics_test):
            row = [test_ids[i]] + [str(x) for x in sparse2full(vector, N_TOPICS)]
            writer.writerow(row)

def read_topics(path):
    t = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(np.array([row['topic_{0}'.format(i)] for i in range(N_TOPICS)]))
    return np.array(t)

train_t = read_topics(topics_train_path)
dev_t = read_topics(topics_dev_path)
test_t = read_topics(topics_test_path)

def evaluate(predicts, gold):
        accuracy = (np.sum(gold == predicts) / gold.shape[0])
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for c in range(len(np.unique(gold))):
            for g, p in zip(gold, predicts):
                if g == p:
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

def cos_distance(y_true, y_pred):
    return 1-cos_similarity(y_true, y_pred)

def cos_similarity(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return (K.mean(y_true * y_pred, axis=-1)+1.0)*0.5

seed(SEED)

model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)

inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
dropout = keras.layers.Dropout(0.5)(dense)
outputs_t = keras.layers.Dense(N_TOPICS, name="topic")(dropout)
outputs = keras.layers.Dense(2, name="label")(dropout)

model = keras.models.Model(inputs, [outputs, outputs_t])
#model.summary()
losses = {
	"label": "sparse_categorical_crossentropy",
	"topic": cos_similarity
    }
metrics = {
	"label": "accuracy",
	"topic": cos_similarity
    }
lossWeights = {"label": ALPHA, "topic": 1-ALPHA}

model.compile(
    #RAdam(lr=LR),
    Adam(lr=LR),
    loss=losses,
    metrics=metrics,
)



train_yt = np.hstack((np.asmatrix(train_y).transpose(), train_t))
dev_yt = np.hstack((np.asmatrix(dev_y).transpose(), dev_t))

#callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

history = model.fit(
    train_x,
    {"label": train_y, "topic": train_t},
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=True,
    #callbacks=callbacks,
    validation_data=(dev_x, {"label": dev_y, "topic": dev_t}),
    )

experiment_key = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(
    "-".join(TRAIN_SET),
    "-".join(TEST_SET),
    BATCH_SIZE,
    EPOCHS,
    LR,
    N_TOPICS,
    ALPHA,
    SEED
)

BATCH_SIZE = eval(sys.argv[1])
EPOCHS = eval(sys.argv[2])
LR = eval(sys.argv[3])
N_TOPICS = eval(sys.argv[4])
ALPHA = eval(sys.argv[5])
SEED = eval(sys.argv[6])

predicts = model.predict(dev_x, verbose=False)
pred = predicts[0].argmax(axis=1)
scores = evaluate(pred, dev_y)

with open("results.tsv", "a") as fa:
    fa.write("dev\t{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t".format(
        "-".join(TRAIN_SET),
        "-".join(TEST_SET),
        BATCH_SIZE,
        EPOCHS,
        LR,
        N_TOPICS,
        ALPHA,
        SEED
    ))
    fa.write("\t".join(['{0:.3f}'.format(s) for s in scores]))
    fa.write("\n")

predictions_path = os.path.join(data_path, 'predictions/dev_{0}.txt'.format(experiment_key))
with open(predictions_path, "w") as fo:
    for p in pred:
        fo.write("{0}\n".format(p))

predicts = model.predict(test_x, verbose=False)
pred = predicts[0].argmax(axis=1)
scores = evaluate(pred, test_y)
with open("results.tsv", "a") as fa:
    fa.write("test\t{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t".format(
        "-".join(TRAIN_SET),
        "-".join(TEST_SET),
        BATCH_SIZE,
        EPOCHS,
        LR,
        N_TOPICS,
        ALPHA,
        SEED
    ))
    fa.write("\t".join(['{0:.3f}'.format(s) for s in scores]))
    fa.write("\n")

predictions_path = os.path.join(data_path, 'predictions/test_{0}.txt'.format(experiment_key))
with open(predictions_path, "w") as fo:
    for p in pred:
        fo.write("{0}\n".format(p))
