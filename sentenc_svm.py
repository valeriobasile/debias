#!/usr/bin/env python

import os
import numpy as np
import codecs
from tqdm import tqdm
import pickle
import csv
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
from pprint import pprint
from numpy.random import seed
import sys
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_TOPICS = 100
#ALPHA = eval(sys.argv[5])
#SEED = eval(sys.argv[6])
dataset = "hateval"
language = 'en'

data_path = 'data/multilingual_binary/{0}/{1}'.format(language, dataset)
train_path = os.path.join(data_path, 'train.csv')
train_topics_path = os.path.join(data_path, 'train_topics_{0}.csv'.format(N_TOPICS))
train_sentenc_path = os.path.join(data_path, 'train_sentenc.csv')
dev_path = os.path.join(data_path, 'dev.csv')
dev_sentenc_path = os.path.join(data_path, 'dev_sentenc.csv')

def load_data(path, sentenc_path):
    sentences = []
    sentiments = []
    sentenc = []
    instance_ids = []
    with open(path, 'r') as f, open(sentenc_path, 'r') as fs:
        reader = csv.DictReader(f)
        for row in reader:
            sentiments.append(eval(row['label']))
            sentences.append(row['text'])
            instance_ids.append(row['id'])
        for row in tqdm(fs):
            sentenc.append(np.array([eval(x) for x in row.split(' ')]))

    items = list(zip(sentenc, sentiments, sentences))
    np.random.shuffle(items)
    sentenc, sentiments, sentences = zip(*items)
    return np.array(sentenc), np.array(sentiments), instance_ids, sentences

train_x, train_y, train_ids, train_sentences = load_data(train_path, train_sentenc_path)
dev_x, dev_y, dev_ids, dev_sentences = load_data(dev_path, dev_sentenc_path)

def read_topics(path):
    t = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(np.array([row['topic_{0}'.format(i)] for i in range(N_TOPICS)]))
    return np.array(t)


#dev_t = read_topics(topics_dev_path)

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

# train
model = make_pipeline(StandardScaler(), SVC(kernel='linear', verbose=True, class_weight='balanced'))
model.fit(train_x, train_y)

# predict
test_path = os.path.join(data_path, 'test.csv')
test_topics_path = os.path.join(data_path, 'test_topics_{0}.csv'.format(N_TOPICS))
test_sentenc_path = os.path.join(data_path, 'test_sentenc.csv')
test_x, test_y, test_ids, test_sentences = load_data(test_path, test_sentenc_path)
train_t = read_topics(train_topics_path)

pred = model.predict(dev_x)
print (classification_report(dev_y, pred))
print (confusion_matrix(dev_y, pred))
pred = model.predict(test_x)
print (classification_report(test_y, pred))
print (confusion_matrix(test_y, pred))
