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
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import AUC
from keras.models import load_model
import keras
import tensorflow as tf

N_TOPICS = 10
ALPHA = 0.5
dataset = "hateval"
language = 'en'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-6
SEED = 1

data_path = 'data/{0}/{1}'.format(language, dataset)
train_path = os.path.join(data_path, 'train.csv')
train_topics_path = os.path.join(data_path, 'train_topics.csv')
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
            t.append(np.array([row['topic_{0}'.format(i)] for i in range(10)]))
    return np.array(t)


# train

# predict
test_path = os.path.join(data_path, 'test.csv')
test_topics_path = os.path.join(data_path, 'test_topics.csv')
test_sentenc_path = os.path.join(data_path, 'test_sentenc.csv')
test_x, test_y, test_ids, test_sentences = load_data(test_path, test_sentenc_path)
train_t = read_topics(train_topics_path)
print(train_x.shape, train_y.shape, train_t.shape)
print(train_x.shape, train_y.shape, train_t.shape)

def cos_distance(y_true, y_pred):
    return 1-cos_similarity(y_true, y_pred)

def cos_similarity(y_true, y_pred):
    return -(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=1)*.5-1.0)

model = keras

input = Input(shape=(512,))
dropout = keras.layers.Dropout(0.5)(input)
outputs_t = keras.layers.Dense(N_TOPICS, name="topic", activation="sigmoid")(dropout)
outputs = keras.layers.Dense(1, name="label", activation="sigmoid")(dropout)

model = keras.models.Model(input, [outputs, outputs_t])

losses = {
	"label": "binary_crossentropy",
	"topic": cos_similarity
    }
metrics = {
	"label": "label": AUC(name="auc"),
	"topic": cos_distance
    }
lossWeights = {"label": ALPHA, "topic": 1-ALPHA}

model.compile(
    Adam(lr=LR),
    loss=losses,
    metrics=metrics,
    loss_weights=lossWeights
)

try:
    os.remove("best_model.h5")
except:
    print ("file not found")

es = EarlyStopping(monitor='val_label_loss', mode='min', verbose=1, patience=3)
#mc = ModelCheckpoint('best_model.h5', monitor='val_label_auc', mode='max', verbose=1, save_best_only=True)
mc = ModelCheckpoint("best_model.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(
    train_x,
    {"label": train_y, "topic": train_t},
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=True,
    shuffle=True,
    callbacks=[es, mc],
    validation_data=(dev_x, {"label": dev_y, "topic": dev_t})
    )

# pred = model.predict(dev_x)
# print (classification_report(dev_y, pred))
# print (confusion_matrix(dev_y, pred))
# pred = model.predict(test_x)
# print (classification_report(test_y, pred))
# print (confusion_matrix(test_y, pred))
