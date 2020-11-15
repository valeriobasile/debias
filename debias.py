#!/usr/bin/env python

import os
import numpy as np
import codecs
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects
from tqdm import tqdm
import pickle
import csv
import keras
from keras.optimizers import Adam
import pandas as pd
from tensorflow.python.keras import backend as K
import tensorflow as tf
from gensim.corpora.dictionary import Dictionary
from numpy.random import seed
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import AUC
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import class_weight
from glob import glob
from shutil import copyfile

SEQ_LEN = 50
language = 'en'
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-5
N_TOPICS = 10

TRAIN = sys.argv[1]
ALPHA = eval(sys.argv[2])
SEED = eval(sys.argv[3])
seed(SEED)

bert_model = {
        'model': 'uncased_L-12_H-768_A-12',
        'config': 'bert_config.json',
        'checkpoint': 'bert_model.ckpt',
        'vocab': 'vocab.txt'
}

base_path = '.'
data_path = os.path.join(base_path, 'data/multilingual_binary')
model_path = os.path.join(base_path, 'bert-models')
pretrained_path = os.path.join(model_path, 'uncased_L-12_H-768_A-12')
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

train_path = os.path.join(data_path, TRAIN)
topics_train_path = os.path.join(train_path, 'train_topics.csv')
train_path = os.path.join(train_path, 'train.csv')

dev_path = os.path.join(data_path, TRAIN)
topics_dev_path = os.path.join(dev_path, 'dev_topics.csv')
dev_path = os.path.join(dev_path, 'dev.csv')

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

            text_preprocessed = tp.clean(row['text'])
            doc = nlp(text_preprocessed)
            tokens = [t.lemma_.lower() for t in doc if not t.is_stop]
            sentences.append(tokens)

            instance_ids.append(row['id'])
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(sentiments), instance_ids, sentences

train_x, train_y, train_ids, train_sentences = load_data(train_path)
dev_x, dev_y, dev_ids, dev_sentences = load_data(dev_path)

def read_topics(path):
    t = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(np.array([eval(row['topic_{0}'.format(i)]) for i in range(N_TOPICS)]))
    return np.array(t)

train_t = read_topics(topics_train_path)
dev_t = read_topics(topics_dev_path)

def cos_distance(y_true, y_pred):
    return 1-cos_similarity(y_true, y_pred)

def cos_similarity(y_true, y_pred):
    #return -(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=1)*.5-1.0)
    return 1-(tf.keras.losses.mse(y_true, y_pred))



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
outputs_t = keras.layers.Dense(N_TOPICS, name="topic", activation="sigmoid")(dropout)
outputs = keras.layers.Dense(1, name="label", activation="sigmoid")(dropout)

model = keras.models.Model(inputs, [outputs, outputs_t])
#model.summary()
losses = {
	"label": "binary_crossentropy",
	"topic": cos_similarity
    }
metrics = {
	"label": AUC(name="auc"),
	"topic": cos_distance
    }
lossWeights = {"label": ALPHA, "topic": 1-ALPHA}

model.compile(
    Adam(lr=LR),
    loss=losses,
    loss_weights=lossWeights,
    metrics=metrics,
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


obj = get_custom_objects()
obj['cos_distance']=cos_distance
obj['cos_similarity']=cos_similarity
model = load_model('best_model.h5', custom_objects=obj)

for test_dataset in ['hateval_mig', 'hateval_mis', 'hateval', 'abuseval', 'offenseval', 'ami_ibereval', 'ami_evalita', 'founta', 'waseem']:
    test_data_path = os.path.join(base_path, 'data/multilingual_binary/{0}/{1}/'.format(LANGUAGE, test_dataset))
    test_path = os.path.join(test_data_path, 'test.csv')
    test_x, test_y, test_ids, test_sentences = load_data(test_path)

    predicts = model.predict(test_x, verbose=False)
    pred = predicts[0]
    pred[pred>=.5]=1
    pred[pred<.5]=0
    report = classification_report(test_y, pred, output_dict=True)
    df = pd.DataFrame(report)

    with open("results.csv, "a") as fo:
        writer = csv.writer(fo)
        writer.writerow((LANGUAGE, TRAIN, test_dataset, SEED,
                        df['0']['precision'],
                        df['0']['recall'],
                        df['0']['f1-score'],
                        df['1']['precision'],
                        df['1']['recall'],
                        df['1']['f1-score'],
                        df['macro avg']['precision'],
                        df['macro avg']['recall'],
                        df['macro avg']['f1-score']))
