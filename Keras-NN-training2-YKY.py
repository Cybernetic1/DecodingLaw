import os                       # for os.listdir and os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppress Tensorflow warnings
import numpy as np
from nltk.corpus import stopwords
import re                       # for removing punctuations
import sys                      # for sys.stdin.readline()
from collections import defaultdict # for default value of word-vector dictionary
import pickle
import h5py

import tensorflow as tf

# Keras deep learning library
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback
from keras import backend as K

# Load the trained model
#model = keras.models.load_weights('Keras-NN_weigts.h5')
path_to_glove = "wiki-news-300d-1M.vec"
GLOVE_SIZE = 300
batch_size = 512
num_classes = 10
times_steps = 32                        # this number should be same as fixed_seq_len below

# 10 categories:
categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
    "property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

# =================== Read prepared training data from file ======================

pickle_off = open("prepared-data/training-data-3.pickle", "rb")
data = pickle.load(pickle_off)                
pickle_off = open("prepared-data/training-labels-3.pickle", "rb")
labels = pickle.load(pickle_off)

pickle_off = open("prepared-data/training-word-list-3.pickle", "rb")
word_list = pickle.load(pickle_off)

pickle_off = open("prepared-data/training-word2vec-map-3.pickle", "rb")
word2vec_map = pickle.load(pickle_off)

zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

# set default value = zero vector, if word not found in dictionary
zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

num_examples = len(data)

fixed_seq_len = times_steps

# ============ Split data into Training and Testing sets, 50%:50% ============

data_indices = list(range(len(data)))                  
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]             
labels = np.array(labels)[data_indices]
#seqlens = np.array(seqlens)[data_indices]
midpoint = num_examples // 2
train_x = data[:midpoint]              
train_y = labels[:midpoint]             
#train_seqlens = seqlens[:midpoint]

test_x = data[midpoint:]
test_y = labels[midpoint:]
#test_seqlens = seqlens[midpoint:]

# =================== Prepare batch data ============================

def get_sentence_batch(batch_size, data_x, data_y):  # omit: data_seqlens
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    
    x = [[word2vec_map[word]for word in data_x[i].split()]for i in batch]
    y = [data_y[i] for i in batch]
    #seqlens = [data_seqlens[i] for i in batch]
    return x, y     # seqlens

# =========== define objective function for unsupervised competitive learning ==========

    # y_true can be ignored
    """
    threshold = np.partition(y_pred, -3)[-3]           # last 3 elements would be biggest
    loss = np.zeros(10)
    for i in range(0,10):
        y = y_pred[i]
        if y > threshold:                              
            loss[i] = 1.0 - y                          # if it is the winner, ideal value = 1.0
        else:
            loss[i] = y                                # if it is loser, ideal value = 0.0
    """
    #loss = tf.gather(y2, indices)

def bingo_loss(y_true, y_pred):
    one = tf.ones([10])
    y2 = tf.subtract(one, y_pred)
    _, indices = tf.nn.top_k(y_pred, k = 3)
    loss = tf.scatter_update(y2, indices, y_pred)
    return loss

# ========= define input, output, and NN structure - need to modify =========
#define the activation function here
opt = Adam(lr=0.0067, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#how many epoch to train
nb_epochs = 40
x_train,y_train = get_sentence_batch(batch_size,train_x,train_y)        #list
x_test,y_test = get_sentence_batch(batch_size,test_x,test_y)
#print (y_test)


print('Build NN model ...')
model = Sequential() # this is the first layer of keras
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=[times_steps,GLOVE_SIZE]))
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=num_classes, activation='softmax'))

# compile the model -- define loss function and optimizer
print("Compiling ...")
# loss type of cateforical crossentropy is good to classification model
model.compile(loss=bingo_loss, optimizer=opt, metrics=['accuracy'])
model.summary()
model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=nb_epochs)

# testing
print("\nTesting ...")

score, accuracy = model.evaluate(np.array(x_test),np.array(y_test), batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# Save the model #
model.save('Keras-NN-trainging2.h5')
