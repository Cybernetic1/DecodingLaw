import os						# for os.listdir and os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	        # suppress Tensorflow warnings
import numpy as np
from nltk.corpus import stopwords
import re						# for removing punctuations
import sys						# for sys.stdin.readline()
from collections import defaultdict	                # for default value of word-vector dictionary
import pickle
import h5py                                             # for save/load the model


# Keras deep learning library
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback

path_to_glove = "wiki-news-300d-1M.vec"	        # change to your path and filename
GLOVE_SIZE = 300				# dimension of word vectors in GloVe file
batch_size = 512
num_classes = 10
times_steps = 32		                # this number should be same as fixed_seq_len below

# 10 categories:
categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
	"property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

# =================== Read prepared training data from file ======================

pickle_off = open("prepared-data/training-data-3.pickle", "rb")
data = pickle.load(pickle_off)                  #list
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
	return x, y		# seqlens

# ========= define input, output, and NN structure - need to modify =========


#define optimizer here
opt = Adam(lr=0.0067, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#how many epoch to train
nb_epochs = 40
x_train,y_train = get_sentence_batch(batch_size,train_x,train_y)
x_test,y_test = get_sentence_batch(batch_size,test_x,test_y)


print('Build LSTM model ...')
model = Sequential() # this is the first layer of keras
model.add(LSTM(units=256, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=[times_steps,GLOVE_SIZE]))
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=num_classes, activation='softmax'))

# compile the model
print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) # loss type of cateforical crossentropy is good to classification model
model.summary()

# training
print("Training ...")

model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=nb_epochs)
          
# testing
print("\nTesting ...")

score, accuracy = model.evaluate(np.array(x_test),np.array(y_test), batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# Save the model #
model.save('Keras-NN.h5')
