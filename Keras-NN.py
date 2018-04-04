import os						# for os.listdir and os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	# suppress Tensorflow warnings
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
#hidden_layer_size = 64
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


#define objective function here
opt = Adam(lr=0.0067, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#how many epoch to train
nb_epochs = 20

print('Build LSTM RNN model ...')
model = Sequential() # this is the first layer of keras
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=[times_steps,GLOVE_SIZE]))
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=num_classes, activation='softmax'))

# compile the model
print("Compiling ...")
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) # loss type of cateforical crossentropy is good to classification model
model.summary()

# training
print("Training ...")
x_train,y_train = get_sentence_batch(batch_size,train_x,train_y)        #list
model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=nb_epochs)
          
# testing
print("\nTesting ...")
x_test,y_test = get_sentence_batch(batch_size,test_x,test_y)
score, accuracy = model.evaluate(np.array(x_test),np.array(y_test), batch_size=batch_size, verbose=1)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

# save the model
#model.save('RNN-Keras.h5')



# =================== Process a single query =================== #

while True:
        print("----------------------------\n? ", end = '')
        sys.stdout.flush()
        query = sys.stdin.readline()
        query = re.sub(r'[^\w\s-]',' ', query)	# remove punctuations except hyphen
        query_words = []
        for word in query.lower().split():		# convert to lowercase
                if word not in stopwords.words('english'):	# remove stop words
                        query_words.append(word)

        # ===== convert query to word-vectors
        query_vectors = []
        glove_file = open(path_to_glove, "r",encoding = "utf-8")
        count_all_words = 0
        entry_number = 0
        for word_entry in glove_file:
                vals = word_entry.split()
                word = str(vals[0])
                entry_number += 1
                if word in query_words:
                        count_all_words += 1
                        print(count_all_words, word, end = '\r')
                        coefs = np.asarray(vals[1:], dtype='float32')
                        coefs /= np.linalg.norm(coefs)
                        word2vec_map[word] = coefs
                if count_all_words == len(word_list) - 1:
                        break
                if entry_number > 50000:
                        # took too long to find the words
                        break

        # ===== make the query length to be (32) = times_steps size
        long_enough = False
        while not long_enough:
                for word in query_words:
                        query_vectors.append(word2vec_map[word])
                        if len(query_vectors) == times_steps:
                                long_enough = True
                                break

#=================== Not yet finish - Doing prediction ==============================#          
        print(type(query_vectors))
        #result = sess.run(tf.argmax(final_output, 1), feed_dict={_inputs: [query_vectors]})
        indices = list(range(len(query_vectors)))
        np.random.shuffle(indices)
        data = np.array(data)[indices]
        instance_indices = list(range(len(data)))
        np.random.shuffle(instance_indices)
        batch = instance_indices[:batch_size]
        x = [[word2vec_map[word]
			for word in data[i].split()]
			for i in batch]
        result = model.predict(x)
        #print(query_vectors)
        print(type(x))#list
        print(type(query_vectors)) #list
        print(type(result)) #np.array
        #print(result)

        print(" ⟹  category: ", categories[result[0]])


