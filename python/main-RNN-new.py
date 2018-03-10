# -*- coding: utf-8 -*-
"""
The main RNN algorithm.

Updates:
1. RNN _input layer now changed to word-vector encoding
2. can answer single queries
3. use YELLOW highlight sections as training data

Modeled after the code found in Ch.6 of "Learning Tensorflow" by Tom Hope et al.

@author: YKY with advice from Jesmer Wong
"""
import numpy as np
import os						# for os.listdir and os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	# suppress Tensorflow warnings
import tensorflow as tf
from nltk.corpus import stopwords
import re						# for removing punctuations
import sys						# for sys.stdin.readline()
from collections import defaultdict	# for default value of word-vector dictionary
import pickle

path_to_glove = "/data/wiki-news-300d-1M.vec"	# change to your path and filename
PRE_TRAINED = True
GLOVE_SIZE = 300				# dimension of word vectors in GloVe file
batch_size = 100
num_classes = 3
num_layers = 3
hidden_layer_size = 32
times_steps = 32				# this number should be same as fixed_seq_len below

""" These are the 3 classes of laws:
* nuisance
* dangerous driving
* OTHERWISE, or work injuries
"""
categories = ["nuisance", "dangerous-driving", "injuries"]

# =================== Read prepared data from file ======================

pickle_off = open("training-data.pickle", "rb")
data = pickle.load(pickle_off)

pickle_off = open("training-labels.pickle", "rb")
labels = pickle.load(pickle_off)

pickle_off = open("training-word-list.pickle", "rb")
word_list = pickle.load(pickle_off)

# set default value = zero vector, if word not found in dictionary
zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

num_examples = len(data)
print("Data size = ", num_examples, " examples")
print("# unique words = ", len(word_list))
print("# vectorized words = ", len(word2vec_map))

fixed_seq_len = times_steps

# ============ Split data into Training and Testing sets, 50%:50% ============

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]
midpoint = num_examples // 2
train_x = data[:midpoint]
train_y = labels[:midpoint]
train_seqlens = seqlens[:midpoint]

test_x = data[midpoint:]
test_y = labels[midpoint:]
test_seqlens = seqlens[midpoint:]

# =================== Prepare batch data ============================

def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
	instance_indices = list(range(len(data_x)))
	np.random.shuffle(instance_indices)
	batch = instance_indices[:batch_size]
	x = [[word2vec_map[word]
			for word in data_x[i].split()]
			for i in batch]
	y = [data_y[i] for i in batch]
	seqlens = [data_seqlens[i] for i in batch]
	return x, y, seqlens

# ========= define input, output, and RNN structure =========

_inputs = tf.placeholder(tf.float32, shape=[None, times_steps, GLOVE_SIZE])
_labels = tf.placeholder(tf.float32, shape=[None, num_classes])
_seqlens = tf.placeholder(tf.int32, shape=[None])

with tf.name_scope("GRU"):
	_GRU_layer = tf.contrib.rnn.GRUCell(hidden_layer_size)
	_cells = tf.contrib.rnn.MultiRNNCell([_GRU_layer] * num_layers)

	outputs, states = tf.nn.dynamic_rnn(cell=_cells,
								inputs=_inputs,
								# sequence_length=_seqlens,
								dtype=tf.float32,
								scope="GRU")
	# 'outputs' shape = [batch_size, max_time, cell_state_size]
	# outputs = tf.stack(outputs)
	outputs = tf.transpose(outputs, [1, 0, 2])

weights = {
	'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size,
													num_classes],
													mean=0, stddev=.01))
}
biases = {
	'linear_layer': tf.Variable(tf.truncated_normal([num_classes],
													mean=0, stddev=.01))
}

# ========== Define final state and objective function ================

final_output = tf.matmul(outputs, weights["linear_layer"]) + biases["linear_layer"]

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,
												  labels=_labels)
cross_entropy = tf.reduce_mean(softmax)

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels, 1),
							  tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction,
								   tf.float32)))*100

# ================== Run the session =====================

print("\n**** Training RNN....")
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(600 + 1):
		x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
															train_x, train_y,
															train_seqlens)
		sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch, _seqlens: seqlen_batch})

		if step % 25 == 0:
			acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
												_labels: y_batch,
												_seqlens: seqlen_batch})
			print("Accuracy at %d: %.5f" % (step, acc))

	for test_batch in range(5):
		x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
														 test_x, test_y,
														 test_seqlens)
		batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
										 feed_dict={_inputs: x_test,
													_labels: y_test,
													_seqlens: seqlen_test})
		print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))

	# =================== Process a single query ===================

	while True:
		print("----------------------------\n? ", end = '')
		sys.stdout.flush()
		query = sys.stdin.readline()
		query = re.sub(r'[^\w\s-]',' ', query)	# remove punctuations except hyphen
		query_words = []
		for word in query.lower().split():		# convert to lowercase
			if word not in stopwords.words('english'):	# remove stop words
				query_words.append(word)
		
		query_vectors = []
		glove_file = open(path_to_glove, "r")
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
			if entry_number > 20000:
				# took too long to find the words
				break

		long_enough = False
		while not long_enough:					# make up to 128 = times_steps size
			for word in query_words:
				query_vectors.append(word2vec_map[word])
				if len(query_vectors) == times_steps:
					long_enough = True
					break
		result = sess.run(tf.argmax(final_output, 1), feed_dict={_inputs: [query_vectors],
														 _seqlens: [times_steps]})
		print(" ⟹ ", categories[result[0]])
