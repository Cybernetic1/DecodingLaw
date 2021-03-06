# -*- coding: utf-8 -*-
"""
The main RNN algorithm.

Updates:
1. RNN _input layer now changed to word-vector encoding
2. can answer single queries
3. modified for A2J Hackathon, using scraped data for training

Modeled after the code found in Ch.6 of "Learning Tensorflow" by Tom Hope et al.

@author: YKY
"""
import os						# for os.listdir and os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	# suppress Tensorflow warnings
import numpy as np
from nltk.corpus import stopwords
import re						# for removing punctuations
import sys						# for sys.stdin.readline()
from collections import defaultdict	# for default value of word-vector dictionary
import pickle
import tensorflow as tf
# print("\n" * 100)				# get the screen clear of warning messages

path_to_glove = "/data/wiki-news-300d-1M.vec"	# change to your path and filename
GLOVE_SIZE = 300				# dimension of word vectors in GloVe file
batch_size = 512
num_classes = 10
hidden_layer_size = 64
times_steps = 32				# this number should be same as fixed_seq_len below

# 10 categories:
categories = ["matrimonial-rights", "separation", "divorce", "after-divorce", "divorce-maintenance",
	"property-on-divorce", "types-of-marriages", "battered-wife-and-children", "Harmony-House", "divorce-mediation"]

# =================== Read prepared training data from file ======================

suffix = "-3"

pickle_off = open("prepared-data/training-data" + suffix + ".pickle", "rb")
data = pickle.load(pickle_off)

pickle_off = open("prepared-data/training-labels" + suffix + ".pickle", "rb")
labels = pickle.load(pickle_off)

pickle_off = open("prepared-data/training-word-list" + suffix + ".pickle", "rb")
word_list = pickle.load(pickle_off)

pickle_off = open("prepared-data/training-word2vec-map" + suffix + ".pickle", "rb")
word2vec_map = pickle.load(pickle_off)

# set default value = zero vector, if word not found in dictionary
zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

num_examples = len(data)
print("# training examples = ", num_examples)
print("# unique words = ", len(word_list))
print("# vectorized words = ", len(word2vec_map))

fixed_seq_len = times_steps
#seqlens = [times_steps] * num_examples

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
	x = [[word2vec_map[word]
			for word in data_x[i].split()]
			for i in batch]
	y = [data_y[i] for i in batch]
	#seqlens = [data_seqlens[i] for i in batch]
	return x, y		# seqlens

# ========= define input, output, and RNN structure =========

_inputs = tf.placeholder(tf.float32, shape=[None, times_steps, GLOVE_SIZE])
_labels = tf.placeholder(tf.float32, shape=[None, num_classes])
# _seqlens = tf.placeholder(tf.int32, shape=[None])

with tf.name_scope("biGRU"):
	with tf.variable_scope('forward'):
		gru_fw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
		gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell)

	with tf.variable_scope('backward'):
		gru_bw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
		gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell)

		outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell,
														  cell_bw=gru_bw_cell,
														  inputs=_inputs,
														  # sequence_length=_seqlens,
														  dtype=tf.float32,
														  scope="biGRU")
states = tf.concat(values=states, axis=1)
weights = {
	'linear_layer': tf.Variable(tf.truncated_normal([2*hidden_layer_size,
													num_classes],
													mean=0, stddev=.01))
}
biases = {
	'linear_layer': tf.Variable(tf.truncated_normal([num_classes],
													mean=0, stddev=.01))
}

# ========== Define final state and objective function ================

final_output = tf.matmul(states,
						 weights["linear_layer"]) + biases["linear_layer"]

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
	for step in range(500 + 1):
		x_batch, y_batch = get_sentence_batch(batch_size,
											train_x, train_y)
		sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch})

		if step % 25 == 0:
			acc = sess.run(accuracy, feed_dict={_inputs: x_batch, _labels: y_batch})
			print("Accuracy at %d: %.5f" % (step, acc))

	for test_batch in range(5):
		x_test, y_test = get_sentence_batch(batch_size, test_x, test_y)
		batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
								feed_dict={_inputs: x_test, _labels: y_test})
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

		# ===== convert query to word-vectors
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

		result = sess.run(tf.argmax(final_output, 1), feed_dict={_inputs: [query_vectors]})
		print(" ⟹  category: ", categories[result[0]])
