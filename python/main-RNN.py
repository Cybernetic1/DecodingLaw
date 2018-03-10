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

path_to_glove = "/data/wiki-news-300d-1M.vec"	# change to your path and filename
PRE_TRAINED = True
GLOVE_SIZE = 300				# dimension of word vectors in GloVe file
batch_size = 64
embedding_dimension = 64		# this is used only if PRE_TRAINED = False
num_classes = 2
hidden_layer_size = 32
times_steps = 128				# this number should be same as fixed_seq_len below

""" These are the 3 classes of laws:
* nuisance
* dangerous driving
* OTHERWISE, or work injuries
"""

# =================== Read case examples from file ======================
"""
0. for each category:
1. read all cases in YELLOW folder
2.	  for each case generate N examples (consecutive word sequences of fixed length)
"""
seqlens = []
labels = []
data = []
num_examples = 0 				# change to larger later
fixed_seq_len = times_steps		# For each case law, take N consecutive words from text

for i, category in enumerate(["nuisance-YELLOW", "dangerous-driving-YELLOW"]):
	for filename in os.listdir("laws-TXT/" + category):
		yellow_stuff = []
		with open("laws-TXT/" + category + "/" + filename) as f:
			for line in f:
				line = re.sub(r'[^\w\s-]',' ',line)	# remove punctuations except hyphen
				for word in line.lower().split():	# convert to lowercase
					if word not in stopwords.words('english'):	# remove stop words
						yellow_stuff.append(word)
		print("Case-law length = ", len(yellow_stuff))

		for j in range(0, 10):
			# Randomly select a sequence of words (of fixed length) in yellow text
			seqlens.append(fixed_seq_len)		# this is variable in the original code (with zero-padding), but now it's fixed because we don't use zero-padding
			rand_start = np.random.choice(range(0, len(yellow_stuff) - fixed_seq_len))
			data.append(" ".join(yellow_stuff[rand_start: rand_start + fixed_seq_len]))

		# set labels
		labels += [i]

# ================ Set up data (for training & testing) ================

for i in range(len(labels)):
	label = labels[i]
	one_hot_encoding = [0]*2
	one_hot_encoding[label] = 1
	labels[i] = one_hot_encoding

word_list = []				# to store the list of words appearing in case-text
index = 0
for sent in data:
	for word in sent.split():
		if word not in word_list:
			word_list.append(word)
			index += 1

# ============== Create word-to-vector dictionary ===========

word2vec_map = {}

count_all_words = 0
f = open(path_to_glove, "r")
f2 = open("found-words.txt", "w")
for line in f:
	vals = line.split()
	word = str(vals[0])
	if word in word_list:
		print(count_all_words, word, file=f2)
		print(word, "             ", end='\r')
		count_all_words += 1
		coefs = np.asarray(vals[1:], dtype='float32')
		coefs /= np.linalg.norm(coefs)
		word2vec_map[word] = coefs
	if count_all_words == len(word_list) - 1:
		print("*** found all words ***")
		break
	if count_all_words >= 500:			# it takes too long to look up the entire dictionary, so I cut it short
		break
f2.close()
# set default value = zero vector, if word not found in dictionary
zero_vector = np.asarray([0.0]*300, dtype='float32')	# this is for when the word-vector is not found in the file
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

print("Vocabulary size = ", len(word2vec_map))

# ============ Split data into Training and Testing sets, 50%:50% ============

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]
train_x = data[:num_examples]
train_y = labels[:num_examples]
train_seqlens = seqlens[:num_examples]

test_x = data[num_examples:]
test_y = labels[num_examples:]
test_seqlens = seqlens[num_examples:]

# =================== Prepare batch data ============================

def get_sentence_batch(batch_size, data_x,
					   data_y, data_seqlens):
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
														  sequence_length=_seqlens,
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

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(151):
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

	print("Please enter your query: ")
	query = sys.stdin.readline()
	query = re.sub(r'[^\w\s-]',' ', query)	# remove punctuations except hyphen
	query_words = []
	for word in query.lower().split():		# convert to lowercase
		if word not in stopwords.words('english'):	# remove stop words
			query_words.append(word)
	
	query_vectors = []
	f = open(path_to_glove, "r")
	count_all_words = 0
	for line in f:
		vals = line.split()
		word = str(vals[0])
		if word in query_words:
			count_all_words += 1
			print(count_all_words, word)
			coefs = np.asarray(vals[1:], dtype='float32')
			coefs /= np.linalg.norm(coefs)
			word2vec_map[word] = coefs
		if count_all_words == len(query_words) -1:
			print("*** found all words in query ***")
			break

	long_enough = False
	while not long_enough:					# make up to 128 = times_steps size
		for word in query_words:
			query_vectors.append(word2vec_map[word])
			if len(query_vectors) == times_steps:
				long_enough = True
				break
	result = sess.run(correct_prediction, feed_dict={_inputs: [query_vectors],
													 _labels: [[0, 1]],
													 _seqlens: [times_steps]})
	print(result)
