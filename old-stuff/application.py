# -*- coding: utf-8 -*-
"""
The main RNN algorithm.

Updates:
1. RNN _input layer now changed to word-vector encoding
2. can answer single queries
3. using case-law full-texts as training data

Modeled after the code found in Ch.6 of "Learning Tensorflow" by Tom Hope et al.

@author: YKY, Jesmer Wong, Raymond Luk
"""
from flask import Flask, request, jsonify
import os						# for os.listdir and os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	# suppress Tensorflow warnings
import re						# for removing punctuations
import sys						# for sys.stdin.readline()
import json
import zipfile
from collections import defaultdict	# for default value of word-vector dictionary
import pickle
import zipfile
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords

reload(sys)
sys.setdefaultencoding('utf-8')
 
app = Flask(__name__)

# =================== Prepare batch data ============================

def get_sentence_batch(batch_size, data_x, data_y):
	instance_indices = list(range(len(data_x)))
	np.random.shuffle(instance_indices)
	batch = instance_indices[:batch_size]
	x = [[word2vec_map[word]
			for word in data_x[i].split()]
			for i in batch]
	y = [data_y[i] for i in batch]
	return x, y

@app.route('/result',methods = ['GET'])			# define router
def hello():
    search = request.args.get("queryword")

	# ===================== initialize constants =====================
	
    path_to_glove = "/home/ec2-user/eb-flask1/glove.840B.300d.zip"
    GLOVE_SIZE = 300
	batch_size = 512
	num_classes = 3
	hidden_layer_size = 64
	times_steps = 32				# this should be same as fixed_seq_len below
    
	""" These are the 3 classes of laws:
	* nuisance
	* dangerous driving
	* OTHERWISE, or work injuries
	"""
	categories = ["nuisance", "dangerous-driving", "injuries"]
	answers = ["nuisance", "dangerous driving", "work injuries"]

	# =================== Read prepared training data from file ===================

	pickle_off = open("training-data2.pickle", "rb")
	data = pickle.load(pickle_off)

	pickle_off = open("training-labels2.pickle", "rb")
	labels = pickle.load(pickle_off)

	pickle_off = open("training-word-list2.pickle", "rb")
	word_list = pickle.load(pickle_off)

	pickle_off = open("training-word2vec-map2.pickle", "rb")
	word2vec_map = pickle.load(pickle_off)

	# set default value = zero vector, if word not found in dictionary
	zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
	word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

	num_examples = len(data)
	print "# training examples = ", num_examples
	print "# unique words = ", len(word_list)
	print "# vectorized words = ", len(word2vec_map)

	fixed_seq_len = times_steps

	# ============ Split data into Training and Testing sets, 50%:50% ============

	data_indices = list(range(len(data)))
	np.random.shuffle(data_indices)
	data = np.array(data)[data_indices]
	labels = np.array(labels)[data_indices]
	midpoint = num_examples // 2
	train_x = data[:midpoint]
	train_y = labels[:midpoint]

	test_x = data[midpoint:]
	test_y = labels[midpoint:]

	# ========= define input, output, and RNN structure =========

	_inputs = tf.placeholder(tf.float32, shape=[None, times_steps, GLOVE_SIZE])
	_labels = tf.placeholder(tf.float32, shape=[None, num_classes])

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
															  dtype=tf.float32,
															  scope="biGRU")
	states = tf.concat(values=states, axis=1)
	weights = {
		'linear_layer': tf.Variable(tf.truncated_normal([2*hidden_layer_size,
														num_classes],
														mean=0, stddev=.01))}
	biases = {
		'linear_layer': tf.Variable(tf.truncated_normal([num_classes],
														mean=0, stddev=.01))}

	# ========== Define final state and objective function ================

	final_output = tf.matmul(states,
							 weights["linear_layer"]) + biases["linear_layer"]

	softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output,
													  labels=_labels)
	cross_entropy = tf.reduce_mean(softmax)

	train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(_labels, 1),
								  tf.argmax(final_output, 1))
	accuracy = (tf.reduce_mean(tf.cast(correct_prediction,
									   tf.float32)))*100

	# ================== Run the session =====================

	print "\n**** Training RNN...."
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for step in range(600 + 1):
			x_batch, y_batch = get_sentence_batch(batch_size,
												train_x, train_y)
			sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch})

			if step % 25 == 0:
				acc = sess.run(accuracy, feed_dict={_inputs: x_batch, _labels: y_batch})
				print "Accuracy at {0:3d}: {1:5f} %".format(step, acc)

		for test_batch in range(5):
			x_test, y_test = get_sentence_batch(batch_size, test_x, test_y)
			batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
									feed_dict={_inputs: x_test, _labels: y_test})
			print "Test batch accuracy {0:3d}: {1:5f} %".format(test_batch, batch_acc)

		# =================== Process a single query ===================

		while True:
			print "----------------------------\n? ",
			sys.stdout.flush()
			query = sys.stdin.readline()
			query = re.sub(r'[^\w\s-]',' ', query)	# remove punctuations except hyphen
			query_words = []
			for word in query.lower().split():		# convert to lowercase
				if word not in stopwords.words('english'):	# remove stop words
					query_words.append(word)
			
			query_vectors = []
			with zipfile.ZipFile("glove.840B.300d.zip") as z:
				with z.open("glove.840B.300d.txt") as glove_file:
					count_all_words = 0
					entry_number = 0
					for word_entry in glove_file:
						vals = word_entry.split()
						word = str(vals[0])
						entry_number += 1
						if word in query_words:
							count_all_words += 1
							print count_all_words, word, "\r",
							coefs = np.asarray(vals[1:], dtype='float32')
							coefs /= np.linalg.norm(coefs)
							word2vec_map[word] = coefs
						if count_all_words == len(word_list) - 1:
							break
						if entry_number > 80000:
							# took too long to find the words
							break

			long_enough = False
			while not long_enough:					# make up to 128 = times_steps size
				for word in query_words:
					query_vectors.append(word2vec_map[word])
					if len(query_vectors) == times_steps:
						long_enough = True
						break
			result = sess.run(tf.argmax(final_output, 1), feed_dict={_inputs: [query_vectors]})
			answer = answers[result[0]]
			print " ⟹  broad category: ", answer

    return jsonify({'case': answer})

if __name__ == "__main__":
    app.run()
