# -*- coding: utf-8 -*-
"""
The main RNN algorithm.

Modeled after the code found in Ch.6 of "Learning Tensorflow" by Tom Hope et al.

@author: YKY
"""
# import zipfile
import numpy as np
import tensorflow as tf
import re

path_to_glove = "/data/wiki-news-300d-1M.vec"	# change to your path and filename
PRE_TRAINED = True
GLOVE_SIZE = 300				# dimension of word vectors in GloVe file
batch_size = 10
embedding_dimension = 64		# this is used only if PRE_TRAINED = False
num_classes = 2
hidden_layer_size = 32
times_steps = 2328				# this number should be same as fixed_seq_len below

# These are the 2 classes of laws:  nuisance and dangerous driving
class1_sentences = []
class2_sentences = []

# Read case examples from file
example1 = []
example2 = []

with open('class1_example.txt') as f:
	for line in f:
		line = re.sub(r'[^\w\s-]','',line)	# remove punctuations except hyphen
		example1 += line.split()
len1 = len(example1)
print("Example 1 length = ", len1)

with open('class2_example.txt') as f:
	for line in f:
		line = re.sub(r'[^\w\s-]','',line)
		example2 += line.split()
len2 = len(example2)
print("Example 2 length = ", len2)

seqlens = []
num_examples = 10				# change to larger later
fixed_seq_len = times_steps		# For each case law, we take N consecutive words from the text
for i in range(num_examples):
	# seqlens.append(fixed_seq_len)	# this is variable in the original code (with zero-padding), but now it's fixed because we don't use zero-padding
	max_len = max(len(example1), len(example2))
	seqlens.append(max_len)

	# rand_start1 = np.random.choice(range(0, len(example1) - fixed_seq_len))
	# rand_start2 = np.random.choice(range(0, len(example2) - fixed_seq_len))
	# class1_sentences.append(" ".join(example1[rand_start1: rand_start1 + fixed_seq_len]))
	# class2_sentences.append(" ".join(example2[rand_start2: rand_start2 + fixed_seq_len]))
	if len1 < max_len:
		# refill sequence 1 circularly
		sentence1 = ""
		for i in range(0, max_len):
			sentence1 += (" " + example1[i % len1])
		class1_sentences.append(sentence1)
		class2_sentences.append(" ".join(example2))
	else:
		sentence2 = ""
		for i in range(0, max_len):
			sentence2 += (" " + example2[i % len2])
		class1_sentences.append(" ".join(example1))
		class2_sentences.append(sentence2)

data = class1_sentences + class2_sentences
seqlens *= 2
labels = [1]*num_examples + [0]*num_examples
for i in range(len(labels)):
	label = labels[i]
	one_hot_encoding = [0]*2
	one_hot_encoding[label] = 1
	labels[i] = one_hot_encoding

word2index_map = {}
index = 0
for sent in data:
	for word in sent.split():
		if word not in word2index_map:
			word2index_map[word] = index
			index += 1

index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)
print("Vocabulary size = ", vocabulary_size)

def get_glove(path_to_glove, word2index_map):
	embedding_weights = {}
	count_all_words = 0
	f = open(path_to_glove, "r")
	for line in f:
		vals = line.split()
		word = str(vals[0])
		if word in word2index_map:
			print(count_all_words, word)
			count_all_words += 1
			coefs = np.asarray(vals[1:], dtype='float32')
			coefs /= np.linalg.norm(coefs)
			embedding_weights[word] = coefs
		if count_all_words == len(word2index_map) - 1:
			print("*** found all words ***")
			break
		if count_all_words >= 500:			# it takes too long to look up the entire dictionary, so I cut it short
			break
	return embedding_weights

word2embedding_dict = get_glove(path_to_glove, word2index_map)
embedding_matrix = np.zeros((vocabulary_size, GLOVE_SIZE))

zero_vector = np.asarray([0.00]*300, dtype='float32')	# this is for when the word-vector is not found in the file
for word, index in word2index_map.items():
	try:
		word_embedding = word2embedding_dict[word]
	except KeyError:
		word_embedding = zero_vector
	embedding_matrix[index, :] = word_embedding

# Split the data into Training and Testing sets, 50%:50%
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

def get_sentence_batch(batch_size, data_x,
					   data_y, data_seqlens):
	instance_indices = list(range(len(data_x)))
	np.random.shuffle(instance_indices)
	batch = instance_indices[:batch_size]
	x = [[word2index_map[word] for word in data_x[i].split()]
		 for i in batch]
	y = [data_y[i] for i in batch]
	seqlens = [data_seqlens[i] for i in batch]
	return x, y, seqlens


_inputs = tf.placeholder(tf.int32, shape=[batch_size, times_steps])
embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size,
													GLOVE_SIZE])

_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])

if PRE_TRAINED:
		embeddings = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, GLOVE_SIZE]),
								 trainable=True)
		# if using pre-trained embeddings, assign them to the embeddings variable
		embedding_init = embeddings.assign(embedding_placeholder)
		embed = tf.nn.embedding_lookup(embeddings, _inputs)

else:
		embeddings = tf.Variable(
			tf.random_uniform([vocabulary_size,
							   embedding_dimension],
							  -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, _inputs)

with tf.name_scope("biGRU"):
	with tf.variable_scope('forward'):
		gru_fw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
		gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell)

	with tf.variable_scope('backward'):
		gru_bw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
		gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell)

		outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell,
														  cell_bw=gru_bw_cell,
														  inputs=embed,
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

# extract the final state and use in a linear layer
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

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(embedding_init,
			 feed_dict={embedding_placeholder: embedding_matrix})
	for step in range(31):
		x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
															train_x, train_y,
															train_seqlens)
		sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch, _seqlens: seqlen_batch})

		if step % 5 == 0:
			acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
												_labels: y_batch,
												_seqlens: seqlen_batch})
			print("Accuracy at %d: %.5f" % (step, acc))

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),
								 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	normalized_embeddings_matrix = sess.run(normalized_embeddings)

	for test_batch in range(5):
		x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
														 test_x, test_y,
														 test_seqlens)
		batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
										 feed_dict={_inputs: x_test,
													_labels: y_test,
													_seqlens: seqlen_test})
		print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))

ref_word = normalized_embeddings_matrix[word2index_map["water"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][1:10]
for f in ff:
	print(index2word_map[f])
	print(cosine_dists[f])
