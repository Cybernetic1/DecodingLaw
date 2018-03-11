# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
# from flask_cors import CORS

import argparse, time
# from load import load_graph

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

pickle_off = open("training-word-list2.pickle", "rb")
word_list = pickle.load(pickle_off)

pickle_off = open("training-word2vec-map2.pickle", "rb")
word2vec_map = pickle.load(pickle_off)

# set default value = zero vector, if word not found in dictionary
zero_vector = np.asarray([0.0] * GLOVE_SIZE, dtype='float32')
word2vec_map = defaultdict(lambda: zero_vector, word2vec_map)

# =================== Process a single query ===================

print "Re-starting Tensorflow session...."
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('TF-model/data-all.meta')
	saver.restore(sess, 'TF-model/data-all')
	tf.global_variables_initializer().run()
	graph = tf.get_default_graph()
	final_output = graph.get_tensor_by_name("final_output:0")
	in_vecs = graph.get_tensor_by_name("in_vecs:0")

	while True:
		print "----------------------------\n? ",
		sys.stdout.flush()
		query = sys.stdin.readline()
		query = re.sub(r'[^\w\s-]',' ', query)	# remove punctuations except hyphen
		query_words = []
		for word in query.lower().split():		# convert to lowercase
			if word not in stopwords.words('english'):	# remove stop words
				query_words.append(word)

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
					if entry_number > 75000:
						# took too long to find the words
						break

		query_vectors = []
		long_enough = False
		while not long_enough:					# make up to 128 = times_steps size
			for word in query_words:
				query_vectors.append(word2vec_map[word])
				if len(query_vectors) == times_steps:
					long_enough = True
					break
		# print in_vecs.get_shape()
		result = sess.run(tf.argmax(final_output, 1),	\
			feed_dict={in_vecs: np.expand_dims(query_vectors, axis=0)})
		answer = answers[result[0]]
		print "==>  broad category = ", answer

exit(0)

##################################################
# API part
##################################################

#print "Starting router...."
#@app.route('/result',methods = ['GET'])			# define router
#def hello():
#	search = request.args.get("queryword")
#	return jsonify({'case': answer})

app = Flask(__name__)
cors = CORS(app)
@app.route("/result", methods=['POST'])
def predict():
	start = time.time()

	data = request.data.decode("utf-8")
	if data == "":
		params = request.form
		x_in = json.loads(params['x'])
	else:
		params = json.loads(data)
		x_in = params['x']

	##################################################
	# Tensorflow part
	##################################################
	y_out = persistent_sess.run(y, feed_dict={
		x: x_in
		# x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
	})
	##################################################
	# END Tensorflow part
	##################################################
	
	json_data = json.dumps({'y': y_out.tolist()})
	print("Time spent handling the request: %f" % (time.time() - start))
	
	return json_data
##################################################
# END API part
##################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
	parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
	args = parser.parse_args()

	##################################################
	# Tensorflow part
	##################################################
	print('Loading the model')
	graph = load_graph(args.frozen_model_filename)
	x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
	y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')

	print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
	sess_config = tf.ConfigProto(gpu_options=gpu_options)
	persistent_sess = tf.Session(graph=graph, config=sess_config)
	##################################################
	# END Tensorflow part
	##################################################

	print('Starting the API')
	app.run()
