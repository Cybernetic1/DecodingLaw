# -*- coding: utf-8 -*-
"""
* Find all file names, put them in list
* For each file do:
	extract yellow stuff
	extract green stuff
"""
import os

for category in ["nuisance", "dangerous-driving", "injuries"]:

	for filename in os.listdir("laws-TXT/" + category):

		f = open("laws-TXT/" + category + "/" + filename, "r")
		data = f.read()
		f.close()

		"""
		Algorithm:
		* Find <<
		* Scan to >>
		* append to buffer
		"""
		buffer = ""
		while True:
			i = data.find("<Y<")
			if i == -1:
				break
			data = data[i + 3:]
			j = data.find(">Y>")
			if j == -1:
				print("Missing closing >Y>")
				exit(0)
			buffer += data[:j]
			buffer += ' '
			data = data[j + 3:]

		f = open("laws-TXT/" + category + "-YELLOW/" + filename, "w+")
		f.write(buffer)
		f.close()

		# second verse, same as first verse

		f = open("laws-TXT/" + category + "/" + filename, "r")
		data = f.read()
		f.close()

		buffer = ""
		while True:
			i = data.find("<G<")
			if i == -1:
				break
			data = data[i + 3:]
			j = data.find(">G>")
			if j == -1:
				print("Missing closing >G>")
				exit(0)
			buffer += data[:j]
			buffer += ' '
			data = data[j + 3:]

		f = open("laws-TXT/" + category + "-GREEN/" + filename, "w+")
		f.write(buffer)
		f.close()

