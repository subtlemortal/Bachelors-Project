# -*- coding: utf-8 -*-
import os
import string



def getTrainData():
	pos,neg, traindata = [], [], []
	for filename in os.listdir("train"):
	    if filename == "POSITIVE.txt":
		    with open('train/'+filename) as f:
			    pos = [(tweet, 'No Depresion') for tweet in f.readlines()]
	    if filename == "NEGATIVE.txt":
		    with open('train/'+filename) as f:
			    neg = [(tweet, 'Depression Detected') for tweet in f.readlines()]

	for (words, sentiment) in pos + neg:
		words_filtered = [e for e in words.split() if len(e) > 2]
		traindata.append((words_filtered, sentiment))

	return traindata

def export(filename, data, p):
    with open(filename, p) as output:
    	for line in data:
        	output.write(line)
