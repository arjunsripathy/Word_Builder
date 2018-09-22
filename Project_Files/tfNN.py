import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import random

data = []
vocab = np.loadtxt("vocab.txt",dtype='string')
numV = len(vocab)
READ = False
PREDICT = True
RUNNING = False
LEARNING_RATE = 0.01

if(READ):
	datafile = open("words.txt","r")
	for line in datafile:
		dataValue = []
		for j in range(len(line)-1):
			c = line[j]
			dataValue.append(c)

		dataValue.append("END")
		data.append(dataValue)


def encode(s):
	encoding = []
	for i in range(len(s)):
		c = s[i]
		le = np.zeros((1,numV))
		le.itemset(np.where(vocab==c)[0][0],1)
		encoding.append(le)
	le = np.zeros((1,numV))
	le.itemset(0,1)
	encoding.append(le)
	return encoding

numH = 20
Whx = tf.Variable(tf.truncated_normal([numH,numV], stddev=0.1))
Whh = tf.Variable(tf.truncated_normal([numH,numH], stddev=0.1))
Wyh = tf.Variable(tf.truncated_normal([numV,numH], stddev=0.1))
letters = tf.placeholder(tf.int32)
wordEncoding = tf.placeholder(tf.float32,shape = [None,numV])
hidden = tf.zeros([numH,1])

cost = tf.zeros([1,1])

if(RUNNING):

	mp = ""
	for i in range(letters):
		tin = wordEncoding[i:i+1]
		hraw = tf.matmul(Whx,tin)+tf.matmul(Whh,hidden)
		hidden = tf.tanh(hraw)
		oraw = tf.matmul(Wyh,hidden)
		out = tf.nn.softmax(oraw)
		prod = tf.matmul(tf.transpose(wordEncoding[i+1:i+2]),out)
		correctProb = tf.reduce_sum(prod)
		cost-= tf.log(correctProb)
		print(cost)

		if(PREDICT):
			lp = tf.argmax(out)
			if(RUNNING):
				ml = vocab.item[lp]
				mp += ml


if(PREDICT and RUNNING):
	print("Prediction: %s", mp)

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

word = "banana"

RUNNING = True
train_step.run(feed_dict={wordEncoding: encode(word), letters: len(word)})





