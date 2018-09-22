import numpy as np
import tensorflow as tf
import random

vocab = np.loadtxt("vocab.txt",dtype='string')
flpDist = np.loadtxt("flpDist")
numV = len(vocab)
LEARNING_RATE = 5e-4
numH = 30
wStdDev = 0.1
EPOCHS = 20

#0->Train
#1->Use
MODE = 0

NUMG = 10

USE_OLD = False
PRINT_EVERY = 3

def encode(s):
		encoding = []
		for i in range(len(s)):
			c = s[i]
			le = np.zeros(numV)
			le.itemset(np.where(vocab==c)[0][0],1)
			encoding.append(le)
		le = np.zeros(numV)
		le.itemset(0,1)
		encoding.append(le)
		return encoding

def decodeLetter(lE):
		index = np.where(lE==1)[0][0]
		return vocab[index]


datafile = open("words.txt","r")

data = []

for line in datafile:
	dataValue = []
	for j in range(len(line)-1):
		c = line[j]
		dataValue.append(c)
	data.append(dataValue)


random.shuffle(data)
numWords = len(data)


Wxh = tf.Variable(tf.truncated_normal([numH,numV],stddev=wStdDev))
Whh = tf.Variable(tf.truncated_normal([numH,numH],stddev=wStdDev))
Why = tf.Variable(tf.truncated_normal([numV,numH],stddev=wStdDev))

word = tf.placeholder(tf.float32,shape=[None,numV])
expected = tf.placeholder(tf.float32,shape=[None,numV])

initialHidden = tf.zeros([numH,1])


def recur(hidden,letter):
	letter = tf.reshape(letter,[numV,-1])
	hContrib = tf.matmul(Whh,hidden)
	xContrib = tf.matmul(Wxh,letter)
	hInput = tf.add(hContrib,xContrib)
	h = tf.tanh(hInput)
	return h

hiddenStates = tf.scan(recur,word,initialHidden)
hiddenStates = tf.reshape(hiddenStates,[-1,numH])
hiddenStatesT = tf.transpose(hiddenStates)

sInputT = tf.matmul(Why,hiddenStatesT)
sInput = tf.transpose(sInputT)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=expected, logits=sInput))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(expected, 1), tf.argmax(sInput, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
	if(USE_OLD):
		saver.restore(sess,"/tmp/tfWB.ckpt")
	else:
		sess.run(tf.global_variables_initializer())

	if(MODE==0):
		for i in range(EPOCHS):
			tC = 0.0
			tA = 0.0
			if(i%(PRINT_EVERY)==0):
				for j in range(numWords):
					w = data[j]
					encoding = encode(w)
					wE = encoding[:len(encoding)-1]
					expec = encoding[1:]

					[ceCost,acc] = sess.run([cross_entropy,accuracy],feed_dict={word:wE,expected:expec})
					tC+=ceCost
					tA+=acc
				tC/=numWords
				tA/=numWords
				print("EPOCH %i: Cost: %f, Accuracy: %f"%(i,tC,tA))

			for i in range(len(data)):
				w = data[i]
				encoding = encode(w)
				wE = encoding[:len(encoding)-1]
				expec = encoding[1:]

				train_step.run(feed_dict={word: wE, expected: expec})
	
	if(MODE==1):
		Wxh = sess.run(Wxh)
		Whh = sess.run(Whh)
		Why = sess.run(Why)

		def generateNext(prevHidden,inpLetter):

			rowEncoding = encode(inpLetter)[0]

			lE = np.reshape(rowEncoding,(np.size(rowEncoding),1))

			prevHiddenComponent = np.matmul(Whh,prevHidden)
			inpComponent = np.matmul(Wxh,lE)

			thisHidden = np.tanh(np.add(prevHiddenComponent,inpComponent))

			softmaxInput = np.matmul(Why,thisHidden)

			meanInput = np.mean(softmaxInput)
			normalizedInput = np.subtract(softmaxInput,meanInput)
			unNormalizedProbs = np.exp(softmaxInput)
			'''SKETCHY SQUARE'''
			unNormalizedProbs = unNormalizedProbs*unNormalizedProbs
			''''SKETCHY SQUARE'''
			sumUnNormalizedProbs = np.sum(unNormalizedProbs)

			normalizedProbs = unNormalizedProbs/sumUnNormalizedProbs

			pickFromDist = np.random.choice(vocab,p=np.reshape(normalizedProbs,numV))

			return [thisHidden,pickFromDist]

		def generateWord():
			firstLetter = np.random.choice(vocab,p=flpDist)
			word = ""
			word+=firstLetter
			hiddenState, nextLetter = generateNext(np.zeros([numH,1]),firstLetter)
			while nextLetter != "END":
				word+=nextLetter
				hiddenState, nextLetter = generateNext(hiddenState,nextLetter)
			return word

		for i in range(NUMG):
			print(generateWord())


	saver.save(sess, "/tmp/tfWB.ckpt")





