import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import random

data = []
vocab = np.loadtxt("vocab.txt",dtype='string')
flpDist = np.loadtxt("flpDist")
numV = len(vocab)
READ = True
PREDICT = True
RUNNING = False
LEARNING_RATE = 8e-3
MB_Size = 1
EPOCHS = 5
PRINT = False
MOMENTUM  = 0.9

#0->Train
#1->Use
MODE = 1
USE_OLD = True

NUMG = 10

numH = 25

def encode(s):
		encoding = []
		for i in range(len(s)):
			c = s[i]
			le = np.zeros(numV)
			if(np.size(np.where(vocab==c))==0):
				print(c)
			le.itemset(np.where(vocab==c)[0][0],1)
			encoding.append(le)
		le = np.zeros(numV)
		le.itemset(0,1)
		encoding.append(le)
		return encoding

def decodeLetter(lE):
		index = np.where(lE==1)[0][0]
		return vocab[index]

if(MODE==0):
	print("Train")

	if(READ):
		datafile = open("words.txt","r")
		for line in datafile:
			dataValue = []
			for j in range(len(line)-1):
				c = line[j]
				dataValue.append(c)


			dataValue.append("END")
			data.append(dataValue)


	random.shuffle(data)

	data = data

	numWords = len(data)

	'''Initialization'''

	if(USE_OLD):
		Whx = np.loadtxt("Whx.txt")
		Whh = np.loadtxt("Whh.txt")
		Wyh = np.loadtxt("Wyh.txt")
	else:
		Whx = np.random.normal(scale=0.1,size=[numH,numV])
		Whh = np.random.normal(scale=0.1,size=[numH,numH])
		Wyh = np.random.normal(scale=0.1,size=[numV,numH])



	'''Forward Pass'''

	def forwardPass(wordEncoding,letters):
		hStates = []
		hidden = np.zeros([numH,1])
		hStates.append(hidden)

		softmaxOutputs = []

		cost = 0
		for i in range(letters):
			letterEncoding = np.reshape(wordEncoding[i],(numV,1))

			hiddenInput = np.add(np.matmul(Whx,letterEncoding),np.matmul(Whh,hidden))

			hidden = np.tanh(hiddenInput)

			hStates.append(hidden)

			softmaxInput = np.matmul(Wyh,hidden)

			meanInput = np.mean(softmaxInput)
			normalizedInput = np.subtract(softmaxInput,meanInput)
			unNormalizedProbs = np.exp(softmaxInput)
			sumUnNormalizedProbs = np.sum(unNormalizedProbs)

			normalizedProbs = unNormalizedProbs/sumUnNormalizedProbs
			softmaxOutputs.append(normalizedProbs)

			correctLetterEncoding = np.reshape(wordEncoding[i+1],(numV,1))
			correctProb = np.sum(np.matmul(np.transpose(normalizedProbs),correctLetterEncoding))

			cost -= np.log(correctProb)


		return [cost,hStates,softmaxOutputs]

	'''Back Propogation'''
	def getHidden(hiddenStates,i):
		return hiddenStates[i+1]

	def hxG(wordEncoding, t, nhiG):
		inLetterEncoding = wordEncoding[t]
		return np.matmul(nhiG,np.reshape(inLetterEncoding,(1,numV)))


	def hhG(hiddenStates, t,nhiG):
		prevH = getHidden(hiddenStates,t-1)
		return np.matmul(nhiG,np.reshape(prevH,(1,numH)))


	def backprop(wordEncoding, letters, hiddenStates,softmaxOutputs):
		gWhx = np.zeros(shape = [numH,numV])
		gWhh = np.zeros(shape = [numH,numH])
		gWyh = np.zeros(shape = [numV,numH])

		for i in range(letters):
			time = i
			correctPLetterEncoding = np.reshape(wordEncoding[time+1],(numV,1))

			oG = np.subtract(softmaxOutputs[time],correctPLetterEncoding)

			hState = getHidden(hiddenStates,time)
			gWyh += np.matmul(oG,np.transpose(hState))

			hG = np.matmul(np.transpose(Wyh),oG)
			hinG = hG*(1-np.power(hState,2))

			gWhx += hxG(wordEncoding,time,hinG)
			gWhh += hhG(hiddenStates,time,hinG)

			for j in range(i):
				time -= 1

				hState = getHidden(hiddenStates,time)
				hG = np.matmul(np.transpose(Whh),hinG)
				hinG = hG*(1-np.power(hState,2))

				gWhx += hxG(wordEncoding,time,hinG)
				gWhh += hhG(hiddenStates,time,hinG)

		return [gWhx,gWhh,gWyh]

	'''Update'''

	c = []
	yhx = []
	yhh = []
	yyh = []

	for j in range(EPOCHS):
		print("Epoch %d"%(j))
		cost = 0
		for i in range(numWords):
			wordEncoding = encode(data[i])
			wLength = len(wordEncoding)-1

			fpResults = forwardPass(wordEncoding, wLength)
			cost += fpResults[0]
		cost /= float(numWords)
		c.append(cost)
		print(cost)

		yhx.append(np.mean(np.abs(Whx)))
		yhh.append(np.mean(np.abs(Whh)))
		yyh.append(np.mean(np.abs(Wyh)))

		prevgWhx = np.zeros([numH,numV])
		prevgWhh = np.zeros([numH,numH])
		prevgWyh = np.zeros([numV,numH])

		gWhx = np.zeros([numH,numV])
		gWhh = np.zeros([numH,numH])
		gWyh = np.zeros([numV,numH])
		for i in range(numWords):

			if j%50 == 0:
				PRINT=True

			wordEncoding = encode(data[i])
			wLength = len(wordEncoding)-1

			fpResults = forwardPass(wordEncoding, wLength)

			weightGradients = backprop(wordEncoding,wLength,fpResults[1],fpResults[2])
			gWhx+= weightGradients[0]
			gWhh+= weightGradients[1]
			gWyh+= weightGradients[2]


			if (i+1) % MB_Size == 0:

				momCWhx = np.dot(MOMENTUM,prevgWhx)
				newCWhx = np.dot((1-MOMENTUM),gWhx)
				uWhx = np.add(momCWhx,newCWhx)

				momCWhh = np.dot(MOMENTUM,prevgWhh)
				newCWhh = np.dot((1-MOMENTUM),gWhh)
				uWhh = np.add(momCWhh,newCWhh)

				momCWyh = np.dot(MOMENTUM,prevgWyh)
				newCWyh = np.dot((1-MOMENTUM),gWyh)
				uWyh = np.add(momCWyh,newCWyh)


				BLR = LEARNING_RATE
				Whx -= np.dot(BLR,uWhx)
				#Whx = np.multiply(Whx,0.99)
				Whh -= np.dot(BLR,uWhh)
				#Whh = np.multiply(Whh,0.99)
				Wyh -= np.dot(BLR,uWyh)
				#Wyh = np.multiply(Wyh,0.99)

				prevgWhx = gWhx
				prevgWhh = gWhh
				prevgWyh = gWyh
				gWhx = np.zeros([numH,numV])
				gWhh = np.zeros([numH,numH])
				gWyh = np.zeros([numV,numH])

			PRINT = False


	xPlot = []
	for i in range(0,EPOCHS):xPlot.append(i)

	plt.plot(xPlot, c, 'b-', label='cost')
	plt.xlabel('Epoch')
	plt.ylabel('cost')
	plt.legend()
	plt.show()

	#Save weights
	np.savetxt("Whx.txt",Whx)
	np.savetxt("Whh.txt",Whh)
	np.savetxt("Wyh.txt",Wyh)

if(MODE==1):
	print("Use")

	Whx = np.loadtxt("Whx.txt")
	Whh = np.loadtxt("Whh.txt")
	Wyh = np.loadtxt("Wyh.txt")

	def generateNext(prevHidden,inpLetter):

		#only first because encode adds 'END'
		rowEncoding = encode(inpLetter)[0]

		lE = np.reshape(rowEncoding,(np.size(rowEncoding),1))

		prevHiddenComponent = np.matmul(Whh,prevHidden)
		inpComponent = np.matmul(Whx,lE)

		thisHidden = np.tanh(np.add(prevHiddenComponent,inpComponent))

		softmaxInput = np.matmul(Wyh,thisHidden)

		meanInput = np.mean(softmaxInput)
		normalizedInput = np.subtract(softmaxInput,meanInput)
		unNormalizedProbs = np.exp(softmaxInput)
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





