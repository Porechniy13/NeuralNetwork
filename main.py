import numpy as np
import math
import random

def sigmoid(x):
	return 1 / (1 + math.exp(-x))
	
def d_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

class Network:
	def __init__(self, inputSet, y):
		self.input = inputSet
		self.firstWeights = np.random.rand(self.input.shape[1], 4)
		self.secondWeights = np.random.rand(4,1)
		self.outputLayer = y
		self.output = np.zeros(self.outputLayer.shape)
		
	def feedForward(self):
		self.firstLayer = sigmoid(np.dot(self.input, self.firstWeights))
		self.output = sigmoid(np.dot(self.firstLayer, self.secondWeights))
	
	def backLoss(self):
		d_firstWeights = np.dot(self.firstLayer.T, (2*(self.outputLayer - self.output) * d_sigmoid(self.output)))
		d_secondWeights = np.dot(self.input.T, (np.dot(2*(self.outputLayer - self.output) * d_sigmoid(self.output), self.secondWeights.T) * d_sigmoid(self.firstLayer)))
		
		self.firstWeights += d_firstWeights
		self.secondWeights += d_secondWeights
	
	
if __name__ == ("__main__"):
	trainSet = np.array([[0,0],[0,1],[1,0],[1,1]])
	result = np.array([0,1,1,0])
	myNetwork = Network(trainSet, result)
	for i in range(100):
		myNetwork.feedForward()
		myNetwork.backLoss()
	
	print(myNetwork.output)
	
	
