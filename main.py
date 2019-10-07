import numpy as np

class Neuron:
	def __init__(self):
		self.weight = []
	
	def initWeight(self, count):
		for i in range(count):
			self.weight[i] = random()
	
	def setInputValue(self, input):
		self.input = input
	
	def activationFunction(self, func):
		self.output = func(self.input)

class Layer: 
	def __init__(self, neuronCount):
		self.size = neuronCount
		self.neuronList = []
		for i in range(self.size):
			self.neuronList[i] = Neuron() 
	
	def setInputValue(self, inputValueArray):
		for i in range(0, self.size):
			self.neuronList[i].setInputValue(inputValueArray[i])

class Network:
	def __init__(self, counter, neuronCount):
		self.Layers = []
		self.size = counter
		for i in range(self.size):
			self.Layers[i] = Layer(neuronCount[i])
	
	def networkInit():
		for i in range(self.size):
			for j in range(0,self.Layers[i].size-1):
				self.Layers[i].neuronList[j].initWeigth(self.Layers[i+1].size)			
	
	def printNetwork():
		for i in range(self.size):
			for j in range(self.Layers[i].size):
				print(self.Layers[i].neuronList[j].weight)
	
if __name__ == ("__main__"):
	neuronsCounter = [4,5,1]
	myNetwork = Network(len(neuronsCounter), neuronsCounter)
	myNetwork.networkInit()
	myNetwork.printNetwork()
	
