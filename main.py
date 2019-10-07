  import numpy as np

class Neuron:
	def __init__(self):
		self.weight = random()
	
	def setInputValue(self, input):
		self.input = input
	
	def activationFunction(self, func):
		self.output = func(self.input)

class Layer: 
	def __init__(self, neuronCount):
		self.size = neuronCount
		for i in range(0, self.size){
			self.neuronList.append(Neuron())
		}
	
	def setInputValue(self, inputValueArray):
		for i in range(0, self.size){
			self.neuronList[i].setInputValue(inputValueArray[i])
		}

class Network:
	def __init__(self):
		
	def createNetwork(self):
	
	def launchNetwork(self):
		
if __name__ == ("__main__"):
	myNetwork = Network()
	
