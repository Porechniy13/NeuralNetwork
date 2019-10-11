import numpy as np
from PIL import Image
import math
import hickle as hkl

def translate(x):
	result = np.zeros(64)
	counter = 0
	for i in range(8):
		for j in range(8):
			pixel = int(x.getpixel((i,j))[0])
			result[counter] = pixel
			counter += 1
	return result

def sigmoid(x, deriv=False):
	if (deriv==True):
		return sigmoid(x)*(1-sigmoid(x))
	return 1/(1+np.exp(-x))

class Network:
	def __init__(self, inputSet, needResult):
		self.input = inputSet
		self.firstWeights = np.random.random((64, 64))
		self.secondWeights = np.random.random((64, 1))
		self.outputLayer = needResult
	
	def setInput(self, inputSet):
		self.input = inputSet
	
	def feedForward(self):
		self.hiddenLayer = sigmoid(np.dot(self.input, self.firstWeights))
		self.output = sigmoid(np.dot(self.hiddenLayer,self.secondWeights))
	
	def backLoss(self):
		secondError = self.outputLayer - self.output
		d_secondWeights = secondError * sigmoid(self.output,True)
		d_firstWeights = d_secondWeights.dot(self.secondWeights.T) * sigmoid(self.hiddenLayer,True)
		
		self.secondWeights += self.hiddenLayer.T.dot(d_secondWeights)
		self.firstWeights += self.input.T.dot(d_firstWeights)
		
	def saveWeights(self):
		hkl.dump(self.firstWeights, "data/firstLayer.hkl")
		hkl.dump(self.secondWeights, "data/secondLayer.hkl")
	
	def loadWeights(self):
		self.firstWeights = hkl.load("data/firstLayer.hkl")
		self.secondWeights = hkl.load("data/secondLayer.hkl")

if __name__ == ("__main__"):
	n = 10
	template = '{:.' + str(n) + 'f}'
	number1 = Image.open("numbersSet/1.png")
	number2 = Image.open("numbersSet/2.png")
	number3 = Image.open("numbersSet/3.png")
	number4 = Image.open("numbersSet/4.png")
	number5 = Image.open("numbersSet/5.png")
	number6 = Image.open("numbersSet/6.png")
	number7 = Image.open("numbersSet/7.png")
	number8 = Image.open("numbersSet/8.png")
	number9 = Image.open("numbersSet/9.png")
	number0 = Image.open("numbersSet/0.png")
	number11 = Image.open("numbersSet/11.png")
	number22 = Image.open("numbersSet/22.png")
	number33 = Image.open("numbersSet/33.png")
	
	testNumber = Image.open("numbersSet/test.png")
	
	trainSet = np.array([translate(number1), 
						 translate(number2), 
						 translate(number3),
						 translate(number4), 
						 translate(number5), 
						 translate(number6),
						 translate(number7), 
						 translate(number8), 
						 translate(number9),
						 translate(number0),
						 translate(number11), 
						 translate(number22), 
						 translate(number33)
						 ])
	result = np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0,0.1,0.2,0.3]]).T
	
	myNetwork = Network(trainSet, result)
	myNetwork.loadWeights()	
	myNetwork.feedForward()		
	myNetwork.saveWeights()
	
	print("Learning outputs:")
	print(template.format(myNetwork.output[0][0]))
	print(template.format(myNetwork.output[1][0]))
	print(template.format(myNetwork.output[2][0]))
	print(template.format(myNetwork.output[3][0]))
	print(template.format(myNetwork.output[4][0]))
	print(template.format(myNetwork.output[5][0]))
	print(template.format(myNetwork.output[6][0]))
	print(template.format(myNetwork.output[7][0]))
	print(template.format(myNetwork.output[8][0]))
	print(template.format(myNetwork.output[9][0]))
	print("--------------------------------------")
	
	myNetwork.saveWeights()
	
	testSet = np.array([translate(testNumber)])
	myNetwork.setInput(testSet)
	myNetwork.feedForward()
	
	print("Test outputs: " + template.format(myNetwork.output[0][0]))
	
	
