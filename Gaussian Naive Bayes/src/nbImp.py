# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:18:03 2018

@author: AnupamaKesari
"""
import math

def accuracy(testData, predictions):
	correct = 0
	if(len(testData) != len(predictions)):
		print("Both lengths have to be equal.")
		return 0
	for i in range(len(testData)):
		if testData[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testData))) * 100.0

def stdev(numbers):
	avg = sum(numbers)/float(len(numbers))
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def meanSDofClass(dataset):
	separatedClassWise = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separatedClassWise):
			separatedClassWise[vector[-1]] = []
		separatedClassWise[vector[-1]].append(vector)
	meanSDs = {}
	for label, records in separatedClassWise.items():
		meanSD1 = [((sum(attribute)/float(len(attribute))), stdev(attribute),len(records)/len(dataset)) for attribute in zip(*records)]
		del meanSD1[-1]
		meanSDs[label] = meanSD1       
	return meanSDs

def classLikelihoods(meanSDs, testData):
	probabilities = {}
	for label, classMeanSD in meanSDs.items():
		probabilities[label] = 1
		for i in range(len(classMeanSD)):
			mean, stdev, prior = classMeanSD[i]
			x = testData[i]
			if stdev != 0:
				probabilities[label] *= prior * (1 / (math.sqrt(2*math.pi) * stdev)) * (math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2)))))
			else:
				stdev = 0.01            
				probabilities[label] *= prior * (1 / (math.sqrt(2*math.pi) * stdev)) * (math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2)))))
	return probabilities
			
def predict(meanSDs, testData):
	predictions = []
	for i in range(len(testData)):
		probabilities = classLikelihoods(meanSDs, testData[i])
		bestLabel, highestProb = None, -1
		for label, probability in probabilities.items():
			if bestLabel is None or probability > highestProb:
				highestProb = probability
				bestLabel = label
		result = bestLabel
		predictions.append(result)
	return predictions

