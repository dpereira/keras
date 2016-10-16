"""
As seen on: http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

"""


import numpy

from keras.models import Sequential
from keras.layers import Dense


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
