#!/usr/bin/python3

#note: i really have no idea what i'm doing!
import random;
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import math

#train and get predictions and error margin from our model
def testmodel(model,data,lables):
    print("------------------------------------------------")
    print("testing model : ", model.__class__.__name__)

    #train the model with half of the total dataset
    model.fit(data[:0x8000],labels[:0x8000])

    #just to get a peek at what's actually predicted
    out = model.predict([pair_to_bits(1,1), pair_to_bits(1,6)])
    print("predicted should be [2., 7.]: ", out)

    #get the average error
    predictions = model.predict(data[0x8000:])
    lin_mse = mean_squared_error(labels[0x8000:],predictions)
    print("error margin ",np.sqrt(lin_mse))

#after initial tests where the data set was two 8 bit ints
#i wanted to find what the models would make of the data
#if i split it into 16 individual bits this function does
#the conversion
def pair_to_bits(x, y):
    datum = list()
    for i in range(0,8):
        #you can see below that the bits of the two integers
        #are interleaved, but that shouldn't matter to the 
        #machine learning models as long as the order is 
        #consistant
        datum.append( (x>>i) & 1)
        datum.append( (y>>i) & 1)
    return datum

#generate the data set
pairs = list()
labels = list()
for x in range(1,256):
    for y in range(1,256):
        pairs.append([x,y])

#shuffle the data set ideally any subset of the data should
#have a fair random sample of the data, shuffle before 
#generating lables because they lables need to line up with
#the data
random.shuffle(pairs)

#generate labels and transform the data set from 2 ints to
#16 bits
bits_data=list()
for x in pairs:
    labels.append(x[0] + x[1])
    datum = pair_to_bits(x[0],x[1])
    bits_data.append(datum)

#construct default ML models nothing fancy here
lin_reg = LinearRegression()
d_tree_reg = DecisionTreeRegressor()
n_net = MLPRegressor(max_iter=1000)

#all of the models have the same interface so the testing 
#function doesn't even need to care which model it's using
#yay polymorphism

testmodel(lin_reg,bits_data,labels)
testmodel(d_tree_reg,bits_data,labels)
testmodel(n_net,bits_data,labels)


