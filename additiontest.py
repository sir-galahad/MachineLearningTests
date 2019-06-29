#!/usr/bin/python3

import random;
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import math

def testmodel(model,data,lables):
    model.fit(data[:0x8000],labels[:0x8000])

    out = model.predict([pair_to_bits(1,1), pair_to_bits(1,6)])

    print("predicted 1+1 =", out)
    predictions = model.predict(data[0x8000:])
    lin_mse = mean_squared_error(labels[0x8000:],predictions)

    print("error margin ",np.sqrt(lin_mse))

def pair_to_bits(x, y):
    datum = list()
    for i in range(0,8):
        datum.append( (x>>i) & 1)
        datum.append( (y>>i) & 1)
    return datum

pairs = list()
labels = list()
for x in range(1,256):
    for y in range(1,256):
        pairs.append([x,y])

random.shuffle(pairs)

bits_data=list()
for x in pairs:
    labels.append(x[0] + x[1])
    datum = pair_to_bits(x[0],x[1])
    bits_data.append(datum)
        
lin_reg = LinearRegression()
d_tree_reg = DecisionTreeRegressor()
n_net = MLPRegressor(max_iter=1000)


testmodel(lin_reg,bits_data,labels)
testmodel(d_tree_reg,bits_data,labels)
testmodel(n_net,bits_data,labels)
