# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import numpy as np
import time
import datetime
from preprocess import read_file
from preprocess import write_file
stemmer = LancasterStemmer()

classes = []
training = []
output = []
words = []

folders = os.listdir('resources/data/processed')
dictionary = read_file('resources/dictionary').split(', ')
words = dictionary
count = 0

for folder_name in folders:
    train_files = os.listdir(f'resources/data/processed/{folder_name}')
    classes.append(folder_name)

    for file in train_files:
        text = read_file(f'resources/data/processed/{folder_name}/{file}').split(', ')
        if len(text)<= 0:
            continue
        line = [0] * len(dictionary)
        for word in text:
            count += 1
            if word == '':
                continue
            try:
                line[dictionary.index(word)] += 1
                break;
            except:
                print('error: ', file)
                print('word: ', word)
##                print('index: ', dictionary.index(word))
                print('len: ', len(text))
                raise
        training.append(line)
        output_new = [0] * 20
        output_new[classes.index(folder_name)] = 1
        output.append(output_new)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# ANN and Gradient Descent code from https://iamtrask.github.io//2015/07/27/python-network-part2/
def train(X, y, alpha=1, epochs=10000, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (0, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), len(classes))) - 1

    # prev_synapse_0_weight_update = np.zeros_like(synapse_0, 'float32')

    # synapse_0_direction_count = np.zeros_like(synapse_0,'float32')
        
    layer_0 = X
    for j in iter(range(epochs+1)):
        # Feed forward through layers 0, 1, and 2
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),len(classes), 'float32'))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))


        # how much did we miss the target value?
        layer_1_error = y - layer_1

        if (j% 1000) == 0:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_1_error), dtype='float32') < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_1_error), dtype='float32')) )
                last_mean_error = np.mean(np.abs(layer_1_error), dtype='float32')
            else:
                print ("break:", np.mean(np.abs(layer_1_error), dtype='float32'), ">", last_mean_error )
                break

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        # if(j > 0):
        #     synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
        
        synapse_0 += alpha * synapse_0_weight_update
        
        # prev_synapse_0_weight_update = synapse_0_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses-ex.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)

print('training leng: ', len(training))
print('ouput len: ', len(output))
X = np.memmap('X', 'float32', 'w+', shape=(11314, 13895))
X[:] = training[:]
y = np.memmap('y', 'float32', 'w+', shape=(11314, 20))
y[:] = output[:]

start_time = time.time()

train(X, y, alpha=0.1, epochs=10000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")


