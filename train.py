import os, sys, json, time, datetime
import numpy as np
from utils import read_file, write_file, sigmoid, sigmoid_derivative

def train(X, y, alpha=1, epochs=10000, classes=[]):

    print ("Training with alpha:%s" % (str(alpha)) )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),len(X[0]), len(classes)) )
    np.random.seed(1)

    last_mean_error = 1

    synapse_0 = 2 * np.random.random((len(X[0]), len(classes))) - 1
        
    layer_0 = X
    for j in iter(range(epochs+1)):

        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        layer_1_error = y - layer_1
        if (j% 1000) == 0:
            error = np.mean(np.abs(layer_1_error))
            if error >= last_mean_error or error < 1e-2:
                print ('break:', error, ', ', last_mean_error )
                break
            print ('delta after ', j, ' iters:', error)
            last_mean_error = error

        layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)
        
        synapse_0_weight_update = layer_0.T.dot(layer_1_delta)
        
        synapse_0 += alpha * synapse_0_weight_update

    now = datetime.datetime.now()
    synapse = {'synapse0': synapse_0.tolist()}
    with open('synapses.json', 'w') as outfile:
        json.dump(synapse, outfile, indent=4)
    print('Train done.')

def main():
    train_path = 'resources/data/train'
    training = []
    output = []
    # load resources
    classes = read_file('resources/classes').split(', ')
    dictionary = read_file('resources/dictionary')[1:-1].replace("'", "").split(', ')
    train_folders = os.listdir(train_path)
    for folder in train_folders:
        train_files = os.listdir(f'{train_path}/{folder}')
        for file in train_files:
            print("Processing...", file)
            train_case = [0] * len(dictionary)
            file_path = f'{train_path}/{folder}/{file}'
            text = read_file(file_path)[1:-1].replace("'", "").split(', ')
            for item in text:
                if item == '':
                    continue
                try:
                    (word, value) = item.split(': ')
                    train_case[dictionary.index(word)] = int(value)
                except:
                    print("item: ", item)
                    print("words: ", word)
                    print("value: ", value)
                    sys.exit(0)
            training.append(train_case)
            output_case = [0] * len(train_folders)
            output_case[train_folders.index(folder)] = 1
            output.append(output_case)

    print("training: ", len(training), ' x ', len(training[0]))
    print("ouput: ", len(output), ' x ', len(output[0]))
    # write_file('resources/training', str(training))
    # write_file('resources/output', str(output))

    X = np.array(training)
    y = np.array(output)

    start_time = time.time()

    train(X, y, alpha=0.1, epochs=10000, classes=classes)

    elapsed_time = time.time() - start_time
    print ("processing time:", elapsed_time, "seconds")

if __name__ == '__main__':
    main()