import os, json, nltk
import numpy as np
from utils import read_file, sigmoid, wordenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


dictionary = read_file('resources/dictionary')[1:-1].replace("'", "").split(', ')
classes = read_file('resources/classes').split(', ')
synapses_file_path = 'synapses.json'
tests_path = 'data-raw/test'
def load_modal():
    file = open(synapses_file_path)
    json_data = json.load(file)
    return json_data['synapse0']
synapse0 = load_modal()

def run_test():
    sum = 0
    tick = 0
    folders = os.listdir(tests_path)
    for class_name in folders:
        test_cases = os.listdir(f'{tests_path}/{class_name}')
        for test_case in test_cases:
            sum += 1
            file_path= f'{tests_path}/{class_name}/{test_case}'
            result = test(file_path)
            if class_name == classes[result]:
                tick += 1
    
    print('Result: ', tick/sum)

def test(file_path):
    file = read_file(file_path)
    words = preprocess(file)
    l0 = [0] * len(dictionary)
    for word in words:
        if (word in dictionary):
            l0[dictionary.index(word)] += 1
    l1 = sigmoid(np.dot(l0, synapse0))
    indexMax = np.argmax(l1)
    return indexMax

def preprocess(file):
    lemmatizer = WordNetLemmatizer()
    stemmer = LancasterStemmer()
    text = file
    return wordenize(lemmatizer, stemmer, text)

def main():
    run_test()

if __name__ == '__main__':
    main()
