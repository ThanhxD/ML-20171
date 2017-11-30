import nltk, json
import numpy as np
from utils import read_file, wordenize, sigmoid
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def main():
    dictionary = read_file('resources/dictionary')[1:-1].replace("'", "").split(', ')
    classes = read_file('resources/classes').split(', ')
    with open('synapses.json') as json_data:
        synapse0 = json.load(json_data.read())

    file = read_file('input.txt')
    lemmatizer = WordNetLemmatizer()
    stemmer = LancasterStemmer()
    words = wordenize(lemmatizer, stemmer, file)

    l0 = [0] * len(dictionary)
    for word in words:
        if (word in dictionary):
            l0[dictionary.index(word)] += 1
    l1 = sigmoid(np.dot(l0, synapse0))
    indexMax = np.argmax(l1)
    
    print('result: ', classes[indexMax])


if __name__ == '__main__':
    main()