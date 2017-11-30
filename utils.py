import re, os, math, nltk
import numpy as np

""" Collection of util functions """

def normalize(file):
    """
    Normalize text: replace special paterns by word
    file
    """
    text = file
    match = {
        ' currency ': '(\€|\¥|\£|\$)\d+([\.\,]\d+)*',
        ' email ': '[^\r\n @]+@[^ ]+',
        ' url ': '(((http|https):*\/\/[^\s]*)|((www)\.[^\s]*)|([^\s]*(\.com|\.co\.uk|\.net)[^\s]*))',
        ' number ': '\d+[\.\,]*\d*',
        '*': '(\'s|\'ll|n\'t|\'re|\'d|\'ve)',
        '#': '[\r\n]',
        ' ': '[^a-zA-Z]'
    }
    for key in match:
        text = re.sub(match[key], key, text)
    return text

def ensure_path(path):
    """
    Ensure path exists for write file
    path
    """
    subs = path.split('/')
    full_fill = '.'
    for name in subs[:-1]:
        full_fill += f'/{name}'
        if not os.path.exists(full_fill):
            os.makedirs(full_fill)
    full_fill += f'/{subs[-1]}'
    return full_fill

def load_stop_word(lemmatizer, stemmer):
    """
    Load stop words file and stem them
    stemmer
    """
    text = read_file('resources/stopwords')
    words = text.split(', ')
    words = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in words]
    return words

def read_file(file_path):
    """
    Read file from disk
    file_path
    """
    file = open(file_path, 'r')
    text = file.read()
    file.close()
    return text

def write_file(file_path, data):
    """
    Write file to disk
    file_path
    data
    """
    file = open(ensure_path(file_path), 'w')
    file.write(data)
    file.close()

def wordenize(lemmatizer, stemmer, text):
    """
    Split text into words and stem them 
    Notice: no filter stop-word here

    stemmer -- stemmer object
    text    -- text to wordenize
    """
    text = normalize(text)
    stop_words = load_stop_word(lemmatizer, stemmer)
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in words if word not in stop_words]
    return words

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_derivative(output):
    return output*(1-output)