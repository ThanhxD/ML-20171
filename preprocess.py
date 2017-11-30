import os, math, nltk, sys
import numpy as np
from utils import read_file, write_file, wordenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def process_file(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = LancasterStemmer()
    return_text = wordenize(lemmatizer, stemmer, text)
    return return_text

def counting(words):
    freq = {}
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq

def main():
    """ MAIN """
    train_path = 'data-raw/train'
    processed_path = 'resources/data/train'
    folders = os.listdir(train_path)
    bag_of_words = []
    
    # preprocess file
    for folder in folders:
        files_path = f'{train_path}/{folder}'
        files = os.listdir(files_path)
        print('Processing...', folder)
        for file in files:
            file_path = f'{files_path}/{file}'
            text = read_file(file_path)
            words = process_file(text)
            bag_of_words.append({'path': f'{folder}/{file}', 'words': words})

    # for row in bag_of_words:
    #     path = row['path']
    #     write_file(f'resources/data/temp/{path}', str(row['words']))
    # # make dictionary and filter file by this dict
    # list_folders = os.listdir('resources/data/temp')
    # for folder in list_folders:
    #     list_files = os.listdir(f'resources/data/temp/{folder}')
    #     for file in list_files:
    #         text = read_file(f'resources/data/temp/{folder}/{file}').replace("'", "")
    #         text = text[1:-1].split(', ')
    #         bag_of_words.append({'path': f'{folder}/{file}', 'words': text})
    freq_idf = {}
    freq_tf = []
    dictionary = []
    sum_docs = len(bag_of_words)
    for row in bag_of_words:
        count = counting(row['words'])
        freq_tf.append(count)
        words = list(set(row['words']))
        for word in words:
            if word in freq_idf:
                freq_idf[word] += 1
            else:
                freq_idf[word] = 1
    for row in bag_of_words:
        row_index = bag_of_words.index(row)
        print("processing...", row_index)
        words = list(set(row['words']))
        current_freq_tf = freq_tf[row_index]
        return_row = {}
        for word in words:
            count = current_freq_tf[word]
            if count < 3 or count > 3000:
                continue
            tf_idf = (count/len(row['words'])) * math.log(sum_docs/freq_idf[word])
            if tf_idf < 0.002:
                continue
            return_row[word] = count
            dictionary.append(word)
        file_path = row['path']
        write_file(f'{processed_path}/{file_path}', str(return_row))
    dictionary = list(set(dictionary))
    write_file('resources/dictionary', str(dictionary))
    print('Preprocess done.')

if __name__ == '__main__':
    main()
