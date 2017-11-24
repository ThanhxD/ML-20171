import os
import re
import math
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

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

def wordenize(lemmatizer, stemmer, text, stop_words):
    """
    Split text into words and stem them 
    Notice: no filter stop-word here

    stemmer -- stemmer object
    text    -- text to wordenize
    """
    text = normalize(text)
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in words if word not in stop_words]
    return words

def process_data(lemmatizer, stemmer, stop_words):    
    # read train folder
    train_folders = os.listdir('data-raw/train')
    for folder_name in train_folders:
        
        # for each folder: list file
        train_files = os.listdir(f'data-raw/train/{folder_name}')
        print(f'Working: {folder_name}')

        for file in train_files:
            
            # for each file: read file
            text = read_file(f'data-raw/train/{folder_name}/{file}')

            # then wordenize it to array of words
            words = wordenize(lemmatizer, stemmer, text, stop_words)

            # write out
            write_file(f'resources/data/train/{folder_name}/{file}', ', '.join(words))

def tf_idf(word, current_doc, documents):
    current_count = 0
    current_len = len(current_doc)
    for w in current_doc:
        if word == w:
            current_count += 1
    tf = current_count / current_len

    all_count = 0
    all_len = len(documents)
    for d in documents:
        if word in d:
            all_count += 1
    if all_count == 0:
        all_count = 1
    idf = math.log(all_len / all_count, 10)

    return tf * idf

def make_dictionary():
    dictionary = []
    # Read words from file
    train_folders = os.listdir('resources/data/train')
    words = []
    for folder_name in train_folders:
        # for each folder: list file
        train_files = os.listdir(f'resources/data/train/{folder_name}')
        print(f'Working: {folder_name}')
        line = []

        for file in train_files:
            
            # for each file: read file
            text = read_file(f'resources/data/train/{folder_name}/{file}')

            line += text.split(', ');
        
        words.append(line);

    for i in range(len(words)):
        line = list(set(words[i]))
        freq = {}
        for word in line:
            freq[word] = 0
        for j in range(len(words[i])):
            freq[words[i][j]] += 1
        line = [w for w in line if (freq[w] > 4 and freq[w] < 4000)]
        print ('=====================> ', i, ' ============ ', len(line))
        j = 0
        new_line = []
        for word in line:
            # print (j, ' - ', word)
            value = tf_idf(word, words[i], words)
            if value >= 2e-05:
                new_line.append(word)
            j += 1
        dictionary += new_line

    # write out
    write_file('resources/dictionary', ', '.join(dictionary))

    print('dictionary: ', len(dictionary))
    print('train_folder: ', len(train_folders))
    for i in range(len(train_folders)):
        print('i: ', i)
        folder_name = train_folders[i]
        train_files = os.listdir(f'resources/data/train/{folder_name}')
        for file in train_files:
            text = read_file(f'resources/data/train/{folder_name}/{file}')
            arr_text = text.split(', ');
            elements_in_both_lists = [w for w in arr_text if w in dictionary]
            print(f'Write: {folder_name}/{file}')
            write_file(f'resources/data/processed/{folder_name}/{file}', ', '.join(elements_in_both_lists))

def main():
    # """ MAIN FUNCTION """
    
    stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = load_stop_word(lemmatizer, stemmer)
    process_data(lemmatizer, stemmer, stop_words)
    
    # make_dictionary(stemmer)
    make_dictionary()  

if __name__ == '__main__':
    main()
