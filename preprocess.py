import os
import re
import nltk
from nltk.stem.lancaster import LancasterStemmer

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
        ' ': '[\<\,\>\.\?\/\:\;\"\{\[\}\]\-\_\+\=\(\)\|\\\*\&\^\%\$\#\@\!\~\`]'
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

def load_stop_word(stemmer):
    """
    Load stop words file and stem them
    stemmer
    """
    text = read_file('resources/stopwords')
    words = text.split(', ')
    words = [stemmer.stem(word.lower()) for word in words]
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

def wordenize(stemmer, text):
    """
    Split text into words and stem them 
    Notice: no filter stop-word here

    stemmer -- stemmer object
    text    -- text to wordenize
    """
    text = normalize(text)
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word.lower()) for word in words]
    return words

def process_data(stemmer):    
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
            words = wordenize(stemmer, text)

            # write out
            write_file(f'resources/data/train/{folder_name}/{file}', ', '.join(words))

def make_dictionary(stemmer): 
    # load stop word
    stop_words = load_stop_word(stemmer)

    # continue here

def main():
    """ MAIN FUNCTION """
    stemmer = LancasterStemmer()
    process_data(stemmer)
    # make_dictionary(stemmer)    

if __name__ == '__main__':
    main()
