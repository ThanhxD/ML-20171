import re

def format_file(file):
    text = file
    match = {
        ' currency ': '(\€|\¥|\£|\$)\d+([\.\,]\d+)*',
        ' email ': '[^\r\n @]+@[^ ]+',
        ' url ': '(((http|https):*\/\/[^\s]*)|((www)\.[^\s]*)|([^\s]*(\.com|\.co\.uk|\.net)[^\s]*))',
        ' number ': '\d+[\.\,]*\d*',
        '': '[<,>.?\/:;"\'{[}\]-_\+=()]'
    }
    for key in match:
        text = re.sub(match[key], key, text)

    return text


def make_bow(file):
    return file.split(' ')

def main():
    file_input = open('data-raw/train/alt.atheism/51126', 'r').read()
    formated_file = format_file(file_input)
    words = make_bow(formated_file)
    file_output = open('resources/data/train/alt.atheism/51126', 'w')
    file_output.write(', '.join(words))

if __name__ == '__main__':
    main()
