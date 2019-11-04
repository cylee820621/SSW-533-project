from scipy.spatial import distance
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import wordnet
import string
from nltk.corpus import stopwords
from collections import Counter
stop_words = stopwords.words('english')

def tokenize(text):
    text = str(text)
    str1 = text.lower().strip()         # segments the lowercased string into tokens
    str2 = str1.translate(str.maketrans('', '', string.punctuation))     #remove punctuation  
    match = nltk.word_tokenize(str2)                           # word_tokenize
    filtered_sentence = [w for w in match if not w in stop_words] 
    # tagged_token is a list of (word, pos_tag)
    # define a mapping between wordnet tags and POS tags as a function
    #token_count = nltk.FreqDist(filtered_sentence)
    return filtered_sentence

def file_open(path):
    print('Start read data from file')
    input_read = pd.read_excel(path,skip_blank_lines=True) 
    df = pd.DataFrame(input_read)
    print('Read data')
    return df

if __name__ == "__main__":

    path = "data\closed_comment.xlsx"
    comments_file = file_open(path)
    i=0
    token = []
    for text in comments_file["comment_content"]:
        token.extend(tokenize(text))
        i=i+1
        print('processing %d out of %d items...'%(i,len(comments_file)),'\r',end='')        
    
    c = Counter(token)
    a =[]
    b =[]
    for key,value in c.items():
        a.append(key)
        b.append(value)

    dit = {'words':a, 'frequency':b}
    df = pd.DataFrame(dit)
    df.to_csv(r'frequency.csv',columns=['words','frequency'],index=False,sep=',')        