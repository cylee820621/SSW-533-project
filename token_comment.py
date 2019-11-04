from scipy.spatial import distance
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import wordnet
import string
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def tokenize(text):
    token_count = None
    text = str(text)
    str1 = text.lower().strip()
    str2 = str1.translate(str.maketrans('', '', string.punctuation))     # segments the lowercased string into tokens
    match = nltk.word_tokenize(str2)                           # word_tokenize
    # tagged_token is a list of (word, pos_tag)
    # define a mapping between wordnet tags and POS tags as a function
    token_count = nltk.FreqDist(match)
    return token_count

def file_open(path):
    print('Start read data from file')
    input_read = pd.read_csv(path,skip_blank_lines=True) 
    df = pd.DataFrame(input_read)
    print('Read data')
    return df

if __name__ == "__main__":

    path = "closed_comment.csv"
    comments_file = file_open(path)
    new = pd.DataFrame(comments_file,columns = ["issue_ID","commenter","time","comment_content","tokens"])   
    for index, text in enumerate(new["comment_content"]):
        token = tokenize(text)
        new["tokens"][index] =  list(token.items())
        if new["issue_ID"][index] == "NaN":
            break
        print('processing %d out of %d items...'%(index,len(new)),'\r',end='')        

    new.to_csv(r'output_comments_token.csv',index=False,sep=',')        