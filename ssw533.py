import nltk
import pandas as pd
from pandas import ExcelFile
import matplotlib.pyplot as plt
from textblob import TextBlob
from scipy.spatial import distance
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import wordnet
import string
from nltk.corpus import stopwords
'''
def Contributor_total_sentiment(ex):
    contributor_sentiment_dict = dict()
    for i in ex.index:
        text = TextBlob(ex['comment_content'][i])
        if ex['commenter'][i] not in contributor_sentiment_dict:
            contributor_sentiment_dict[ex['commenter'][i]] = [text.sentiment.polarity]
        else:
            contributor_sentiment_dict[ex['commenter'][i]].append(text.sentiment.polarity)
    return contributor_sentiment_dict

def sentiment(text):
    process_text = TextBlob(text)
    return process_text.sentiment
'''

def get_commenter_comments_dict(ex):
    """
    df用法= df[第幾欄][第幾列]
    輸入excel
    輸出dict{'contributor1': #comments } }
    """
    dict_contributor_comments = dict()
    for i in ex.index:
        if ex['commenter'][i] not in dict_contributor_comments:
            dict_contributor_comments[ex['commenter'][i]] = 1
        else:
            dict_contributor_comments[ex['commenter'][i]] += 1
    
    return dict_contributor_comments # key= contributor, value = # of comments

def read_excel(file_name):
    df = pd.read_excel(file_name)
    return df

def output_excel(df):
    a = []
    b = []
    for key, value in df.items():
        a.append(key)
        b.append(value)
    dit = {'Contributor':a, 'Numbers of comments':b}
    writer = pd.ExcelWriter('output_Contributor_comments.xlsx',engine='xlsxwriter')
    df = pd.DataFrame(dit)
    #columns参数用于指定生成的excel中列的顺序
    df.to_excel(writer, columns=['Contributor','Numbers of comments'], index=False,encoding='utf-8',sheet_name='Sheet')
    writer.save()
    df.to_csv(r'output_Contributor_comments.csv',columns=['Contributor','Numbers of comments'],index=False,sep=',')

def commentor_words(ex):
    commentor_words={}
    for i in ex.index:
        if not ex['commenter'][i] in commentor_words:
            token_sentence = tokenize(ex['comment_content'][i])
            comment_word = len(token_sentence)
            commentor_words[ex['commenter'][i]] = comment_word
        else:
            commentor_words[ex['commenter'][i]] += comment_word
    return commentor_words

def total_posts(ex):
    total_post = 0
    for i in ex.index:
        total_post += 1
    return total_post

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

def creat_actvity_words_dict(words_dict, comments_dict, total_posts):
    activity_words_dict= dict()
    cd = comments_dict
    wd = words_dict
    for commenter, posts in cd.items():
        activity_words_dict[commenter] = [posts/total_posts]
    for commenter, posts in wd.items():
        activity_words_dict[commenter].append(posts)
    return activity_words_dict

def creat_actvity_words_dict_excel(awd):
    C = []
    A = []
    W = []
    for commenter, value in awd.items():
        C.append(commenter)
        A.append(value[0])
        W.append(value[1])

    dit = {'Contributor':C, 'Activity':A, 'Words':W}
    writer = pd.ExcelWriter('Activity.xlsx',engine='xlsxwriter')
    df = pd.DataFrame(dit)
    #columns参数用于指定生成的excel中列的顺序
    df.to_excel(writer, columns=['Contributor','Activity','Words'], index=True,encoding='utf-8',sheet_name='Sheet')
    writer.save()
    #df.to_csv(r'output_Contributor_comments.csv',columns=['Contributor','Numbers of comments'],index=False,sep=',')

if __name__ == "__main__":
    excel = read_excel('data/closed_comment.xlsx') 
    #output_excel(commenter_comments_dict)
    #sentiment_dict = Contributor_total_sentiment(excel)
    total_post = total_posts(excel)
    commentor_words_dict = commentor_words(excel)
    commenter_comments_dict = get_commenter_comments_dict(excel)
    print(commenter_comments_dict)
    awd = creat_actvity_words_dict(commentor_words_dict,commenter_comments_dict, total_post)
    creat_actvity_words_dict_excel(awd)