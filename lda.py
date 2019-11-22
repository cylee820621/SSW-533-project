# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from glob import glob
import gensim
from gensim import corpora, models
import nltk
from collections import Counter

import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer 

import numpy as np
#import lda
#import lda.datasets

def flatten(seq,container=None):
    if container is None:
        container = []
    for s in seq:
        if hasattr(s,'__iter__'):
            flatten(s,container)
        else:
            container.append(s)
    return container
  
def get_comment_list(excel):
    comment_list = []
    for i in excel.index:
        comment_list.append(excel['comment_content'][i])
    return comment_list


#1. DATA PREPARATION

#Get all files from one project folder
#files = glob('../Cassandra/record/*.txt')
file_name = 'data/closed_comment.xlsx'
df = pd.read_excel(file_name)
comment_list = get_comment_list(df)
print(comment_list)


#load files and combine content to form a corpus
"""
for file_name in files:
  input_file = open(file_name,"r")
  input_data.append(input_file.read()) 
 
  input_file.close()
"""
  #input_data.append(input_file.read())
  #data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
  
#print len(doc_clean)
input_data = comment_list

#2. DATA CLEANING
nltk.download('stopwords')
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

stop.update(('file','look','used','still','20171215','20151113','view','wrote','maybe','sure','something','work','right','may','good','know','need','fatal','00','gnu','found','splash,''result','bytes','using','added','usb','lukejr','seems','way','csmain','20140423','—','o2','last','use','get','need''still','even','20170813','20141124','stdchartraitschar','stdallocatorchar','kb','r','tx','boost','whether','new','boostdetailvariantvoid','0x0','bitcoin','11','make','100','13','void','threadrpcserver','rw','qt','afaik','theuni','000','10','20140417','diff','boostsystemerrorcode','maino','intin','stdallocator','stdchartraits','char','g','c','int','0','1','2','3','4','5','6','7','8','9','const','yes','would', 'like', 'think', 'cassandra', 'mod_mbox', 'apache', 'jira', 'could', 'its', 'one', 'hi', 'thats', 'youre', '2', '3', 'e', '3d', 'see', 'im', 'youre', 'sounds', 'want', 'totally', '1', 'thanks', 'really', 'great', 'also')),
#stop.update(('2', '3', 'e', '3d', 'see', 'im', 'youre', 'sounds', 'want', 'totally', '1', 'thanks', 'really', 'great', 'also'))

print(stop)

def clean(doc):
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    #print "Step: 1 ", stop_free
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    #print "Setp: 2 ", punc_free
    stop_free = ' '.join([i for i in punc_free.split() if i not in stop])
    #print "Step: 3 ", stop_free
    #normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return stop_free

doc_clean = [clean(doc).split() for doc in input_data]
#words_to_count = (word for word in flatten(doc_clean))

#print(Counter(words_to_count).most_common(100))

#3. DOCUMENT TERM MATRIX

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
corpus = [dictionary.doc2bow(text) for text in doc_clean]
    
#print corpus[0]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(corpus, num_topics=5, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=5, num_words=20))
print('Done!')
