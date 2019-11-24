import numpy as np
import pandas as pd
import string
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt 


def file_open(path):
    print('Start read data from file')
    input_read = pd.read_excel(path,skip_blank_lines=True) 
    df = pd.DataFrame(input_read)
    print('Read data')
    return df

def relation_data_sorted(path):
    res = []
    df = file_open(path)
      
    for index,data in enumerate(df["issue_ID"],start=1):
        if index == df.shape[0]:
            break
        if df["issue_ID"][index]== df["issue_ID"][index-1]:
            if df["commenter"][index-1] != df["commenter"][index]:
                res.append((df["commenter"][index-1],df["commenter"][index]))
    
    return [tuple(sorted(i)) for i in res]
    
def output_excel(df):
    a = []
    b = []
    c = []
    for key,value in df.items():
        a.append(key[0])
        b.append(key[1])
        c.append(value)
    dit = {'commenter1':a, 'commenter2':b, 'weights':c}
    df = pd.DataFrame(dit)
    #columns参数用于指定生成的excel中列的顺序
    df.to_csv(r'relation_data.csv',columns=['commenter1','commenter2','weights'],index=False,sep=',')



if __name__ == "__main__":
    path = "data\closed_comment.xlsx"
    data = relation_data_sorted(path)
    c_Undirected = Counter(data)   #undirected relation  ex: [a,b] == [b,a]
    output_excel(c_Undirected)