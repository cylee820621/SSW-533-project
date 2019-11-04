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

def relation_data(path):
    res = []
    df = file_open(path)
      
    for index,data in enumerate(df["issue_ID"],start=1):
        if index == df.shape[0]:
            break
        if df["issue_ID"][index]== df["issue_ID"][index-1]:
            res.append((df["commenter"][index-1],df["commenter"][index]))
    return res

if __name__ == "__main__":
    path = "data\closed_comment_test.xlsx"
    data = relation_data(path)
    c_directed = Counter(data)    #directed relation  ex: [a,b] != [b,a]
    c_Undirected = Counter(sorted(data))   #undirected relation  ex: [a,b] == [b,a]