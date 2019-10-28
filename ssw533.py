import nltk
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt

def get_commenter_comments_dict(ex):
    """
    df用法= df[第幾欄][第幾列]
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


if __name__ == "__main__":
    excel = read_excel('closed_comment.xlsx') 
    commenter_comments_dict = get_commenter_comments_dict(excel)
    print(commenter_comments_dict)