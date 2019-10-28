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

def output_excel(df):
    a = []
    b = []
    for key, value in df.items():
        a.append(key)
        b.append(value)

    dit = {'Contributor':a, 'Numbers of comments':b}
    writer = pd.ExcelWriter('output.xlsx',engine='xlsxwriter')
    df = pd.DataFrame(dit)
    #columns参数用于指定生成的excel中列的顺序
    df.to_excel(writer, columns=['Contributor','Numbers of comments'], index=False,encoding='utf-8',sheet_name='Sheet')
    writer.save()
    df.to_csv(r'./1.csv',columns=['Contributor','Numbers of comments'],index=False,sep=',')


if __name__ == "__main__":
    excel = read_excel('closed_comment.xlsx') 
    commenter_comments_dict = get_commenter_comments_dict(excel)
    print(commenter_comments_dict)
    output_excel(commenter_comments_dict)