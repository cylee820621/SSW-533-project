import nltk
import pandas as pd
from pandas import ExcelFile
import matplotlib.pyplot as plt
from textblob import TextBlob

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


if __name__ == "__main__":
    excel = read_excel('closed_comment.xlsx') 
    commenter_comments_dict = get_commenter_comments_dict(excel)
    print(commenter_comments_dict)
    output_excel(commenter_comments_dict)
    sentiment_dict = Contributor_total_sentiment(excel)
    print(sentiment_dict)