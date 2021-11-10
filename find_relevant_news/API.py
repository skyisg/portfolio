#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, Response
from find_relevant_functions.find_relevant import Preprocessing_related_news
import requests
import numpy as np
import os


app = Flask('__name__')

preprocessing_related_news = Preprocessing_related_news()
    
glove_file_path = 'data/glove.6B.100d.txt'

if not os.path.exists(glove_file_path):
    preprocessing_related_news.download_glove_in_data_folder()
    
glove_dict = preprocessing_related_news.get_glove_dict(glove_file_path)


@app.route('/related_news', methods = ['POST'])
def update_related_news():
        
    data = eval(request.data)

    article_num = data['crawlingdata']
    
    vector = data['related_news_vector']
   
    related_article = preprocessing_related_news.get_related_articles(article_num, vector)

    send_data = {'crawlingdata': article_num, 'related_news_id_list': related_article}

    return send_data


@app.route('/keyword_related_news', methods = ['POST'])
def update_keyword_related_news():
        
    data = eval(request.data)
   
    keyword = data['keyword'].replace('\n','')
               
    keyword_lower = keyword.lower()

    vector = preprocessing_related_news.get_vector_for_a_word(glove_dict, keyword_lower)

    default_vector = np.zeros(100,)

    if list(vector) == list(default_vector):
        print(keyword_lower, list(vector))
        return  Response("{'error':'no word in dictionary'}", status=400)
            
    article_num = -1
    
    related_article = preprocessing_related_news.get_related_articles(article_num, vector)

    send_data = {'keyword': keyword, 'related_news_id_list': related_article}

    return send_data
        
        

app.run(host='0.0.0.0', port=50001)

