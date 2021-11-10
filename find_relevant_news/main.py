#!/usr/bin/env python
# coding: utf-8

from find_relevant_functions.find_relevant import Preprocessing_related_news, Update_related_news
import time
import os

pr = Preprocessing_related_news()
glove_dict_path = 'data/glove.6B.100d.txt'
if not os.path.exists(glove_dict_path):
    pr.download_glove_in_data_folder()
glove_dict = pr.get_glove_dict(glove_dict_path)

duration = 300
saving_duration_for_redis = 1

while True:
    
    update_related_news = Update_related_news()
    
    start, end = update_related_news.get_index_for_updating_related_news()
    
    print(start, end)
    
    update_related_news.update_db_and_redis_by_range(start, end, glove_dict, saving_duration_for_redis)
    
    time.sleep(duration)