from pymongo import MongoClient
from collections import Counter
import pymongo
import datetime

class Init_mongo():
    def __init__(self, url: str="mongodb://10.0.1.19:30123"):
        self.client = MongoClient(url)
        self.vector_collection = self.client.kd.vector
        self.keyword_collection = self.client.kd.kc

    # save keywords for news_list
    # savedata = [{id, keywords, company_list, industry_list, post_date} ...]
    def save_keywords(self, savedata: list):
        db = self.client.kd
        db_collection = db.kc
        db_collection.insert_many(savedata)
        # print()
        
    def save_vector(self, news_id, vector, date):
        db = self.client.kd
        vector_collection = db.vector
        post_date = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f+09:00")
        save_data = {"id": news_id, "vector": vector, 'post_date': post_date}
        vector_collection.insert(save_data)
        print(news_id, 'saved')

    def query_by_date(self, from_date, to_date, collection_object):
        return list(collection_object.find({'post_date':{'$lt':to_date, '$gte':from_date}})) 

    def get_keyword_list(self, news_queried = list, column = str):    
        keyword_list = []    
        for news in news_queried:        
            for source in news[column]:
                keyword_list.append(source)        
        return keyword_list

    def count_keyword_list_by_rank(self, keyword_list):    
        keyword_by_rank = Counter(keyword_list).most_common()
    
        return keyword_by_rank
    
    def get_wordrank_from_keyword_by_rank(self, keyword_by_rank, num):
        rank = []
        keyword_by_rank_cut = keyword_by_rank[:num]
        for word, _ in keyword_by_rank_cut:
            rank.append(word)
        return rank
