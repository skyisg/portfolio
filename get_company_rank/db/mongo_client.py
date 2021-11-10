from pymongo import MongoClient
from collections import Counter
import pymongo
import datetime

class Init_mongo():
    
    def __init__(self, url: str="mongodb://10.0.1.19:30123"):
        self.client = MongoClient(url)
        self.keyword_collection =  self.client.kd.kc

    # save keywords for news_list
    # savedata = [{id, keywords, company_list, industry_list, post_date} ...]
    def save_keywords(self, savedata: list):
        db = self.client.kd
        db_collection = db.kc
        db_collection.insert_many(savedata)
        # print()
       
    def query_by_date(self, collection, from_date, to_date, time_column='post_date'):
        return list(collection.find({time_column:{'$lt':to_date, '$gte':from_date}})) 
