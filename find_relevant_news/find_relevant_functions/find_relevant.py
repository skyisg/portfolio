import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import requests 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json 
import urllib
import redis
import datetime
import requests
from tensorflow import keras
import os


# related_news와 관련하여 data를 전처리하는 class
class Preprocessing_related_news():
    # 특정 번호의 크롤링된 기사를 dictionary형식으로 가져오는 함수
    def get_article_dictionary(self, article_num, crawling_data_url):
        individual_crawling_data = crawling_data_url _ article_num + '/'
        # 추후 api 수정 예정
        response = requests.get(individual_crawling_data)
        if str(response) == '<Response [200]>':
            article_dictionary = response.json()
            return article_dictionary
        else:
            return

    # artitle_dictionary에서 text만 extract하는 함수
    def extract_text_from_article(self, article = dict):
        text = article['eng_title'] + ' ' + article['eng_sub_title'] + ' ' + article['eng_content']
        return text

     # text를 입력하면 상위 10개 키워드와 그 비율을 return하는 함수
    def get_keywords(self, string, stop_words):
        text = sent_tokenize(string)
        # 정제와 단어 토큰화
        vocab = {} # 파이썬의 dictionary 자료형
        sentences = []
        for raw_sentence in text:
            sentence = word_tokenize(raw_sentence) # 단어 토큰화
            result = []
            for word in sentence: 
                word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
                if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거
                    if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거
                        result.append(word)
                        if word not in vocab:
                            vocab[word] = 0 
                        vocab[word] += 1
            sentences.append(result)
        words = sum(sentences, [])
        vocab = Counter(words) 
        vocab_list = list(vocab.items()) # 튜플을 갖고 있는 리스트로 변환
        vocab_list.sort(key=lambda x: x[1], reverse=True)
        top_10  = vocab_list[:10]
        sum_num = 0 
        for word, num in top_10:
            sum_num += num
        top_10_prop = []
        for word, num in top_10:
            top_10_prop.append((word, num/sum_num))
        return top_10_prop
    
    # glove data file 다운받는 코드
    def download_glove_in_data_folder(self):
        if not os.path.exists('data/glove.6B.100d.txt'):
            url = 'http://nlp.stanford.edu/data/glove.6B.zip'
            data_folder_path = os.getcwd() + '/data'
            keras.utils.get_file('glove.zip', url, cache_subdir= data_folder_path, extract = True)
            print('Download Completed in path:data/glove.6B.100d.txt')

        else:
            print('file already exists in path:data/glove.6B.100d.txt')

    # glove_dictionary 만드는 코드
    def get_glove_dict(self, glove_file_path='data/glove.6B.100d.txt'):
        with open(glove_file_path, 'r', encoding='latin1') as f:
            glove = f.readlines()
        glove_dict = {}
        for vector in glove:
            vector_lst = vector.split(' ')
            glove_dict[vector_lst[0]] = list(map(float, vector_lst[1:]))
        return glove_dict

    # glove_dict를 이용해서 한 단어를 벡터화    
    def get_vector_for_a_word(self, glove_dict, word):        
        default_vector = np.zeros(100,)        
        word_vector =  np.array(glove_dict.get(word, default_vector))        
        return word_vector   
    
    # glove_dict를 이용해서 키워드들을 벡터화
    def get_vector_for_keywords(self, glove_dict, keywords):
        vector = [0 for _ in range(100)]
        for word, prop in keywords:
            word_vector =  self.get_vector_for_a_word(glove_dict, word)
            word_vector_by_weight = word_vector * prop
            vector += word_vector_by_weight
        return vector    

    # vector를 입력하면 redis에 저장된 벡터들을 기반으로 관련 기사를 return 하는 함수
    def get_related_articles(self, article_num, vector):
        # redis 호출하기
        rd = redis.StrictRedis(host='localhost', port=6379, db="1", charset='utf-8', decode_responses=True)
        # redis에서 모든 key, value 값을 가져와서 입력되는 related_news_vector와 cos score 구하기
        cosine_scores = []
        for key in rd.keys():
            if str(key) == str(article_num):
                continue
            vector_in_redis = rd.lrange(key, 0, -1)
            score = float(cosine_similarity([vector], [vector_in_redis])[0][0])
            cosine_scores.append((score, key))
        rank = sorted(cosine_scores, key = lambda x: x[0], reverse = True)
        # rank 중에 상위 20개 추출
        article_rank = []
        for score, article_num in rank:
            if score != 1:
                article_rank.append(article_num)
            if len(article_rank) == 20:
                break
        return article_rank

    
# related_news와 관련하여 db와 redis를 갱신하는 클라스
class Update_related_news(Preprocessing_related_news):

    # db와 redis에 업데이트해야 하는 범위를 return하는 함수
    def get_index_for_updating_related_news(self, related_news_url='', crawling_data_url=''):
        # check index
        latest_related_news_index = int(requests.get(related_news_url).json()['results'][0]['crawlingdata'])
        latest_crawling_data_index = int(requests.get(crawling_data_url).json()['results'][0]['id'])
        return latest_related_news_index, latest_crawling_data_index

    # article_num을 key로 vector를 value로 redis에 저장하는 함수
    def save_in_redis(self, article_num = int, vector = list, saving_duration = 1):
        rd = redis.StrictRedis(host='localhost', port=6379, db="1", charset='utf-8', decode_responses=True)
        if rd.keys(article_num) == []:
            rd.rpush(article_num, *vector)
            rd.expire(article_num, datetime.timedelta(days = saving_duration))
            print(article_num, '번이 redis에 저장되었습니다.')
        else:
            print('redis saving error:', article_num, '번은 이미 redis에 저장되어있습니다')

    # article_num, vector, related_articles을 DB에 저장하는 함수
    def save_in_db(self, article_num, vector, related_articles, related_news_url=''):
        send_data = {'crawlingdata': article_num, 'related_news_vector': vector, 'related_news_id_list': related_articles}
        response = requests.post(related_news_url, data = send_data)
        if str(response) == '<Response [201]>':
            print(article_num, '번 DB에 저장되었습니다.')
        else:
            print(article_num, '번 post method 에러 -> put method 시도')
            response_again = requests.put(related_news_url + str(article_num) + '/', data = send_data)
            if str(response_again) == '<Response [200]>':
                print(article_num, '번 DB에 갱신되었습니다')
            else:
                print(article_num, '번 put method 에러 -> 원인을 파악하세요')

    # 범위에 따라 db와 redis를 갱신하는 함수
    def update_db_and_redis_by_range(self, start = int, end = int, glove_dict = dict, saving_duration = 1):
        for article_num in range(start, end + 1):
            article_dictionary = self.get_article_dictionary(article_num)
            if not article_dictionary:
                print('No data for {} in API' .format(article(num)))
                continue
            article_text = self.extract_text_from_article(article_dictionary)
            keywords = self.get_keywords(article_text)
            vector = self.get_vector_for_keywords(glove_dict, keywords)
            related_articles = self.get_related_articles(article_num, vector)
            self.save_in_db(article_num, vector, related_articles)
            self.save_in_redis(article_num, vector)        
