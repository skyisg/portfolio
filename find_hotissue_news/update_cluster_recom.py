#!/usr/bin/env python
# coding: utf-8

# public 라이브러리 import
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import time

# private 라이브러리 import 
from db.mongo_client import *
from rank.issue_rank import *
from recommendations.social_issue import *

# class가져오기
mongo_c = Init_mongo()
cr = Cluster_recom()

# vector collection 변수에 vector가 저장된 콜렉션 저장
vector_collection = mongo_c.vector_collection

# update간격 지정(seconds)
update_duration = 60

while True:

    # datetime 객체 반환
    oldest_standard = 7
    older_standard = 1
    standard = 0

    oldest_time, older_time, standard_time = cr.get_datetime_objects(oldest_standard, older_standard, standard)

    # oldest_time과 standard 사이에 생성된 뉴스기사의 벡터 추출
    newsdata_oldest = mongo_c.query_by_date(oldest_time, standard_time, vector_collection)

    # 해당시간 사이의 아이디와 벡터 각각 리스트로 저장
    id_list = [newsdata['id'] for newsdata in newsdata_oldest]
    vector_list =  [newsdata['vector'] for newsdata in newsdata_oldest]
    print(len(id_list))

    # 계층적 군집화
    # method 종류: average, complete, single
    # criterion 종류: inconsistent, distance, maxclust, monocrit, maxclust_monocrit
    method = 'average'
    criterion = 'inconsistent'
    fcluster_threshold = 0.9

    cluster_index = cr.hierarchy_clustering(vector_list, fcluster_threshold, method, criterion)

    # 클러스터id를 key값으로 기사아이디 list를 value로 가지는 dictionary 생성
    cluster_dict = cr.get_cluster_dict(id_list, cluster_index)

    # 클러스터별 점수1 구하기
    cluster_score1_dict = cr.get_score1_dict(cluster_index)

    # 클러스터별 점수2 구하기
    cluster_score2_dict = cr.get_score2_dict(cluster_dict, newsdata_oldest, older_time, standard_time)

    # 최종 점수 구하기
    score1_weight=0.63
    score2_weight=0.37
    final_score_dict = cr.get_final_score_dict(cluster_score1_dict, cluster_score2_dict, score1_weight, score2_weight)

    # n개 클러스터 기사 아이디 dictionary로 추출
    num_of_top_cluster = 10
    min_cluster = 3
    top_cluster_dict = cr.get_topN_cluster_dict(cluster_dict, final_score_dict, num_of_top_cluster, min_cluster)

    print(top_cluster_dict)
    
    # 결과 출력 함수
#     cr.show_ko_title(top_cluster_dict)
#     end_time = time.time()

    # api를 이용하여 백엔드 db에 저장하는 함수
    
    url = ''    
    
    hot_news_list = []
    
    for _, news_list in top_cluster_dict.items():
        hot_news_list.append(news_list)
    
    lengths = [len(news_list) for news_list in hot_news_list]
    
    max_lengths = max(lengths)
    
    for news_id in hot_news_list:
        while len(news_id) < max_lengths:
            news_id += [0]
            
    send_data = {"hot_news_list": hot_news_list}
    
    print(send_data)
   
    
    response = requests.post(url, json = send_data)
    print(response.status_code)
    
    # update_duration만큼 정지
    print()
    print('--updated--')
    time.sleep(update_duration)

send_data = {"hot_news_list": hot_news_list}

print(send_data)
   

response = requests.post(url, json = send_data)
print(response.status_code)

# update_duration만큼 정지
print()
print('--updated--')
    time.sleep(update_duration)

