#!/usr/bin/env python
# coding: utf-8

import sys, os
current_path = os.getcwd()
upper_path = '/'.join(current_path.split('/')[:-1])
sys.path.append(upper_path)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import requests
from rank.issue_rank import Issue_rank

class Cluster_recom(Issue_rank):
    
    def get_one_news_data(self, index, url):
        response = requests.get(url + str(index) + '/')
        if not response.status_code == 200:
            return
        return response.json()
    
    def get_datetime_objects(self, oldest_standard, older_standard, standard):
        oldest_time = self.get_datetime_object_by_days_passed(oldest_standard)
        older_time = self.get_datetime_object_by_days_passed(older_standard)
        standard_time = self.get_datetime_object_by_days_passed(standard)
        return oldest_time, older_time, standard_time
    
    # method 종류: average, complete, single
    # criterion 종류: inconsistent, distance, maxclust, monocrit, maxclust_monocrit
    def hierarchy_clustering(self, vector_list, fcluster_threshold, method='complete', criterion='distance'):
        merging = linkage(vector_list, method=method)
        cluster_index = fcluster(merging, fcluster_threshold, criterion=criterion)
        return cluster_index

    def get_cluster_dict(self, id_list, cluster_index):
        unique = np.unique(cluster_index)
        cluster_dict = dict()
        for cluster_num in unique:
            cluster_dict[cluster_num] = []
        for index, cluster_num in zip(id_list, cluster_index):
            cluster_dict[cluster_num] += [index]
        return cluster_dict
    
    def get_ko_title(self, news_id):
        news_data = self.get_one_news_data(news_id)
        ko_news_title = news_data['ko_title']
        return (news_id, ko_news_title)   
    
    def get_score1_dict(self, cluster_index):
        overall_counts = len(cluster_index)
        unique, counts = np.unique(cluster_index, return_counts=True)
        cluster_score1_dict = dict()
        for cluster_num, count in zip(unique, counts):
            cluster_score1_dict[cluster_num] = count/overall_counts
        return cluster_score1_dict
    
    def calc_score2(self, cluster_num, cluster_dict, news_data, from_date, to_date):
        recent_news_list = self.time_filter_news_list(news_data, from_date, to_date)
        recent_news_id_list = [news['id'] for news in recent_news_list]
        cluster_id_list = cluster_dict[cluster_num]
        cluster_id_length = len(cluster_id_list)
        count = 0    
        for news_id in recent_news_id_list:
            if news_id in cluster_id_list:
                count += 1
        score2 = count / cluster_id_length
        return score2
    
    def get_score2_dict(self, cluster_dict, news_data, from_date, to_date):
        cluster_score2_dict = dict()
        keys = cluster_dict.keys()
        for key in keys:
            score2 = self.calc_score2(key, cluster_dict, news_data, from_date, to_date)
            cluster_score2_dict[key] = score2
        return cluster_score2_dict
    
    def get_final_score_dict(self,cluster_score1_dict, cluster_score2_dict, score1_weight=0.63, score2_weight=0.37):
        final_score_dict = dict()
        keys = cluster_score1_dict.keys()
        for key in keys:
            score1 = cluster_score1_dict[key] * score1_weight
            score2 = cluster_score2_dict[key] * score2_weight
            final_score = score1 + score2
            final_score_dict[key] = final_score
        return final_score_dict
    
    def get_topN_cluster_dict(self, cluster_dict, final_score_dict, num_of_cluster, min_cluster):
        rank = sorted(final_score_dict.items(), key=lambda x: x[1], reverse=True)
        rank_cut = [news_id for news_id, _ in rank]
        top_cluster_dict = dict()        
        for top_cluster in rank_cut:
            if len(cluster_dict[top_cluster]) < min_cluster:
                continue
            top_cluster_dict[top_cluster] = cluster_dict[top_cluster]
            if len(top_cluster_dict.keys()) >= num_of_cluster:
                break
        return top_cluster_dict
    
    def show_ko_title(self, top_cluster_dict):
        for cluster_num, news_id_list in top_cluster_dict.items():
            print(cluster_num)
            print('갯수:', len(news_id_list))
            for news_id in news_id_list:
                ko_title = self.get_ko_title(news_id)
                print(ko_title)
                