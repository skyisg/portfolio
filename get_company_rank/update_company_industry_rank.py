#!/usr/bin/env python
# coding: utf-8
from db.mongo_client import *
from rank.issue_rank import *

mongo_c = Init_mongo()
keyword_collection = mongo_c.keyword_collection
ir = Issue_rank()

# query와 filter를 위한 datetime객체 반환
now = ir.get_datetime_object_by_days_passed(0)
yesterday = ir.get_datetime_object_by_days_passed(1)
two_days_ago = ir.get_datetime_object_by_days_passed(2)
week_ago = ir.get_datetime_object_by_days_passed(7)

# 몽고db에서 현재부터 일주일 전까지 쿼리
news_list_week = mongo_c.query_by_date(keyword_collection, week_ago, now, 'post_date')

# 일주일전까지 뉴스 리스트에서 오늘 뉴스 리스트와 어제 뉴스 리스트를 추출
news_list_yesterday = ir.time_filter_news_list(news_list_week, two_days_ago, yesterday, 'post_date')
news_list_today = ir.time_filter_news_list(news_list_week, yesterday, now, 'post_date')

# 오늘, 어제, 주간 기업 등장횟수 카운트

company_list_week = ir.get_company_or_industry_list(news_list_week, 'company_list')
company_dict_week = ir.count_company_or_industry(company_list_week)

company_list_yesterday = ir.get_company_or_industry_list(news_list_yesterday, 'company_list')
company_dict_yesterday = ir.count_company_or_industry(company_list_yesterday)

company_list_today = ir.get_company_or_industry_list(news_list_today, 'company_list')
company_dict_today = ir.count_company_or_industry(company_list_today)

# 오늘 어제, 주간 산업 등장횟수 카운트

industry_list_week = ir.get_company_or_industry_list(news_list_week, 'industry_list')
industry_dict_week = ir.count_company_or_industry(industry_list_week)

industry_list_yesterday = ir.get_company_or_industry_list(news_list_yesterday, 'industry_list')
industry_dict_yesterday = ir.count_company_or_industry(industry_list_yesterday)

industry_list_today = ir.get_company_or_industry_list(news_list_today, 'industry_list')
industry_dict_today = ir.count_company_or_industry(industry_list_today)

# 기업과 산업 점수 계산

company_scores = ir.scoring_source(company_dict_today, company_dict_yesterday, company_dict_week)
industry_scores = ir.scoring_source(industry_dict_today, industry_dict_yesterday, industry_dict_week)

# 로그 출력

print('기업 순위')
print(ir.print_scores(company_scores))
print()
print('산업 순위')
print(ir.print_scores(industry_scores))

# api에 저장

save_count = 10
redis_object = redis.StrictRedis(host='localhost', port=6379, db=7, charset='utf-8', decode_responses=True)
company_kor_to_eng_dict = ir.get_company_kor_to_eng_dict(redis_object)

ir.save_data_by_api(save_count, company_scores, industry_scores, redis_object, company_kor_to_eng_dict)
