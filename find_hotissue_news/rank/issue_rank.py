import datetime
from collections import Counter
import requests

class Issue_rank():

    # get datetime object by days_passed
    def get_datetime_object_by_days_passed(self, days_passed):  
        now = datetime.datetime.now()
        target_date = now - datetime.timedelta(days_passed)
        return target_date
    
    # 뉴스 기사 리스트를 시간을 기준으로 필터(from_data부터 to_date까지)
    def time_filter_news_list(self, news_list, from_date, to_date, time_column='post_date'):
        filtered_news = []
        for news in news_list:
            if (news[time_column] <= to_date) and (news[time_column] >= from_date):
                filtered_news.append(news)
        return filtered_news
    
    def get_company_or_industry_list(self, news_queried=list, column=str):
        source_list = []
        for news in news_queried:        
            for source in news[column]:
                source_list.append(source)
        return source_list

    # columns에 있는 단어들을 모두 더한 keyword list에서 단어들의 수를 세는 함수
    def count_company_or_industry(self, source_list):
        source_count = Counter(source_list).most_common()
        source_dictionary = dict()
        for source, count in source_count:
            source_dictionary[source] = count
        return source_dictionary
    
    # 어제와 일주일 평균을 기준으로 오늘 해당 기업or산업의 이슈점수 내는 함수
    def scoring_source(self, source_dict_today, source_dict_yesterday, source_dict_week, min_correction=2):
        scores = []
        for source, count in source_dict_today.items():
            expected_yesterday = source_dict_yesterday.get(source, 0)
            expected_week_average = source_dict_week[source]/7
            
            expected_count = max(expected_yesterday, expected_week_average, min_correction)
            score = count / expected_count
            scores.append((source, score, count, expected_yesterday, expected_week_average, min_correction))

        sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)        
        return sorted_scores
    
    def print_scores(self, score_list):
        for score in score_list:
            print(score[0],'/', '점수:', score[1], '/', '오늘:', score[2], '/', '어제:', score[3], 
                  '일주일 평균:', round(score[4],3), '최소 보정치:', score[5])
    
    # api에 num 개수만큼 실시간 기업/산업 저장하는 함수
    def save_data_by_api(self, num, company_scores, industry_scores, url):
        company_scores_cut = [company[0] for company in company_scores[:num]]
        industry_scores_cut = [industry[0] for industry in industry_scores[:num]]        
        send_data = {'company_list': company_scores_cut, 'industry_list': industry_scores_cut}
        print(send_data)

        response = requests.post(url, send_data)
        return response.status_code


