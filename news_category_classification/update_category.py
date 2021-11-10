#!/usr/bin/env python
# coding: utf-8

import sys, os

pardir_path = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(pardir_path)

# public lib
import pickle
import numpy as np
import requests
from nltk.corpus import stopwords
import time

# private lib
from classification.news_classification import * 


# set path
model_dir = 'model/CNN_202111011737/' 
model_name = 'CNN'

model_path = model_dir + model_name + '.h5'
data_dir = model_dir + 'data/'

with open(data_dir + 'model_score.pickle', 'rb') as handle:
    model_score = pickle.load(handle)

print(model_score)


# load tokenizer

tokenizer_save_path = model_dir + 'pickle/tokenizer.pickle'

with open(tokenizer_save_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

    
# load label tokenizer

label_tokenizer_save_path = model_dir + 'pickle/label_tokenizer.pickle'

with open(label_tokenizer_save_path, 'rb') as handle:
    label_tokenizer = pickle.load(handle)


# make label dictionary
label_tokenizer_dict= dict()
for category, num in label_tokenizer.word_index.items():
    label_tokenizer_dict[num] = category   

    
# get max length
with open(data_dir + 'pad_length_save_path.pickle', 'rb') as handle:
    max_len = pickle.load(handle)

# get translation dictionary

eng_kor_dict_path = './data/merge/eng_kor_dictionary.xlsx'

eng_kor_dict_df = pd.read_excel(eng_kor_dict_path, index_col=0)

eng_kor_dict = dict()

for eng_cat, kor_cat in zip(eng_kor_dict_df.eng_category, eng_kor_dict_df.kor_category):
    eng_kor_dict[eng_cat] = kor_cat


# get max length

with open(data_dir + 'pad_length_save_path.pickle', 'rb') as handle:
    max_len = pickle.load(handle)


# load model

with open(data_dir + 'kwargs.pickle', 'rb') as handle:
    kwargs = pickle.load(handle)
    
with open(data_dir + 'input_shape.pickle', 'rb') as handle:
    input_shape = pickle.load(handle)

if model_name == 'CNN':    
    model = CNNClassifier(**kwargs)
    
elif mode_name == 'LSTM':
    model = LSTM_Classifier(**kwargs)

with open (data_dir + 'compile_arguments.pickle', 'rb') as handle:
    compile_arguments = pickle.load(handle)
model.compile(loss=compile_arguments['loss'], optimizer=compile_arguments['optimizer'], metrics=compile_arguments['metrics'])
model.build(input_shape['input_shape'])
model.load_weights(model_path)


# run scripts

# import predict class

pm = Predict_Model()

index = 0

while True:

    try:
        # api for check latest index
        check_url = ''
        latest_index = requests.get(check_url).json()['summarizeddata_last']
    
        # api for update category
        url = ''
        category = pm.get_category(model, index, tokenizer, label_tokenizer, eng_kor_dict, max_len)
        send_data = {'crawlingdata':index, 'related_category': category}
        if (not category) and (latest_index > index):
            print(index, 'no data')
            index += 1
            continue
        response = requests.put(url + str(index) + '/',data = send_data)
        print(index, send_data)
        print(response.status_code)

        if response.status_code == 200:
            index += 1
        else:
            check_data = requests.get(url + str(index) +'/').json()
            if check_data['related_category']:
                index += 1
        
        if latest_index <= index:
            print('---resting---')
            time.sleep(1)
            
    except Exception as e:
        print('error occured:', e)
        time.sleep(1)

