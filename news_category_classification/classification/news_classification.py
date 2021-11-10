import re
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, Embedding
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Text_Preprocessing():
    
    def parse_mind_dataset(self, file_path):
        news_df_ori = pd.read_csv(file_path, header=None, sep='\t')
        news_column = ['NewsId', 'Category', 'SubCategory', 'Title', 'Abstract','URL', 'TitleEntities', 'AbstractEntites']
        news_df_ori.columns = news_column
        news_df_classi = news_df_ori[['Title', 'Category', 'SubCategory']]
        return news_df_classi
    
    def parse_bbc_ag_dataset(self, file_path):
        df = pd.read_csv(file_path)
        return df
    
    def parse_webapp_dataset(self, dir_path):
        file_list = os.listdir(dir_path)
        df_all = pd.DataFrame()
        for file in file_list:
            df = pd.read_csv(dir_path+file, index_col=0)
            df_all = pd.concat([df_all, df])
        df_all.drop_duplicates(inplace=True)
        df_all.reset_index(inplace=True)
        df_all.drop('index', axis=1, inplace=True)
        return df_all
    
    def parse_huff_dataset(self, file_path):
        with open(file_path, 'r') as file:
            string_data = file.readlines()
            dict_data = list(map(eval, string_data))
            df = pd.DataFrame(dict_data)
        return df
    
    def filter_alphabet(self, text):
        filtered_text = re.sub(r"[^a-zA-Z]", " ", text)
        return filtered_text
    
    def word_tokenizing(self, text):
        word_token = word_tokenize(text)
        return word_token

    def remove_stopwords(self, word_token_list, stopword_list):
        cleaned_word_list = [word.lower() for word in word_token_list if (word not in stopword_list)]
        return cleaned_word_list
    

class Evaluate_Model():
    
    # answer should start from number 1, not 0
    def get_confusion_matrix(self, num_category, prediction_lists, answer_lists):
        confusion_matrix = [[0 for _ in range(num_category)] for _ in range(num_category)]
        for prediction_list, answer_list in zip(prediction_lists, answer_lists):
            prediction = np.argmax(prediction_list)
            answer = np.argmax(answer_list)
            confusion_matrix[answer-1][prediction-1] += 1
        return confusion_matrix
    
    def get_accuaracy_score(self, confusion_matrix):
        answer_count = 0
        sum_count = 0
        for idx,row in enumerate(confusion_matrix):
            answer_count += row[idx]
            sum_count += sum(row)
        accuarcy_score = answer_count/sum_count
        return accuarcy_score
    
    def get_recall_score(self, confusion_matrix):
        total_score = 0
        num_of_category = len(confusion_matrix)
        for idx, row in enumerate(confusion_matrix):
            recall_score = row[idx] / sum(row)
            total_score += recall_score
        avg_recall_score = total_score/num_of_category
        return avg_recall_score

    def get_precision_score(self, confusion_matrix):
        transposed_matrix = np.array(confusion_matrix).T
        total_score = 0
        num_of_category = len(confusion_matrix)
        for idx, row in enumerate(transposed_matrix):
            precision_score = row[idx] / sum(row)
            total_score += precision_score
        avg_precision_score = total_score/num_of_category
        return avg_precision_score
    
    def get_f1_score(self, precision_score, recall_score):
        add = precision_score + recall_score
        multiple = precision_score * recall_score
        f1_score = (2 * multiple) / add
        return f1_score
    
    
class LSTM_Classifier(tf.keras.Model):
    
    def __init__(self, **kargs):
        super(LSTM_Classifier, self).__init__()
        self.embedding = Embedding(kargs['vocab_size'], kargs['embedding_dim'])
        self.lstm = LSTM(kargs['hidden_states'])
        self.dense = Dense(kargs['num_classes'], activation='softmax')
        
    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        return self.dense(x)

    
class CNNClassifier(tf.keras.Model):
    
    def __init__(self, **kargs):
        super(CNNClassifier, self).__init__()
        self.embedding = layers.Embedding(input_dim=kargs['vocab_size'],
                                         output_dim=kargs['embedding_dim'])
        self.conv_list = [layers.Conv1D(filters=kargs['num_filters'],
                                       kernel_size=kernel_size,
                                       padding='valid',
                                       activation = tf.keras.activations.relu,
                                       kernel_constraint = tf.keras.constraints.MaxNorm(max_value=3.)) for kernel_size in [3,4,5]]
        self.pooling = layers.GlobalMaxPooling1D()
        self.dropout = layers.Dropout(kargs['dropout_rate'])
        self.fc1 = layers.Dense(units=kargs['hidden_dimension'], activation = tf.keras.activations.relu)
        self.fc2 = layers.Dense(units=kargs['num_classes'], activation=tf.keras.activations.softmax,
                               kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3))
        
    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = tf.concat([self.pooling(conv(x)) for conv in self.conv_list], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Predict_Model:
    
    def return_text(self, num, title_url, summarized_url):
        title_response = requests.get(title_url + str(num) +'/')
        summa_response = requests.get(summarized_url + str(num) + '/') 
        if (not title_response )or (not summa_response):
            return 
        title = title_response.json()['eng_title']
        summa = summa_response.json()['summarized_content']
        text = title + '. ' + summa
        return text
    
    def return_padded_sequence(self, text, tokenizer, max_len):
        tp = Text_Preprocessing()

        # clean text 
        filter_alphabet_text = tp.filter_alphabet(text)
        word_tokenized = tp.word_tokenizing(filter_alphabet_text)
        stopword_list = stopwords.words('english')
        clean_word_list = tp.remove_stopwords(word_tokenized, stopword_list)

        # vectorization
        tokenizer_dict = tokenizer.word_index
        sequence = []
        for word in clean_word_list:
            sequence.append(tokenizer_dict.get(word,0))

        # pad sequence
        max_len = 17
        padded_sequence = pad_sequences([sequence], 17)

        return padded_sequence
    
    def make_prediction(self, sequence, model, label_tokenizer):
        # prediction
        pred =model.predict(sequence)
        max_pred_index = np.argmax(pred)

        # make label dictionary
        label_tokenizer_dict= dict()
        for category, num in label_tokenizer.word_index.items():
            label_tokenizer_dict[num] = category

        pred_category = label_tokenizer_dict[max_pred_index]

        return pred_category    
    
    def get_category(self, model, num, tokenizer, label_tokenizer, eng_kor_dict, max_len):
        text = self.return_text(num)
        if not text:
            return 
        sequence = self.return_padded_sequence(text, tokenizer, max_len)
        prediction = self.make_prediction(sequence, model, label_tokenizer)

    #     kor_title = requests.get(title_url + str(num) +'/').json()['ko_title']
    #     print(kor_title, prediction)
        prediction_kor = eng_kor_dict[prediction]

        return prediction_kor    
    
    
    
    
    
    
    
    
    
    
