#!/usr/bin/env python
# coding: utf-8

# 상위 폴더 import경로에 추가
import sys, os

upper_file_path = '/'.join(os.getcwd().split('/')[:-1])

sys.path.append(upper_file_path)

# public library
import pandas as pd
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
import datetime
import pickle

# private library
from classification.news_classification import *

# load data

tp = Text_Preprocessing()

file_path = ''
df_raw = pd.read_csv(file_path, index_col=0)


df = df_raw[df_raw['new_category'] != '0']

df.reset_index(inplace=True, drop=True)


# text cleaning
df['clean_text'] = df['text'].apply(lambda text: tp.filter_alphabet(text))
df['clean_text'] = df['clean_text'].apply(lambda text: tp.word_tokenizing(text))


# remove stopwords
stopword_list = stopwords.words('english')

df['clean_text'] = df['clean_text'].apply(lambda text: tp.remove_stopwords(text, stopword_list))


# texts to sequences(text)

words_list = df.clean_text

tokenizer = Tokenizer()

tokenizer.fit_on_texts(words_list)

sequences = tokenizer.texts_to_sequences(words_list)


# get avg of length
count = 0
length = 0

for sequence in sequences:
    count += 1
    length += len(sequence)

average = length/count
print('average_length:', length/count)


# sequence padding
padded_sequences = pad_sequences(sequences, maxlen=int(average))


# texts to sequences(label)

label_list = df.new_category

label_tokenizer = Tokenizer()

label_tokenizer.fit_on_texts(label_list)

label_sequences = label_tokenizer.texts_to_sequences(label_list)

answers = to_categorical(label_sequences)


X_train, X_test, Y_train, Y_test = train_test_split(padded_sequences, answers)

print('X_train:', X_train.shape)
print('X_test:',X_test.shape)
print('Y_train:',Y_train.shape)
print('Y_test:',Y_test.shape)


# model_choice ['CNN', 'LSTM']

model_choice = 'CNN'
print('model_choice:', model_choice)

if model_choice == 'CNN':

    kwargs = {'vocab_size':len(tokenizer.word_index) + 1,
            'embedding_dim': 128,
            'hidden_states':32,
            'num_classes':len(answers[0]),
            'num_filters': 6,
            'dropout_rate': 0.3,
            'hidden_dimension': 128}

    model = CNNClassifier(**kwargs)
    
elif model_choice == 'LSTM':
    
    kwargs = {'vocab_size':len(tokenizer.word_index) + 1,
            'embedding_dim': 128,
            'hidden_states':32,
            'num_classes':len(answers[0])}

    model = LSTM_Classifier(**kwargs)


# assign model save directory

now = datetime.datetime.now()
now_str = now.strftime('%Y%m%d%H%M')
model_dir = 'model/' + '_'.join([model_choice, now_str]) + '/'
pickle_dir = model_dir + 'pickle/'
data_dir = model_dir + 'data/'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
    
if not os.path.isdir(pickle_dir):
    os.mkdir(pickle_dir)
    
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)


# save kwargs

kwargs_save_path = data_dir + 'kwargs.pickle'

with open(kwargs_save_path, 'wb') as handle:
    pickle.dump(kwargs, handle)


# save pad length

pad_length_save_path = data_dir + 'pad_length_save_path.pickle'

with open(pad_length_save_path, 'wb') as handle:
    pickle.dump(int(average), handle)


# save input shape

input_shape_save_path = data_dir + 'input_shape.pickle'

input_shape = {'input_shape': X_train.shape}

with open(input_shape_save_path, 'wb') as handle:
    pickle.dump(input_shape, handle)


# save pickle_file

pickle_save_path = pickle_dir + 'tokenizer.pickle'

with open(pickle_save_path, 'wb') as handle:
    pickle.dump(tokenizer, handle)

    
label_pickle_save_path = pickle_dir + 'label_tokenizer.pickle'

with open(label_pickle_save_path, 'wb') as handle:
    pickle.dump(label_tokenizer, handle)


# set EarlyStopping and ModelCheckpoint

model_name = model_dir + '{}.h5'.format(model_choice)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint(model_name, monitor='val_acc', save_weights_only=True)

#compile model

loss = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']

compile_arguments = {
    'loss':loss,
    'optimizer': optimizer,
    'metrics': metrics
}

compile_arguments_save_path = data_dir + 'compile_arguments.pickle'

with open(compile_arguments_save_path, 'wb') as handle:
    pickle.dump(compile_arguments, handle)

model.compile(loss=compile_arguments['loss'], optimizer=compile_arguments['optimizer'], metrics=compile_arguments['metrics'])


# do learning
epochs = 20
history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test),
                   callbacks=[es, mc])

# get f1_score

em = Evaluate_Model()
num_category = len(Y_train[0]) - 1
prediction_list = model.predict(X_test)
answer_list = Y_test

confusion_matrix = em.get_confusion_matrix(num_category, prediction_list, answer_list)
recall_score = em.get_recall_score(confusion_matrix)
precision_score = em.get_precision_score(confusion_matrix)
f1_score = em.get_f1_score(precision_score, recall_score)

accuracy_score = em.get_accuaracy_score(confusion_matrix)

print('f1_score:', f1_score)
print('accuracy_score:', accuracy_score)
print('f1_score - accuracy_score:', f1_score - accuracy_score)

model_score = {'f1_score':f1_score, 'accuracy_score': accuracy_score}

# save score

model_score_save_path = data_dir + 'model_score.pickle'

with open(model_score_save_path, 'wb') as handle:
    pickle.dump(model_score, handle)

