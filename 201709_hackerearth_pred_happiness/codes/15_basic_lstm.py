# first create a mapping from word to integer in the vocabulary
import pandas as pd
import numpy as np
np.random.seed(42)
import common_vars as c_vars

from sklearn.model_selection import train_test_split

import gc
import pickle
'''
# read the train and test preprocessed files
df = pd.read_csv(c_vars.train_file_processed, encoding = 'cp1252')
df['Description_Clean'].fillna('', inplace = True, axis = 0)
df['Description_Clean_Adj'].fillna('', inplace = True, axis = 0)
df['is_train'] = 1

df_submit = pd.read_csv(c_vars.test_file_processed, encoding = 'cp1252')
df_submit['Description_Clean'].fillna('', inplace = True, axis = 0)
df_submit['Description_Clean_Adj'].fillna('', inplace = True, axis = 0)
df_submit['is_train'] = 0

df = pd.concat([df, df_submit])
df = df[['Description_Clean', 'is_train', 'User_ID', 'Is_Response']]
df.reset_index(inplace = True)

del df_submit
gc.collect()

global_vocab_mapping = {}
counter = 1
max_length_sentence = 0
sentence_length_dict = {}
word_freq_dict = {}

for i in range(len(df)):
    current_sentence = df.iloc[i]['Description_Clean'].split(' ')
    max_length_sentence = len(current_sentence) if len(current_sentence) > max_length_sentence else max_length_sentence
    try:
        sentence_length_dict[len(current_sentence)] += 1
    except KeyError:
        sentence_length_dict[len(current_sentence)] = 1
    # if i in [0, 1]:
        # print (df.iloc[i])
        # print (current_sentence)
    for word in current_sentence:
        try:
            word_freq_dict[word] += 1
        except KeyError:
            word_freq_dict[word] = 1
        if word not in global_vocab_mapping:
            global_vocab_mapping[word] = counter
            counter += 1

print (len(global_vocab_mapping))
# vocab size is 100674, but after removing words with corpus freq < 4, size drops to 18k, remove freq < 18, size drops to 7k
print (max_length_sentence)
# max sentence is 1358 words long
# from a freq distribution, 99% of the sentences have length 420, 95% have length 220

df['lstm_feature'] = 'dummy'
# df['lstm_feature'] = df['lstm_feature'].astype(object)
# remove the words which have corpus frequency of 3 or less
words_gt_18_freq = set([k for k in word_freq_dict if word_freq_dict[k] >= 18])
words_gt_18_freq_map = sorted([k for k in word_freq_dict if word_freq_dict[k] >= 18], reverse = True)
words_gt_18_freq_map = dict(zip(words_gt_18_freq_map, list(range(len(words_gt_18_freq_map)))))

for i in range(len(df)):
    current_sentence = df.iloc[i]['Description_Clean'].split(' ')
    current_sentence = list(filter(lambda x: x in words_gt_18_freq, current_sentence))
    current_sentence = current_sentence[:min(len(current_sentence), 220)]
    current_sentence = [words_gt_18_freq_map[word] for word in current_sentence]
    if i in [1, 10]:
        print (current_sentence)
    df.set_value(i, 'lstm_feature', list(current_sentence))
    # df.loc[i, 'lstm_feature'] = list(current_sentence)

# print (df.head)

df.to_csv('../inputs/train_test_preprocessed.csv', index = False)
'''
'''
for k in word_freq_dict:
    print (k.encode('utf-8'), word_freq_dict[k])
for k in sentence_length_dict:
    print (k, sentence_length_dict[k])
'''

import ast
df = pd.read_csv('../inputs/train_test_preprocessed.csv', encoding = 'cp1252',
                 converters={5:ast.literal_eval})
# get the vocabulary size
vocab_size = 0
for i in range(len(df)):
    vocab_size = max(max(df.iloc[i]['lstm_feature']),  vocab_size)
vocab_size += 1
df_train = df.loc[df['is_train'] == 1, :]
df_test = df.loc[df['is_train'] == 0, :]
df_train.reset_index(inplace = True)
df_test.reset_index(inplace = True)
X_train = []
X_val = []
y_train = []
y_val = []

for i in range(len(df_train)):
    if np.random.rand() > 0.1:
        X_train.append(df_train.loc[i, 'lstm_feature'])
        y_train.append(df_train.loc[i, 'Is_Response'])
    else:
        X_val.append(df_train.loc[i, 'lstm_feature'])
        y_val.append(df_train.loc[i, 'Is_Response'])

y_train = np.array(y_train)
y_val = np.array(y_val)

X_test = []
for i in range(len(df_test)):
    X_test.append(df_test.loc[i, 'lstm_feature'])

# LSTM for sequence classification in the IMDB dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

max_review_length = 220
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_val = sequence.pad_sequences(X_val, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vector_length = 4
# top_words = 18892
top_words = vocab_size
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
# model.add(Conv1D(filters=8, kernel_size=4, padding='same', activation='relu'))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=8, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(LSTM(16, dropout = 0.3))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=5, batch_size=64, class_weight = {0:2, 1:1})
# model.fit(X_train, y_train, epochs=4, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
model.save('../models/lstm_model.h5')
'''
y_test = model.predict(X_test)
df_test.loc[:,'Is_Response'] = y_test
# df_test['Is_Response'] = df_test['Is_Response'].apply(lambda x: 'happy' if x == 1 else 'not_happy')
df_test[['User_ID', 'Is_Response']].to_csv('../output/submit_20171117_1200_cnn.csv', index = False)
'''