# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:28:08 2023

@author: dr_vi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()


regressor.fit(X, y)

# print(regressor.predict([[8,9,7]]))

pickle.dump(regressor, open('model.pkl', 'wb'))

print("model.pkl file got generated")



# TEST PKL FILE BY INVOKING PREDICT METHOD 

model = pickle.load(open('model.pkl', 'rb'))

print("Prediction from pickle file ", model.predict([[3,4,5]]))