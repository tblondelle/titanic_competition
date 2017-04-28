# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 00:54:09 2017

@author: thomas
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import numpy as np



PERCENTAGE_TRAIN = 0.9
# Percentage of 'train.csv' for actual training.
# The rest is for testing.



def modify_columns(raw_data):
	"""
	Transform the raw data into something pre-processed.
	
	More specifically, 
	  - 'male' and 'female' are turned into -1 and 1,
	  - 'Age', 'Pclass', 'SibSb', 'Fare', 'Parch' are centered
	  	and normalized,
	  - 'Name' is checked if it contains special titles for people
	  	(like Master, Miss, Col, Reverend...) that may be 
	  	susceptible to influence their survival.
	"""
	
    # Transform 'female' into 1 and 'male' into -1.
    raw_data.ix[raw_data.Sex == 'female', 'Sex'] = 1
    raw_data.ix[raw_data.Sex == 'male', 'Sex'] = -1
    
    # Transform 'NaN' age into the average age.
    raw_data['Age'].fillna(raw_data['Age'].mean(), inplace=True)
    
    # Center and normalize data.
    raw_data.Age = (raw_data.Age - raw_data.Age.mean())/(raw_data.Age.max() - raw_data.Age.min())
    raw_data.Pclass = (raw_data.Pclass - raw_data.Pclass.mean() )/ 2
    raw_data.SibSp = (raw_data.SibSp - raw_data.SibSp.mean())/ raw_data.SibSp.max()
    raw_data.Fare = (raw_data.Fare - raw_data.Fare.mean())/ (raw_data.Fare.max() - raw_data.Fare.min())
    raw_data.Parch = (raw_data.Parch - raw_data.Parch.mean()) / raw_data.Parch.max()
    
    # Take into account the not Mrs or Mr.
    names = list(raw_data["Name"])
    for i in range(len(names)):
        names[i] = names[i].split(', ')[1]
        names[i] = names[i].split(' ')[0]
        if names[i] in ['Mrs.', 'Mr.']:
            names[i] = 0.0
        else:
            names[i] = 1.0
    
    raw_data["Name"] = pd.Series(names)    
    
    # Set PassengerId as the main index.
    raw_data.set_index("PassengerId", inplace=True)
    
    return raw_data


def transform_raw_data(raw_data, percentage_train):
    """
    Prepare the train and test set from raw_data.
    """
    
    raw_data = modify_columns(raw_data)

    # Choose only these columns for the training.
    raw_data = raw_data[["Pclass", "Sex", "Age", "Name", "SibSp", "Fare", "Parch", "Survived"]]
    
    # Add column: NotSurvived = 1 - Survived (#TODO)
    raw_data["NotSurvived"] = pd.Series([0] * len(raw_data.Sex))
    raw_data.ix[raw_data["Survived"] == 0, 'NotSurvived'] = 1
    
    # Look for the first lines
    #print(raw_data.head())
    
    raw_data = np.array(raw_data)
    
    idx_where_train_stops = int(raw_data.shape[0]*percentage_train)
    
    train_set = raw_data[ :idx_where_train_stops, : ]
    test_set = raw_data[ idx_where_train_stops:, : ]
    
    return (train_set, test_set)
    

def get_new_batch(n_lines):
    """ 
    A generator that returns a couple of the input and the output of the 
    training set.
    
    Argument:
        * n_lines: (Integer) the batch size.
        
    Returns:
        * A couple (A,B). A is a numpy array of shape (n_lines, X) with
        X corresponing to the number of features given. B is also a numpy
        array of shape (n_lines) of 0 and 1.    
    """
    raw_data = pd.read_csv("./data/train.csv")
    train_data, _ = transform_raw_data(raw_data, PERCENTAGE_TRAIN)
    
    while True:
        # For each epoch, shuffle the data.
        np.random.shuffle(train_data)
        
        x_data = train_data[:, :-2]
        y_data = train_data[:, -2:]
        
        # Get n_lines of this data.
        values_of_index = np.arange(0, x_data.shape[0], n_lines)[:-1]
        for i in values_of_index:
            yield (x_data[i:i+n_lines, :], y_data[i:i+n_lines])


def get_test_data():
	"""
	Returns the couple (test_set_x, test_set_y).
	"""

    raw_data = pd.read_csv("./data/train.csv")
    
    _, test_data = transform_raw_data(raw_data, PERCENTAGE_TRAIN)
        
    return (test_data[:, :-2], test_data[:, -2:])


def get_submit_data():
	"""
	Returns the submit set x.
	"""

    raw_submit_data = pd.read_csv("./data/test.csv")
    raw_submit_data = modify_columns(raw_submit_data)
    raw_submit_data = raw_submit_data[["Pclass", "Sex", "Age", "Name", "SibSp", "Fare", "Parch"]]
    
    return np.array(raw_submit_data)
    

if __name__ == '__main__':
   
   	# Some tests.
    a = get_new_batch(2)
    x_batch, y_batch = next(a)
    print("x_batch is:\n", x_batch, '\n')
    print("y_batch is:\n", y_batch, '\n')

