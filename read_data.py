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
# X% of train.csv is for actual training.
# 1-X is for testing.

"""
train et test sont des Dataframes de Pandas.

Il faut créer un générateur qui renvoie un nombre N de lignes de données
directement compréhensibles par un modèle TF.

Auparavent, il faut séparer train en 2 parties : 80 % des données seront 
selectionnées au hasard dès le début et serviront à l'entrainement.
Les 20% restants seront utilisés pour le test. Les données de test de test.csv
resteront au chaud.

Au final, on a cinq gros fichiers :
- les 80% de données provenant de train.csv et déjà prêtes. (input)
- les 20% de données provenant de train.csv et déjà prêtes. (input)
- les données de test.csv, déjà prêtes (input)
- les 80% de résultats désirés provenant de train.csv, sous forme de 0 et 1 (output)
- les 20% de résultats désirés provenant de train.csv, sous forme de 0 et 1 (output)

Comment traiter les données manquantes? (age : 0?)

NEXT TO DO: 
--> http://pandas.pydata.org/pandas-docs/stable/10min.html

"""

def modify_columns(raw_data):
    # Transform 'female' into 1 and 'male' into -1.
    raw_data.ix[raw_data.Sex == 'female', 'Sex'] = 1
    raw_data.ix[raw_data.Sex == 'male', 'Sex'] = -1
    
    # Transform 'NaN age' into 0. ## To be discussed.
    raw_data['Age'].fillna(raw_data['Age'].mean(), inplace=True)
    
    raw_data['Age'] = (raw_data['Age'] - raw_data['Age'].mean())/(raw_data['Age'].max() - raw_data['Age'].min())
    
    raw_data['Pclass'] = (raw_data['Pclass'] - raw_data['Pclass'].mean() )/ 2
    
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
    
    ## Look for the first lines
    #print(raw_data.head())
    
    
    return raw_data



def transform_raw_data(raw_data, percentage_train):
    
    raw_data = modify_columns(raw_data)

    # Choose only these columns for the training.
    raw_data = raw_data[["Pclass", "Sex", "Age", "Name", "SibSp", "Fare", "Parch", "Survived"]]
    # Add a column "NotSurvived".
    raw_data["NotSurvived"] = pd.Series([0] * len(raw_data.Sex))
    raw_data.ix[raw_data["Survived"] == 0, 'NotSurvived'] = 1
    
    ## Look for the first lines
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
    raw_data = pd.read_csv("./data/train.csv")
    
    _, test_data = transform_raw_data(raw_data, PERCENTAGE_TRAIN)
        
    return (test_data[:, :-2], test_data[:, -2:])

def get_submit_data():
    raw_submit_data = pd.read_csv("./data/test.csv")
    raw_submit_data = modify_columns(raw_submit_data)
    raw_submit_data = raw_submit_data[["Pclass", "Sex", "Age", "Name", "SibSp", "Fare", "Parch"]]
    
    return np.array(raw_submit_data)



if __name__ == '__main__':
   
    a = get_new_batch(2)
    x_batch, y_batch = next(a)
    print("x_batch is:\n", x_batch, '\n')
    print("y_batch is:\n", y_batch, '\n')






