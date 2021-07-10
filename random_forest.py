# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 18:18:56 2021

@author: HP OMEN
"""
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('Crop_recommendation.csv')

df.dtypes

df.head()

df.rainfall.describe()

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']    
    

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state =1)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)


from sklearn.metrics import accuracy_score
x = accuracy_score(y_test, predictions)
x

###saving model to disk
pickle.dump(rf, open('model.pkl','wb'))


#test prediction
# data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
# prediction = rf.predict(data)
# print(prediction)
