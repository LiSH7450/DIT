# -*- coding: utf-8 -*-

import pandas as pd 
from sklearn import preprocessing
from sklearn import metrics
import joblib
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


filepath_train = 'Training set.csv'
filepath_FAERS = 'FDA FAERS dataset.csv'
filepath_AEDs = 'AEDs dataset.csv'

X = pd.read_csv(filepath_train,header = 0,index_col = 0)
X_FAERS = pd.read_csv(filepath_FAERS,header = 0,index_col = 0)
X_AEDs = pd.read_csv(filepath_AEDs,header = 0,index_col = 0)


y_train = [1]*67+[0]*45


model = Pipeline([('ss',StandardScaler()), ('svc', SVC(kernel='rbf',gamma=0.001, C=2.0,random_state=5,probability=True))])
model.fit(X,y_train) 

# joblib.dump(model, "SVM.m")
# model = joblib.load("SVM.m")


y_predict_FAERS = model.predict(X_FAERS)
y_predict_AEDs = model.predict_proba(X_AEDs)
