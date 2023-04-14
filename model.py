#Import Evaluation matrics to evaluate our models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle

#Get dataset
x_train = pd.read_csv("./datasets/xtrain_data.csv")
x_test = pd.read_csv("./datasets/xtest_data.csv")

y_train = pd.read_csv("./datasets/ytrain_data.csv")
y_test = pd.read_csv("./datasets/ytest_data.csv")

def buildModel(): 

    # Initialize and fit on train datasets
    cls_r =RandomForestClassifier(n_estimators=600,min_samples_split=10,max_depth=100,criterion="gini")
    cls_r.fit(x_train, y_train.values.ravel())

    ypred = cls_r.predict(x_test)
    score = accuracy_score(y_test, ypred)
    return {"score": score}

def buildAndSaveModel():

    # Initialize and fit on train datasets
    cls_r =RandomForestClassifier(n_estimators=600,min_samples_split=10,max_depth=100,criterion="gini")
    cls_r.fit(x_train, y_train.values.ravel())

    pickle.dump(cls_r, open("model.pkl","wb"))
