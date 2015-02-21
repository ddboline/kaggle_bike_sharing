#!/usr/bin/python

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd

import datetime

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score

def load_data():
    train_df = pd.read_csv('train.csv', parse_dates=[0,])
    test_df = pd.read_csv('test.csv', parse_dates=[0,])
    sub_df = pd.read_csv('sampleSubmission.csv', parse_dates=[0,])
    
    print train_df.columns
    print test_df.columns
    
    #train_df['datetime'] = train_df['datetime'].map(lambda d: d.strftime("%s")).astype(np.int64)
    train_df['datetime'] = train_df['datetime'].map(lambda d: d.hour).astype(np.int64)
    test_df['datetime'] = test_df['datetime'].map(lambda d: d.hour).astype(np.int64)

    print train_df.describe()
    print train_df.columns[:9]
    
    for c in train_df.columns:
        print train_df[c].dtype, c, list(train_df.columns).index(c)
    
    xtrain = train_df.values[:,0]
    ytrain = train_df.values[:,11]
    xtest = test_df.values[:,0]
    ytest = sub_df['datetime'].values
   
    print ytrain[:10]
   
    return xtrain, ytrain, xtest, ytest

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtrain,
                                                                     ytrain,
                                                                     test_size=0.4, random_state=randint)
    model.fit(xTrain, yTrain)
    #cvAccuracy = np.mean(cross_val_score(model, xtrain, ytrain, cv=2))
    #print yTest
    return model.score(xTest, yTest)

def prepare_submission(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ytest2 = model.predict(xtest)
    dateobj = map(datetime.datetime.fromtimestamp, ytest)
    
    df = pd.DataFrame({'datetime': dateobj, 'count': ytest2}, columns=('datetime','count'))
    df.to_csv('submission.csv', index=False)
    
    return

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()
    
    print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape
    
    #pca = PCA()
    #x_pca = np.vstack([xtrain, xtest])
    #print x_pca.shape
    #pca.fit(xtrain)
    
    #xtrain = pca.transform(xtrain)
    #xtest = pca.transform(xtest)
    
    #compare_models(xtrain, ytrain)
    #model = RandomForestClassifier(n_estimators=200)
    model = RandomForestClassifier()
    #model = SVC(kernel="linear", C=0.025)
    print 'score', score_model(model, xtrain, ytrain)
    print model.feature_importances_
    #prepare_submission(model, xtrain, ytrain, xtest, ytest)
