#!/usr/bin/python

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


def load_data():
    train_df = pd.read_csv('train.csv', parse_dates=[0,])
    test_df = pd.read_csv('test.csv', parse_dates=[0,])
    sub_df = pd.read_csv('sampleSubmission.csv', parse_dates=[0,])
    
    print train_df.columns
    print test_df.columns
    
    train_df['datetime'] = train_df['datetime'].map(lambda d: d.strftime("%s")).astype(np.int64)
    
    for c in train_df.columns:
        print train_df[c].dtype, c, list(train_df.columns).index(c)
    
    xtrain = train_df.values[:,:9]
    ytrain = train_df.values[:,11]
    xtest = test_df.values
    ytest = sub_df.values
    
    return xtrain, ytrain, xtest, ytest

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtrain,
                                                                     ytrain,
                                                                     test_size=0.4, random_state=randint)
    model.fit(xTrain, yTrain)
    ytpred = model.predict(xTest)
    print 'roc', roc_auc_score(yTest, ytpred)
    return model.score(xTest, yTest)

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()
    
    print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape
    
    #pca = PCA(n_components=10)
    #x_pca = np.vstack([xtrain, xtest])
    #print x_pca.shape
    #pca.fit(xtrain)
    
    #xtrain = pca.transform(xtrain)
    #xtest = pca.transform(xtest)
    
    #compare_models(xtrain, ytrain)
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    print 'score', score_model(model, xtrain, ytrain)
    print model.feature_importances_
    #prepare_submission(model, xtrain, ytrain, xtest, ytest)
