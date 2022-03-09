
import os

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import json
import jsonschema
import jsonpickle
import random
import csv
import pickle

import scipy.stats
import scipy.spatial
import scipy.special
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import sklearn.metrics
np.set_printoptions(precision=2)


def test_cls_param(Xs, ys, ne, md):

    print('ne', ne, 'md', md)
    
    # train_accs = []
    test_accs = []
    roc_aucs = []
    ce_losses = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(Xs):
        try:
        # if True:
            train_X, test_X = Xs[train_index], Xs[test_index]
            train_y, test_y = ys[train_index], ys[test_index]
    
            cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', warm_start=False)
            cls.fit(train_X, train_y)
    
            preds = cls.predict(train_X)
            
            preds = cls.predict(test_X)
            
            fp = 0
            tp = 0
            fn = 0
            tn = 0
            tps = []
            fps = []
            fns = []
            for i in range(test_y.shape[0]):
                if preds[i] > 0.5 and test_y[i] == 1:
                    tp += 1
                elif preds[i] > 0.5 and test_y[i] == 0:
                    fp += 1
                elif preds[i] <= 0.5 and test_y[i] == 0:
                    tn += 1
                elif preds[i]<= 0.5 and test_y[i] == 1:
                    fn += 1
            test_accs.append((tp+tn)/float(tp+fp+fn+tn))
    
            confs = cls.predict_proba(test_X)[:,1]
    
            confs = np.clip(confs, 0.025, 0.975)
    
            # lr_reg = LogisticRegression(C=100, max_iter=10000, tol=1e-4)
            lr_reg = LogisticRegression(max_iter=10000, tol=1e-4)
    
            lr_reg.fit(np.concatenate([train_X, cls.predict_proba(train_X)], axis=1) , train_y)
            confs = lr_reg.predict_proba( np.concatenate([test_X, cls.predict_proba(test_X)], axis=1) )[:,1]
    
            confs = np.clip(confs, 0.025, 0.975)
    
            roc_auc = sklearn.metrics.roc_auc_score(test_y, confs)
            celoss  = sklearn.metrics.log_loss(test_y, confs)
            roc_aucs.append(roc_auc)
            ce_losses.append(celoss)
        except:
            continue
    test_accs  = np.array(test_accs)
    roc_aucs = np.array(roc_aucs)
    ce_losses  = np.array(ce_losses)
    print('test accs', np.mean(test_accs), np.var(test_accs), test_accs, )
    print('test roc_aucs', np.mean(roc_aucs), np.var(roc_aucs), roc_aucs)
    print('test ce_losses', np.mean(ce_losses), np.var(ce_losses), ce_losses, )

    return np.mean(ce_losses), np.mean(roc_aucs), np.mean(test_accs)

# xs, ys = pickle.load(open('./new_learned_parameters_qa1/features.pkl', 'rb')) 
xs, ys = pickle.load(open('./new_learned_parameters_sc2/features.pkl', 'rb')) 
print('xs', xs.shape, 'ys', ys.shape)

# TODO set the nes and mds to be tunable parameters
ne0 = 2000
md0 = 2
# try:
if True:
    params = []
    ces = []
    for ne in [200, 2000, 5000,]:
        for md in [2,4,6]:
            params.append((ne, md))
            ce, auc, acc = test_cls_param(xs, ys, ne, md)
            ces.append(ce)
    ces = np.array(ces)
    best_param = params[np.argmin(ces)]
# except:
#     print('error in training classifier')
#     best_param = (ne0, md0)

print('best_param', best_param)
ne, md = best_param
cls = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion='entropy', warm_start=False)
cls.fit(xs, ys)
lr_reg = LogisticRegression(max_iter=10000, tol=1e-4)
lr_reg.fit(np.concatenate([xs, cls.predict_proba(xs)], axis=1) , ys)
confs = lr_reg.predict_proba( np.concatenate([xs, cls.predict_proba(xs)], axis=1) )[:,1]
confs = np.clip(confs, 0.025, 0.975)
print('after confs', confs)
roc_auc = sklearn.metrics.roc_auc_score(ys, confs)
ce_loss  = sklearn.metrics.log_loss(ys, confs)
print('overall roc_auc', roc_auc, 'celoss', ce_loss)
