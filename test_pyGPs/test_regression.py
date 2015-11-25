#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import load_svmlight_file

import pandas as pd
import numpy as np
import sys, os
import random
import re

def data_read(filename):
    # read the input data
    X = [];
    y = [];
    with open(filename) as f:
        # Matrix = [map(float, line.split()) for line in f]
        for line in f:
            sample = line.split();
            y.append(sample[0])
            x = [];
            for element in sample[1:]:
                tmp = element.split(':');
                x.append(tmp[1]);
            x = map(float, x);
            X.append(x);
    y = map(float, y);
    X = np.array(X);
    y = np.array(y);
    return (X, y);


def test_only(filename, linkindex, reg):
    # initial the parameters
    #(X, y) = data_read('./data/eta_reg_605603_p0.data');
    # (X, y) = load_svmlight_file('./data/eta_reg_605603_p0.data');
    (X, y) = load_svmlight_file(filename);
    #X = X[:,:9]
    print("Data imported!");

    # divide into training and test_pyGPs datasets
    ratio = 0.8;
    num = len(y);
    idx = range(num);
    random.shuffle(idx);
    tn = int(np.floor(num*ratio));
    tr_idx = idx[:tn]
    te_idx = idx[tn:]
    X_train = X[tr_idx]
    X_test = X[te_idx]
    y_train = y[tr_idx]
    y_test = y[te_idx]

    # get link id


    # train linear ridge regression model
    print("Model training!");
    p1 = float(reg);
    clf = linear_model.Ridge (alpha = p1);
    # clf.fit(list(X_train), y_train);
    clf.fit(X_train, y_train);
    # abe = np.mean( abs(clf.predict(X_test) - y_test)/y_test )
    #np.mean((clf.predict(X_test) - y_test) ** 2))
    
    pt = clf.predict(X_test);
    gt = y_test;
    abe = np.mean(abs(pt - gt)/gt)
    print("Absolute error is: %.2f" % abe );

    #gt_pd = pd.DataFrame([gt, pd], columns = ['groundtruth', 'prediction']);
    df_gt = pd.DataFrame(gt, columns = ['groundtruth']);
    df_pt = pd.DataFrame(pt, columns = ['prediction']);
    gt_pt = df_gt.join(df_pt);
    sn = filename.split("/")[-1]
    gt_pt.to_csv('result_'+sn);
    print("Save the results ... ");

def train(filename, reg):
    (X, y) = load_svmlight_file(filename);
    print("Data imported!");

    p1 = float(reg);
    clf = linear_model.Ridge (alpha = p1);
    clf.fit(X, y);
    return clf

def test(filename, clf):
    (X, y) = load_svmlight_file(filename);

    ratio = 0.3
    num = len(y);
    idx = range(num);
    random.shuffle(idx);
    tn = int(np.floor(num*ratio));
    te_idx = idx[:tn]
    X = X[te_idx]
    y = y[te_idx]

    pt = clf.predict(X);
    abe = np.mean(abs(pt - y)/y)
    print("Absolute error is: %.2f" % abe );


if __name__ == '__main__':
    # unit test_pyGPs
    if len(sys.argv) != 3:
        print 'Usage: python input_file_name'
        exit(1)
    #f_input = sys.argv[1]
    #f_output = sys.argv[2]
    fn = sys.argv[1] #'./data/eta_reg_605603_p1.data'
    r = sys.argv[2]
    print "Read folder: ", fn
    print "Reg parameter: ", r
    
    flist = os.listdir(fn);
    #clf = train(fn+'/'+flist[0], r);
    tf = os.path.join(fn, flist[0]);
    clf = train(tf, r);
    testf = os.path.join(fn, flist[1]);
    test(testf,clf);
    # for f in flist:
    #     #test_pyGPs(fn+'/'+f,r);
    #     train(fn)
    #     test_pyGPs(fn+'/'+f,r);
    #     i += 1
    #     if i>2:
    #     	break;




