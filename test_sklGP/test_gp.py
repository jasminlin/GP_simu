#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_svmlight_file
from sklearn.gaussian_process import GaussianProcess

import numpy as np
import sys, os
import random

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

def test(filename, reg):
    # initial the parameters
    #(X, y) = data_read('./data/eta_reg_605603_p0.data');
    # (X, y) = load_svmlight_file('./data/eta_reg_605603_p0.data');
    (X, y) = load_svmlight_file(filename);
    print("Data imported!");

    # divide into training and test_pyGPs datasets
    ratio = 0.8;
    num = len(y);
    idx = range(num);
    random.shuffle(idx);
    tn = int(np.floor(num*ratio));
    tr_idx = idx[:tn]
    te_idx = idx[tn:]
    # X_train = X[tr_idx]
    # y_train = y[tr_idx]
    tnum = int(np.floor(0.2*num))
    X_train = X[tr_idx[:tnum]]
    y_train = y[tr_idx[:tnum]]    
    X_test = X[te_idx]
    y_test = y[te_idx]

    # train linear ridge regression model
    print("Model training!");
    p1 = float(reg);
    clf = GaussianProcess(corr='squared_exponential');
    # clf.fit(list(X_train), y_train);
    clf.fit(X_train.toarray() , y_train);  
    abe = np.mean( abs(clf.predict(X_test.toarray()) - y_test)/y_test )
    #np.mean((clf.predict(X_test) - y_test) ** 2))
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
    print "Read file: ", fn
    print "Reg parameter: ", r
    test(fn, r);