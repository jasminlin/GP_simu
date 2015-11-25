__author__ = 'lulin'

from sklearn.datasets import load_svmlight_file
import random
import numpy as np
from sklearn.gaussian_process import GaussianProcess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pyGPs

[XX, yy] = load_svmlight_file('./preliminary/eta_reg_605603_p0_1based.data')
print 'data loaded!'

# divide into training and test_pyGPs set
ratio1 = 0.7
ratio2s = [0.001, 0.005, 0.009, 0.013, 0.017, 0.021]#[0.001, 0.005, 0.01, 0.02]
mode1s = [1,2]
mode2s = [1,2,3,4,5]
fea_nums = [9, 12, 13, 107]
para1s = [np.log10(1), np.log10(0.01)]# [-10.,-8.,-6.,-4.,-2.,0.]
para2s = [np.log10(0.0001), np.log10(0.001), np.log10(0.01)]# [-2.,0.,2.,4.,6.,8.,10.]

def fit_GP(ratio1, ratio2, mode1, mode2, fea_num, para1,para2):
    num = int(len(yy) * ratio2)
    X = XX[:num, :fea_num]
    y = yy[:num]
    print y
    print XX.shape[0]
    print XX.shape[1]

    idx = range(num)
    random.shuffle(idx)
    tn = int(np.floor(num*ratio1))
    tr_idx = idx[:tn]
    te_idx = idx[tn:]
    X_train = X[tr_idx]
    X_test = X[te_idx]
    y_train = y[tr_idx]
    y_test = y[te_idx]

    # GP fit
    # gp = GaussianProcess(corr='linear')
    # gp.fit(X_train.toarray(), y_train)
    #
    # y_predict = gp.predict(X_test.toarray(), eval_MSE=True)
    # abe = np.mean( abs(y_predict - y_test)/y_test )

    # print abe

    if mode1 == 1:
        m = pyGPs.mean.Const()
    elif mode1 == 2:
        m = pyGPs.mean.Linear( D=X_train.shape[1] )

    if mode2 == 1:
        k = pyGPs.cov.Linear()
    elif mode2 == 2:
        k = pyGPs.cov.Periodic()
    elif mode2 == 3:
        k = pyGPs.cov.Poly()
    elif mode2 == 4:
        k = pyGPs.cov.RBF(para1, para2)
    elif mode2 == 5:
        k = pyGPs.cov.RQ()

    gp = pyGPs.GPR()
    gp.setPrior(mean=m, kernel=k)
    gp.getPosterior(X_train.toarray(), y_train)
    gp.optimize(X_train.toarray(), y_train)
    y_predict = gp.predict(X_test.toarray())
    # print y_predict
    abe = np.mean( abs(y_predict[0] - y_test)/y_test )
    out = open('./RBF_results.txt','a')
    out.write(str(ratio2) + ',' + str(mode1) + ',' + str(mode2) + \
              ',' + str(para1) + ',' + str(para2) + ',' + str(abe) + '\n')
    out.close()

    print ratio2, mode1, mode2, para1, para2, abe

if __name__ == '__main__':
    # for ratio2 in ratio2s:
    #     for model1 in model1s:
    #         for model2 in model2s:
    #             fit_GP(ratio1, ratio2, model1, model2)
    # try RBF
    for ratio2 in ratio2s:
        for para1 in para1s:
            for para2 in para2s:
                for i in range(10):
                    fit_GP(ratio1, ratio2, mode1s[0], mode2s[3], fea_nums[3], para1, para2)