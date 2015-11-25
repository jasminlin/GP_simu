__author__ = 'lulin'

import numpy as np
from scipy import *
import csv
from collections import defaultdict
import random
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
from sklearn.cross_validation import cross_val_score, KFold
from numpy.matlib import rand
from scipy.linalg.decomp_svd import orth

linkM = 50 #13169
dateD = 5
timeT = 48

DIS_CONNECT = 100.
length = 0.32
mu = 35.

data_dir = './GPsimu'
speeds_file_pre = data_dir + '/simu_speeds'

def gen_network():
    print 'begin generate road network...'

    connect_info = defaultdict(lambda: 0)
    net = np.array([[DIS_CONNECT] * linkM] * linkM)
    for i in range(linkM):
        ava = linkM - i - 1

        # randomize and store the links related to i
        conlist = random.sample(range(i+1, linkM), random.randint(ava / 2, ava))
        connect_info[i] = conlist

        # randomize the distance
        for j in range(len(conlist)):
            net[i][conlist[j]] = random.uniform(1, 10)

    for i in range(linkM):
        for j in range(i):
            net[i][j] = net[j][i]

    netfile = open(data_dir + '/network.txt','w')
    for i in range(linkM):
        for j in range(linkM):
            netfile.write(str('%6.1f' % net[i][j]) + '\t')
        netfile.write('\n')
    netfile.close()

    infofile = open(data_dir + '/connect_info.txt','w')
    for key in connect_info:
        infofile.write(str(connect_info[key]))
        infofile.write('\n')
    infofile.close()

    return (net, connect_info)

def load_net(filepath):
    netfile = open(filepath,'r')
    net = np.array([[0.] * linkM] * linkM)
    i=0
    for line in netfile:
        parseLine = line.split()
        for item in range(len(parseLine)):
            net[i][item] = float(parseLine[item])
        i += 1
    return net

def gen_SEkernel_data(net, connect_info):
    print 'begin generate SE-kernel multi-Gaussian simulated data...'
    kernel = np.array([[0.] * linkM] * linkM)
    for i in range(linkM):
        for j in range(linkM):
            kernel[i][j] = 5 * np.exp(-1 * np.power(net[i][j]/10.,2) / (2 * np.power(length, 2)))

    train = open(speeds_file_pre + '_SEkernel.csv','w')
    writer = csv.writer(train)
    writer.writerow(['d','t','s','speed'])
    MU = np.array([mu] * linkM)
    # speeds = list()
    for d in range(dateD):
        # speeds_d = list()
        for t in range(timeT):
            speeds_t = np.random.multivariate_normal(MU, kernel)

            for s in range(linkM):
                if speeds_t[s] < 0:
                    speeds_t[s] = np.abs(speeds_t[s])
                writer.writerow([d,t,s,speeds_t[s]])

            # speeds_d.append(speeds_t)
        # speeds.append(speeds_d)
    train.close()



def gen_Rkernel_data():
    print 'begin generate random-kernel multi-Gaussian simulated data...'

    # Generate mean vector mu
    mu = np.array([0.] * linkM)
    for i in range(linkM):
        mu[i] = np.random.uniform(30,40)

    # Generate covariance matrix K with noise sigma = 0.5
    A = np.random.rand(linkM,linkM) * 1.0 + 0.5
    K = np.dot(A, A.transpose())
    K = K / len(K[0])

    train = open(speeds_file_pre + '_Rkernel.csv','w')
    writer = csv.writer(train)
    writer.writerow(['d','t','s','speed'])
    for d in range(dateD):
        print '----- begin date: %d...' % d
        for t in range(timeT):
            speeds1 = np.random.multivariate_normal(mu,K)

            for s in range(linkM):
                if speeds1[s] < 0:
                    speeds1[s] = np.abs(speeds1[s])
                writer.writerow([d,t,s,speeds1[s]])
            # speeds1 = list(speeds1)
            # his_speed.append(speeds1)
    train.close()

def gen_NOkernel_data():
    print 'begin generate NO-kernel uni-Gaussian simulated data...'

    # Generate mean vector mu
    mu = np.array([0.] * linkM)
    for i in range(linkM):
        mu[i] = np.random.uniform(30,40)

    # Generate independent covariance ind_k with noise sigma = 0.5
    ind_k = 20.5 + 1.0 * np.random.random(mu.shape)

    train = open(speeds_file_pre + '_Nokernel.csv','w')
    writer = csv.writer(train)
    writer.writerow(['d','t','s','speed'])
    for d in range(dateD):
        for t in range(timeT):
            noise = np.random.normal(0, ind_k)
            speeds1 = mu + noise

            for s in range(linkM):
                writer.writerow([d,t,s,speeds1[s]])
            # speeds1 = list(speeds1)
            # his_speed.append(speeds1)
    train.close()

    # return (his_speed)

def load_speeds(filepath):
    print 'begin loading speed data...'
    file = open(filepath,'rb')
    reader = csv.reader(file)
    all_speed = np.array([[[0.] * linkM] * timeT] * dateD)

    for l in reader:
        # print l
        if len(l)==0 or l[0] == 'd':
            continue
        all_speed[int(l[0])][int(l[1])][int(l[2])] = float(l[3])

    return all_speed

def sep_train(all_speeds, miss_rate, train_rate):
    train_div = int(len(all_speeds) * train_rate)
    miss_num = int(linkM * miss_rate)
    print '-----miss number: %d' % miss_num

    train_sp = defaultdict(lambda:0)
    test_sp = defaultdict(lambda: 0)

    for d in range(len(all_speeds)):
        for t in range(len(all_speeds[d])):
            miss_link = random.sample(range(linkM), miss_num)
            sp = list()
            for s in range(linkM):
                sp.append(all_speeds[d][t][s])
            for l in range(len(miss_link)):
                sp[miss_link[l]] = -1
            if d < train_div:
                train_sp[d,t] = sp
            else:
                test_sp[d,t] = sp

    return train_sp, test_sp

def calc_mean(train_sp):
    lk_sp = defaultdict(lambda :[0.,0])
    for key in train_sp:
        for l in range(len(train_sp[key])):
            if train_sp[key][l] >= 0:
                lk_sp[l][0] += train_sp[key][l]
                lk_sp[l][1] += 1

    ave_sp = list()
    for key in lk_sp:
        ave_sp.append(lk_sp[key][0] / lk_sp[key][1])

    return ave_sp

def fit_GP(all_sp, miss_rate, train_rate):
    # seperate train and test_pyGPs data, unobserved speeds are set as -1
    train_sp, test_sp = sep_train(all_sp, miss_rate, train_rate)

    # calculate historical mean
    his_mean = calc_mean(train_sp)

    # fit and predict
    mape_dates = defaultdict(lambda :0)
    rmse_dates = defaultdict(lambda :0)

    for key in test_sp:

        # X: historical mean, Y: current speed
        miss_links = list()
        miss_true_sp = list()
        miss_mean_sp = list()
        obser_links = list()
        obser_true_sp = list() # Y
        obser_mean_sp = list() # X

        for l in range(len(test_sp[key])):
            if test_sp[key][l] == -1:
                miss_links.append(l)
                miss_true_sp.append(all_sp[int(key[0])][int(key[1])][l])
                miss_mean_sp.append(his_mean[l])
            else:
                obser_links.append(l)
                obser_true_sp.append(all_sp[int(key[0])][int(key[1])][l])
                obser_mean_sp.append(his_mean[l])

        X = np.atleast_2d(obser_mean_sp).T
        Y = np.array(obser_true_sp).T
        gp = GaussianProcess(corr='absolute_exponential')
        gp.fit(X, Y)

        x = np.atleast_2d(miss_mean_sp).T
        predict_sp, MSE = gp.predict(x, eval_MSE=True)
        sigma = np.sqrt(MSE)

        mape = 0.
        rmse = 0.
        for i in range(len(miss_links)):
            rmse += (predict_sp[i] - miss_true_sp[i])**2
            mape += np.abs(predict_sp[i] - miss_true_sp[i]) / miss_true_sp[i]
        rmse = np.sqrt(rmse / len(miss_links))
        mape = mape / len(miss_links)

        mape_dates[key] = mape
        rmse_dates[key] = rmse


    return (mape_dates, rmse_dates)

if __name__ == '__main__':

    net, connect_info = gen_network()
    gen_SEkernel_data(net, connect_info)

    all_sp = load_speeds(speeds_file_pre + '_NOkernel.csv')
    # print all_sp[1]
    miss_rate=list()
    for i in range(1, 10):
        miss_rate.append(float(i / 10.))

    train_rate = 0.8
    mape_file = open(data_dir + '/mape.csv','w')
    mape_writer = csv.writer(mape_file)
    mape_writer.writerow(['miss_rate','d','t','mape'])
    rmse_file = open(data_dir + '/rmse.csv','w')
    rmse_writer = csv.writer(rmse_file)
    rmse_writer.writerow(['miss_rate','d','t','rmse'])
    for i in range(len(miss_rate)):
        print 'miss_rate = %f...' % miss_rate[i]
        mape_dates, rmse_dates = fit_GP(all_sp,miss_rate[i], train_rate)
        mean_mape = 0.
        mean_rmse = 0.
        for key in mape_dates:
            mean_mape += mape_dates[key]
            mean_rmse += rmse_dates[key]

            mape_writer.writerow([miss_rate[i], key[0], key[1], mape_dates[key]])
            rmse_writer.writerow([miss_rate[i], key[0], key[1], rmse_dates[key]])

        mean_mape = mean_mape / len(mape_dates)
        mean_rmse = mean_rmse / len(mape_dates)

        print '----- mape: %f, rmse: %f' % (mean_mape, mean_rmse)
    mape_file.close()
    rmse_file.close()

