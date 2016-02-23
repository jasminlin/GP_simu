__author__ = 'lulin'

import numpy as np
from scipy import *
import csv
from collections import defaultdict
import random
import datetime
from sklearn.gaussian_process import GaussianProcess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pyGPs
from sklearn.cross_validation import cross_val_score, KFold
from numpy.matlib import rand
from scipy.linalg.decomp_svd import orth
from pyGPs.GraphExtensions import graphUtil, nodeKernels
from pyGPs.Validation import valid

grid_edge = 10
road_length = 0.5
linkM = grid_edge * grid_edge # 13169
dateD = 14
timeT = 122

length = float(grid_edge)/ 10# longest distance
sigma = 10.
# mu = np.array([random.uniform(10,50) for i in range(linkM)])
mu = np.array([30. for i in range(linkM)])

data_dir = './GP_simu/M_' + str(grid_edge)
# data_dir = './mean_M_' + str(int(grid_edge))

def gen_grid():
    print 'begin generating grid...'

    grid_row = np.array([[road_length] * grid_edge] * grid_edge)
    grid_colume = np.array([[road_length] * grid_edge] * grid_edge)

    link_num = grid_edge * grid_edge
    dis = np.array([[0.] * link_num] * link_num)
    for i in range(link_num):
        for j in range(link_num):
            i_x = i / grid_edge
            i_y = i % grid_edge
            j_x = j / grid_edge
            j_y = j % grid_edge

            big_x = max(i_x, j_x)
            small_x = min(i_x, j_x)
            big_y = max(i_y, j_y)
            small_y = min(i_y, j_y)

            dis[i][j] = np.sum(grid_row[small_x, small_y: big_y]) + np.sum(grid_colume[small_x:big_x, small_y])
    # print dis[0:30,0:30]

    gridfile = open(data_dir + '/distance_grid.txt','w')
    for i in range(linkM):
        for j in range(linkM):
            gridfile.write(str('%6.1f' % dis[i][j]) + '\t')
        gridfile.write('\n')
    gridfile.close()

    return dis

def load_net(filepath):
    print 'begin loading network...'
    netfile = open(filepath,'r')
    net = np.array([[0.] * linkM] * linkM)
    i=0
    for line in netfile:
        parseLine = line.split()
        for item in range(len(parseLine)):
            net[i][item] = float(parseLine[item])
        i += 1
    return net

def load_kernel(filepath):
    # print 'begin loading kernel...'
    netfile = open(filepath,'r')
    net = np.array([[0.] * linkM] * linkM)
    i=0
    for line in netfile:
        parseLine = line.split()
        for item in range(len(parseLine)):
            net[i][item] = float(parseLine[item])
        i += 1
    return net

def gen_SEkernel_data(net):
    print 'begin generate SE-kernel multi-Gaussian simulated data...'
    kernel = np.array([[0.] * linkM] * linkM)
    for i in range(linkM):
        for j in range(linkM):
            kernel[i][j] = sigma * np.exp(-np.power(net[i][j], 2)/(2*np.power(length, 2)))

    # chara_value, chara_vector = np.linalg.eig(kernel)
    # value = np.abs(min(chara_value))
    # print chara_value
    # kernel = kernel + (value+2.) * np.eye(linkM)

    kefile = open(data_dir + '/kernel.txt','w')
    for i in range(linkM):
        for j in range(linkM):
            kefile.write(str('%4.2f' % kernel[i][j]) + '\t')
        kefile.write('\n')
    kefile.close()

    train = open(data_dir + '/simu_speed_SEkernel.csv','w')
    writer = csv.writer(train)
    writer.writerow(['d','t','s','speed'])

    # speeds = list()
    for d in range(dateD):
        # speeds_d = list()
        for t in range(timeT):
            speeds_t = np.random.multivariate_normal(mu, kernel)

            for s in range(linkM):
                writer.writerow([d,t,s,speeds_t[s]])

            # speeds_d.append(speeds_t)
        # speeds.append(speeds_d)
    train.close()

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

# generate train and test_pyGPs set with missing ratio
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

# calculate the mean based on historical training data
def calc_mean(train_sp):
    lk_sp = defaultdict(lambda :[0.,0])
    for key in train_sp:
        for l in range(len(train_sp[key])):
            if train_sp[key][l] >= 0:
                lk_sp[l][0] += train_sp[key][l]
                lk_sp[l][1] += 1

    ave_sp = defaultdict(lambda :0)
    for key in lk_sp:
        ave_sp[key] = (lk_sp[key][0] / lk_sp[key][1])

    return ave_sp

def calc_errors(y_predict, miss_true_sp):
    mape = 0.
    rmse = 0.
    for i in range(len(y_predict)):
        rmse += (y_predict[i] - miss_true_sp[i])**2
        mape += np.abs(y_predict[i] - miss_true_sp[i]) / miss_true_sp[i]
    rmse = np.sqrt(rmse / len(y_predict))
    mape = mape / len(y_predict)

    return (mape, rmse)

# assume mean=0
def fit_GP_useObser(all_sp, net, miss_rate, train_rate, mode):
    train_sp, test_sp = sep_train(all_sp, miss_rate, train_rate)

    # fit and predict
    mape_dates = defaultdict(lambda :0)
    rmse_dates = defaultdict(lambda :0)

    his_mean = calc_mean(train_sp)

    for key in test_sp:

        # X: current observe road distances, Y: current road speed
        miss_true_sp = list()

        x_train = list()
        y_train = list()
        link_train = list()
        x_test = list()
        link_test = list()

        for l in range(len(test_sp[key])):
            if test_sp[key][l] == -1:
                link_test.append(l)
                miss_true_sp.append(all_sp[int(key[0])][int(key[1])][l])
                x_feature = list()
                for i in range(linkM):
                    x_feature.append(net[i][l])
                x_test.append(x_feature)
                # x_test.append(net[0][l])

            else:
                link_train.append(l)
                x_feature = list()
                for i in range(linkM):
                    x_feature.append(net[i][l])
                x_train.append(x_feature)
                # x_train.append(net[0][l])
                # if mode == 'mean':
                #     y_train.append(all_sp[int(key[0])][int(key[1])][l]-his_mean[l])
                # elif mode == 'mean0':
                y_train.append(all_sp[int(key[0])][int(key[1])][l])

        x_train = np.array(x_train)
        # print x_train.shape
        y_train = np.array(y_train)
        if mode == 'mean':
            for i in range(len(y_train)):
                y_train[i] = y_train[i] - his_mean[link_train[i]]

        m = pyGPs.mean.Const()
        k = pyGPs.cov.RBF(np.log(sigma), np.log(length))
        gp = pyGPs.GPR()
        gp.setPrior(mean=m, kernel=k)
        gp.getPosterior(np.array(x_train), np.array(y_train))
        gp.optimize(np.array(x_train), np.array(y_train))

        predict_results = gp.predict(np.array(x_test))
        y_predict = predict_results[0]
        if mode=='mean':
            for i in range(len(y_predict)):
                y_predict[i] = his_mean[link_test[i]] + y_predict[i]
        # print y_predict

        out = open(data_dir + '/predict_true_speed.txt','a')
        for i in range(len(y_predict)):
            out.write(str(key) + ',' + str(y_predict[i]) + ',' + str(miss_true_sp[i]) + '\n')
        out.close()
        mape_dates[key], rmse_dates[key] = calc_errors(y_predict, miss_true_sp)
    return (mape_dates, rmse_dates)

def fit_GP_useTrueKernel(all_sp, net, miss_rate, train_rate):
    # seperate train and test_pyGPs data, unobserved speeds are set as -1
    # train_sp[d,t] = list(link_id, speed)
    train_sp, test_sp = sep_train(all_sp, miss_rate, train_rate)

    # fit and predict
    mape_dates = defaultdict(lambda :0)
    rmse_dates = defaultdict(lambda :0)

    # mean of test_pyGPs data
    his_mean = calc_mean(test_sp)

    for key in test_sp:

        # X: current observe road distances, Y: current road speed
        miss_true_sp = list()

        x_train = list()
        y_train = list()
        train_list = list()
        train_mean = list()
        x_test = list()
        test_list = list()
        test_mean = list()

        kernel = load_kernel(data_dir + '/kernel.txt')
        # print kernel

        for l in range(len(test_sp[key])):
            if test_sp[key][l] == -1:
                miss_true_sp.append(all_sp[int(key[0])][int(key[1])][l])
                x_test.append(net[0][l])
                test_list.append(l)
                test_mean.append(his_mean[l])
            else:
                x_train.append(net[0][l])
                y_train.append(all_sp[int(key[0])][int(key[1])][l])
                train_list.append(l)
                train_mean.append(his_mean[l])

        # form K
        K = np.array([[0.] * len(train_list)] * len(train_list))
        for i in range(len(train_list)):
            for j in range(len(train_list)):
                K[i][j] = kernel[train_list[i]][train_list[j]]

        # form K*
        K_re = np.array([[0.] * len(train_list)] * len(test_list))
        for i in range(len(test_list)):
            for j in range(len(train_list)):
                K_re[i][j] = kernel[test_list[i]][train_list[j]]

        # form y-mu
        y_mu = (np.matrix(y_train) - np.matrix(train_mean)).T

        # posterior y
        y_predict = np.matrix(test_mean).T + np.matrix(K_re) * np.matrix(K).I * y_mu
        # print y_predict

        mape_dates[key], rmse_dates[key] = calc_errors(y_predict, miss_true_sp)
    return (mape_dates, rmse_dates)

if __name__ == '__main__':
    logfile = open(data_dir + '/log.txt','w')
    start = datetime.datetime.now()

    net = gen_grid()
    end1 = datetime.datetime.now()
    logfile.write('gen_grid() cost: ' + str(end1-start))

    # net = gen_network()
    net = load_net(data_dir + '/distance_grid.txt')
    gen_SEkernel_data(net)
    end2 = datetime.datetime.now()
    logfile.write('gen SEkernel data cost: ' + str(end2 - end1))

    all_sp = load_speeds(data_dir + '/simu_speed_SEkernel.csv')
    # print all_sp[1]
    miss_rate=list()
    for i in range(1, 10):
        miss_rate.append(float(i / 10.))

    # miss_rate = [0.5]
    train_rate = 0.8
    mape_file = open(data_dir + '/mape.csv','w')
    mape_writer = csv.writer(mape_file)
    mape_writer.writerow(['miss_rate','d','t','mape'])
    rmse_file = open(data_dir + '/rmse.csv','w')
    rmse_writer = csv.writer(rmse_file)
    rmse_writer.writerow(['miss_rate','d','t','rmse'])
    error_file = open(data_dir + '/error.csv','w')
    error_writer = csv.writer(error_file)
    error_writer.writerow(['miss ratio','mape','rmse'])
    for i in range(len(miss_rate)):
        print 'miss_rate = %f...' % miss_rate[i]
        end3 = datetime.datetime.now()

        mape_dates, rmse_dates = fit_GP_useObser(all_sp, net, miss_rate[i], train_rate,'mean0')
        mean_mape = 0.
        mean_rmse = 0.
        for key in mape_dates:
            mean_mape += mape_dates[key]
            mean_rmse += rmse_dates[key]

            mape_writer.writerow([miss_rate[i], key[0], key[1], mape_dates[key]])
            rmse_writer.writerow([miss_rate[i], key[0], key[1], rmse_dates[key]])

        mean_mape = mean_mape / len(mape_dates)
        mean_rmse = mean_rmse / len(mape_dates)

        end4 = datetime.datetime.now()
        logfile.write('--- miss ratio %f cost: %s' % (miss_rate[i], str(end4-end3)))
        error_writer.writerow([miss_rate[i], mean_mape[0], mean_rmse[0]])

        print '----- mape: %f, rmse: %f' % (mean_mape, mean_rmse)
    mape_file.close()
    rmse_file.close()
    error_file.close()
    logfile.close()

