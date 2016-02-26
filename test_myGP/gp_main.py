'''
Created on 

@author: linlu

### traindata_mode ###
all
weekday
weekend
same

### model ###
GP_spatial
GP_temporal
GP

'''

from collections import defaultdict
from datetime import date,timedelta,datetime
import random
import csv
import os
import numpy as np
import sys

import myGP

# miss_ratio = [0.1,0.3,0.5,0.7,0.9]
miss_ratio = [0.5]
train_mode_list = ['all', 'weekday', 'same']
gp_model_list = ['GP_spatial','GP_temporal', 'GP']

# data_dir = '/Users/lulin/baiduyun/intern/GP/GP_simu/simu_code/GP_simu/M_10/'
root_dir = '/Users/lulin/baiduyun/intern/data-preanalysis/output_analysis_results/records/near'
data_relative_dir = '/data-gp/'
speed_relative_path = 'gp_impu.txt'
date_relative_path = 'date.txt'

output_dir = './'

def load_speeds(filepath):
    print 'begin loading speed data...'
    file = open(filepath,'rb')
    reader = csv.reader(file)
    speed_dict = defaultdict(lambda : 0)
    for l in reader:
        # print l
        if len(l)==0 or l[0] == 'd':
            continue
        speed_dict[(int(l[0]),int(l[1]),int(l[2]))] = float(l[3])

    speed_dict = sorted(speed_dict.iteritems(), key = lambda d:d[0])

    length = len(speed_dict)
    linkM = speed_dict[length-1][0][2]+1
    timeT = speed_dict[length-1][0][1]+1
    dateD = speed_dict[length-1][0][0]+1
    all_speed = np.array([[[0.] * linkM] * timeT] * dateD)
    for l in speed_dict:
        all_speed[int(l[0][0])][int(l[0][1])][int(l[0][2])] = l[1]

    return (all_speed, dateD, timeT, linkM)

# Function for dividing speed into a traindata-testdata pair
def sep_data(speeds, test_datestr, datestr_list, traindata_mode, window_size):
    # gen datestrs for training
    date_span = datestr_list.index(test_datestr)
    print '----- date_span',date_span
    train_datestr_list = gen_trainingdatewindow(test_datestr, date_span, traindata_mode, window_size)
    # print train_datestr_list

    # trans datestr to date idx
    train_dateidx_list = [datestr_list.index(datestr) for datestr in train_datestr_list]
    # print train_dateidx_list
    # divide data into train and test
    traindata = speeds[train_dateidx_list,:,:]
    testdata = speeds[datestr_list.index(test_datestr),:,:]

    return (traindata, testdata)

# Function for generating date list within window
def gen_trainingdatewindow(test_datestr, date_span, traindata_mode, window_size):
    train_datestr_list = []
    date_test = datetime.strptime(test_datestr, "%Y%m%d")

    jump_step_1 = 1

    # all
    if traindata_mode == train_mode_list[0]:
        jump_step_1 = 1
    # same
    elif traindata_mode == train_mode_list[2]:
        jump_step_1 = 7
    # weekday or weekend
    elif traindata_mode == train_mode_list[1]:
        jump_step_1 = 1

    for n in range(1, date_span, jump_step_1):
        this_day = date_test - timedelta(n)
        if len(train_datestr_list) >= window_size:
            break
        # if this day and test day are not weekday or weekend simultaneously
        if traindata_mode == train_mode_list[1]\
                and (date_test.weekday() < 5) != (this_day.weekday() < 5):
            continue
        train_datestr_list.append(datetime.strftime(this_day,"%Y%m%d"))

    if len(train_datestr_list) < window_size:
        print '! window size is out of dataset, true window: %d' % len(train_datestr_list)
    return train_datestr_list

# train within window and test on a day
def predict_by_GP_daily(traindata, testdata, model, miss_idx, observe_idx):

    # GP: predict speed
    # results[(s,d,t)] = (true_speed, pred_speed, k)
    results = defaultdict(lambda :0)
    mapes = np.array([[0.] * (len(miss_idx) + len(observe_idx))] * len(testdata))
    rmses = np.array([[0.] * (len(miss_idx) + len(observe_idx))] * len(testdata))

    # return resutls = linkN * timeT
    if model == gp_model_list[0]:
        trainer = myGP.GP_spatial(traindata)
        print '----- %s train succeed: mu size %d' % (model, len(trainer._mu))

        # test on single time idx
        for t in range(len(testdata)):
            # test for every time idx
            result_s= trainer.gp(testdata[t], miss_idx, observe_idx)
            for s, r_s in result_s.iteritems():
                results[(t,s)] = r_s
                # exclude links with no records
                if r_s[0] == -1.:
                    continue
                mapes[t][s] = np.absolute(r_s[0] - r_s[1]) / (r_s[0] + 1e-4)
                rmses[t][s] = np.power(r_s[0] - r_s[1], 2)

    elif model == gp_model_list[1]:
        trainer = myGP.GP_temporal(traindata)
        print '----- %s train succeed: mu size %d' % (model, len(trainer._mu))

        #test
        results = trainer.gp(testdata, miss_idx, observe_idx)

    elif model == gp_model_list[2]:
        trainer = myGP.GP(traindata)
        print '----- %s train succeed: mu size %d' % (model, len(trainer._mu))

        #test
        results = trainer.gp(testdata, miss_idx, observe_idx)
        for k_ts, r_s in results.iteritems():
            if r_s[0] == -1.:
                continue
            mapes[k_ts[0]][k_ts[1]] = np.absolute(r_s[0] - r_s[1]) / (r_s[0] + 1e-4)
            rmses[k_ts[0]][k_ts[1]] = np.power(r_s[0] - r_s[1], 2)

    return (results, mapes, rmses)

def ouput_model_eval(result_dir, results, mapes, rmses):

    # average mape and rmse for every link
    mape_by_s = np.sum(mapes, axis=0) / (mapes != 0).sum(0)
    rmse_by_s = np.sqrt(np.sum(rmses, axis=0) / (mapes != 0).sum(0))

    # plot true vs. pred

   # save results to file
    fout_mape = open(result_dir + 'mape_' +
                     str(traindata_mode) + '_' + str(window_size) + '.txt', 'w')
    fout_mape.write(','.join(map(str,range(len(mapes[0])))) + '\n')
    fout_mape.write(','.join(map(str, mape_by_s)))
    fout_mape.close()
    fout_rmse = open(result_dir + 'rmse_' +
                     str(traindata_mode) + '_' + str(window_size) + '.txt', 'w')
    fout_rmse.write(','.join(map(str,range(len(rmses[0])))) + '\n')
    fout_rmse.write(','.join(map(str,rmse_by_s)))
    fout_rmse.close()

    print 'done!'

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print 'wrong parameters!'

    model = sys.argv[1]
    traindata_mode = sys.argv[2]
    window_size = int(sys.argv[3])

    # model = 'GP'
    # traindata_mode = 'weekday'
    # window_size = 15

    dir_list = os.walk(root_dir)
    dir_n = 0
    for root, dirs, files in dir_list:

        # not data folder
        last_namefield = root.strip().split('/')[-1]
        if last_namefield == '':
            last_namefield = root.strip().split('/')[-2]

        if (last_namefield.isdigit() == False) or (last_namefield.startswith('2015')):
            continue

        dir_n += 1
        if dir_n == 1:
            continue

        print last_namefield
        print root

        (speeds, dateD, timeT, linkM) = load_speeds(root + data_relative_dir + speed_relative_path)
        print last_namefield, speeds.shape

        # read 'date.txt' to get datestr & idx info
        fin_date = open(root + data_relative_dir + date_relative_path, 'r')
        # skip idx line
        fin_date.readline()
        datestr_list = fin_date.readline().strip().split(',')

        # different missing ratio
        for ratio in miss_ratio:
            # randomly miss links with ratio
            idx = range(linkM);

            # random.shuffle(idx);
            # here we fix the missing links first

            tn = int(np.floor(linkM*ratio));
            miss_idx = idx[:tn]
            observe_idx = idx[tn:]

            # predict speed daily for the last week in dataset
            for test_datestr in datestr_list[len(datestr_list)-7:len(datestr_list)]:
                print '----- test date: ' + test_datestr
                # traindata: d * t * s; testdata: t*s
                (traindata, testdata) = sep_data(speeds, test_datestr, datestr_list,
                                                 traindata_mode, window_size)
                print '----- train shape: ' + str(traindata.shape)\
                      + '; test shape: ' + str(testdata.shape)
                (results, mapes, rmses) = predict_by_GP_daily(traindata, testdata, model,
                                                              miss_idx, observe_idx)

                # output model evaluation
                result_dir = output_dir + '%s/%.1f/%s/%s/' %\
                                          (str(last_namefield), ratio, test_datestr, model)
                if os.path.exists(result_dir) == False:
                    os.makedirs(result_dir)
                ouput_model_eval(result_dir, results, mapes, rmses)

        # break

