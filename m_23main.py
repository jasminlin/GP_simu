'''
Created on 

@author: DELL
'''
import csv

from pylab import *

import m_23train
from GP_simu import simulate_1_10len

train_split = 10

# def read_GPS(filename):
#     csvfile = open(filename,'rb')
#     reader = csv.reader(csvfile)
#     
#     dt = dict()
#     ds = dict()
#     speeds = list()
#     speed_t = list()
#     for t in reader:
#         if t[0] == 't':
#             continue
#         if not t[0] in dt:
#             dt[t[0]] = dict()
#             if len(speed_t) > 0:
#                 speeds.append(speed_t)
#                 speed_t = []
#         if not t[1] in dt[t[0]]:
#             dt[t[0]][t[1]] = 1
#             speed_t.append(float(t[2]))
#     if len(speed_t)>0:
#         speeds.append(speed_t)    
#     print 'read GPS succeed: time slices %d' % (len(speeds))
#     return speeds

def read_RMSE(filename):
    csvfile = open(filename,'rb')
    reader = csv.reader(csvfile)
    
    dpct = dict()
    ds = dict()
    RMSE = list()
    for pct in reader:
        if pct[0] == 'train_pct':
            continue
        if not pct[0] in dpct:
            dpct[pct[0]] = 1
            RMSE.append(pct[1])
    RMSE = np.array(RMSE)
    print 'read RMSE succeed: RMSE size %d' % len(RMSE)
    return RMSE
    
def read_MAPE(filename):
    csvfile = open(filename,'rb')
    reader = csv.reader(csvfile)
    
    dpct = dict()
    ds = dict()
    MAPE = list()
    for pct in reader:
        if pct[0] == 'train_pct':
            continue
        if not pct[0] in dpct:
            dpct[pct[0]] = 1
            MAPE.append(pct[1])
    MAPE = np.array(MAPE)
    print 'read MAPE succeed: RMSE size %d' % len(MAPE)
    return MAPE
    

def train(filename, setname):

    speeds = simulate_1_10len.load_speeds(filename)
    print speeds.shape
    
    rmse = list()
    mape = list()
    z = list()
    
    mu = list()
    true_speed = list()

    divide_step = len(speeds) / float(train_split)
    for i in range(7, train_split-2):
        divide_index = int(divide_step * (i+1))
        traindata = np.array(speeds[:divide_index])
        testdata = np.array(speeds[divide_index:])

        print 'train-test_pyGPs percent %f' % (float(i+1) / train_split)
        print '-----divid_index is: %d' % divide_index

        trainer = m_23train.TSP(traindata)
        print '-----train succeed: mu size %d' % len(trainer._mu)
        (rmse_pct, mape_pct, mu_pct, k_pct, speed_pct) = trainer.gaussian_process(testdata)
        print '-----rmse succeed: rmse size %d' % len(rmse_pct)
        rmse.append(rmse_pct)
        mape.append(mape_pct)
        mu.append(mu_pct)
        true_speed.append(speed_pct)
    
    rmse = np.array(rmse)
    mape = np.array(mape)
    mu = np.array(mu)
    true_speed = np.array(true_speed)
    
    out = open('./' + setname + 'rmse.csv','w')
    writer = csv.writer(out)
    writer.writerow(['train_pct','rmse'])
    for train_p in range(len(rmse)):
        writer.writerow([train_p, rmse[train_p]])
    out.close()
    
    out = open('./' + setname + 'mape.csv','w')
    writer = csv.writer(out)
    writer.writerow(['train_pct','mape'])
    for train_p in range(len(mape)):
        writer.writerow([train_p, mape[train_p]])
    out.close()
    
    out = open('./' + setname + 'pre_mu.csv','w')
    writer = csv.writer(out)
    writer.writerow(['train_pct', 'missing_pct', 's', 'mu'])
    for train_p in range(len(mu)):
        for missing_pct in range(len(mu[train_p])):
            for s in range(len(mu[train_p][missing_pct])):
                writer.writerow([train_p, missing_pct, s, mu[train_p][missing_pct][s]])
    out.close()
    
    out = open('./' + setname + 'true_speed.csv','w')
    writer = csv.writer(out)
    writer.writerow(['train_pct', 'missing_pct', 's', 'speed'])
    for train_p in range(len(mu)):
        for missing_pct in range(len(mu[train_p])):
            for s in range(len(mu[train_p][missing_pct])):
                writer.writerow([train_p, missing_pct, s, \
                                 true_speed[train_p][missing_pct][s]])
    out.close()

    print rmse.shape   
    return (rmse)

def main():
    setname = '122days'
    
    filename = './simu_speeds_simple.csv'
    (rmse) = train(filename, setname)
     
#     (rmse) = read_RMSE('./rmse.csv')
#     rmse = np.array(rmse)

if __name__ == '__main__':
    main()