'''
Created on 

@author: linlu
'''
import csv
import os
from collections import defaultdict

from pylab import *

import myGP

# miss_ratio = [0.1,0.3,0.5,0.7,0.9]
miss_ratio = [0.5]

# data_dir = '/Users/lulin/baiduyun/intern/GP/GP_simu/simu_code/GP_simu/M_10/'
datafile_path = '/Users/lulin/baiduyun/intern/data-preanalysis/output_analysis_results/records/near/3498140/gp.txt'

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
    print linkM, timeT, dateD
    all_speed = np.array([[[0.] * linkM] * timeT] * dateD)
    for l in speed_dict:
        all_speed[int(l[0][0])][int(l[0][1])][int(l[0][2])] = l[1]

    return all_speed

def sep_data(speeds, group_strategy=1):
    traindata=[]
    testdata=[]

    return (traindata, testdata)


def train(filename, group_strategy, model):

    speeds = load_speeds(filename)
    print speeds.shape
    
    rmse = list()
    mape = list()
    
    mu = list()
    true_speed = list()

    # d * t * s
    (traindata, testdata) = sep_data(speeds, group_strategy)

    if model == 'GP_spatial':
        trainer = myGP.GP_spatial(traindata)
        print '-----train succeed: mu size %d' % len(trainer._mu)

        # test
        for ratio in miss_ratio:
            # randomly miss links with ratio
            num = len(testdata);
            idx = range(num);
            random.shuffle(idx);
            tn = int(np.floor(num*ratio));
            miss_idx = idx[:tn]
            observe_idx = idx[tn:]


            for d in range(len(testdata)):
                for t in range(len(testdata[d])):
                    # test for every time idx
                    result_per_t= trainer.gp(testdata, miss_idx, observe_idx)

    elif model == 'GP_temporal':

    elif model == 'GP':



    print '-----test succeed!'


                   for t in range(len(testdata)):
                    rmse_t += np.power(testdata[t][s1] - (mu_post.A)[0][0], 2)
                    if testdata[t][s1] == 0:
                        zeros += 1
                        continue
                    mape_t += np.absolute(testdata[t][s1]-(mu_post.A)[0][0]) / testdata[t][s1]
                rmse += rmse_t / len(testdata)
                mape += mape_t / (len(testdata)-zeros)

    
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


if __name__ == '__main__':
    setname = './3498140/'

    if os.path.exists(setname) == False:
        os.mkdir(setname)

    filename = datafile_path
    (rmse) = train(filename, setname)
