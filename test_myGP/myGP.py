'''
Created on 20160203

@author: linlu 
'''

import numpy as np
import random
from collections import defaultdict
from scipy import stats

class GP_spatial:

    # train: calculate prior mu and K
    def __init__(self, traindata):
        
        # Generate mu from training data
        mu = np.sum(traindata,axis = 0)
        mu = mu / len(traindata)
        self._mu = np.sum(mu,axis=0)
        self._mu = self._mu / len(mu)
        
        # Generate K from training data
        self._K = np.array([[0] * len(traindata[0][0])] * len(traindata[0][0]))
        for d in range(len(traindata)):
            for t in range(len(traindata[d])):
                for s in range(len(traindata[0][0])):
                    for s1 in range(len(traindata[0][0])):
                        self._K[s][s1] += (traindata[d][t][s] - self._mu[s]) * (traindata[d][t][s1] - self._mu[s1])

        self._K = self._K / (len(traindata)*len(traindata[0]))
        
        # Generate noise sigma
        self._sigma2 = np.matrix(5. * np.identity(len(traindata[0][0]), float))

    # test on single time index: randomly miss some links
    def gp(self, testdata, miss_idx, observe_idx):

        results = defaultdict(lambda :0)

        for s1 in miss_idx:
            # posterior mu and k for every missing link
            mu_post = self._mu[s1] + np.dot( np.dot(self._K[observe_idx,s1].T, \
                                      (self._K[observe_idx,observe_idx]\
                                       +self._sigma2[observe_idx,observe_idx]).I),\
                               (testdata[observe_idx]-self._mu[observe_idx]))
            k_post = self._K[s1][s1] - np.dot( np.dot(self._K[observe_idx,s1].T, \
                                      (self._K[observe_idx,observe_idx]\
                                       +self._sigma2[observe_idx,observe_idx]).I),\
                                        self._K[observe_idx,s1])

            results[s1] = (testdata[s1], mu_post.A[0][0], k_post.A[0][0])

        return results
        
class GP_temporal:


class GP:
