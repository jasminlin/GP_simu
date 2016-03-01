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
        # and deal with imputation
        # for every link
        # linkN
        num_impu_mu = 0
        num_impu_k = 0
        linkN = traindata.shape[2]
        timeT = traindata.shape[1]
        self._mu = np.array([0.] * linkN)
        for s in range(linkN):
            non_imputation_idx = traindata[:,:,s] != -1
            if non_imputation_idx.sum() == 0:
                num_impu_mu += 1
                self._mu[s] = 20.
            else:
                self._mu[s] = np.sum(traindata[non_imputation_idx,s], axis=0) / non_imputation_idx.sum()
        
        # Generate K from training data
        self._K = np.array([[0.] * linkN] * linkN)
        for s in range(linkN):
            for s1 in range(linkN):
                non_imputation_idx = (traindata[:,:,s] != -1) & (traindata[:,:,s1] != -1)
                if non_imputation_idx.sum() == 0:
                    num_impu_k += 1
                    self._K[s][s1] = 5.
                else:
                    k_temp = (traindata[non_imputation_idx,s] - self._mu[s])\
                         * (traindata[non_imputation_idx,s1] - self._mu[s1])
                    self._K[s][s1] = np.sum(k_temp) / non_imputation_idx.sum()
        
        # Generate noise sigma
        self._sigma2 = np.matrix(5. * np.identity(len(traindata[0][0]), float))
        print '---------- imputation num: mu-%d, k-%d' % (num_impu_mu, num_impu_k)


    # test on single time index: randomly miss some links
    def gp(self, testdata, miss_idx, observe_idx):
        # imputate observe links with no records
        non_imputation_idx = testdata != -1
        # !! cannot write as: testdata_impu = testdata
        testdata_impu  = testdata[:]
        testdata_impu[non_imputation_idx] = self._mu[non_imputation_idx]

        result_s = defaultdict(lambda :0)

        for s1 in miss_idx:
            # posterior mu and k for every missing link
            # !!! a block of array is a[b,:][:,b], not a[b,b]!!!
            mu_post = self._mu[s1] + np.dot( np.dot(self._K[observe_idx,s1].T, \
                                      (self._K[observe_idx,:][:,observe_idx]\
                                       +self._sigma2[observe_idx,:][:,observe_idx]).I),\
                               (testdata_impu[observe_idx]-self._mu[observe_idx]))
            k_post = self._K[s1][s1] - np.dot( np.dot(self._K[observe_idx,s1].T, \
                                      (self._K[observe_idx,:][:,observe_idx]\
                                       +self._sigma2[observe_idx,:][:,observe_idx]).I),\
                                        self._K[observe_idx,s1])

            result_s[s1] = (testdata[s1], mu_post.A[0][0], k_post.A[0][0])

        return result_s
        
class GP_temporal:
    # train: calculate prior mu and K
    def __init__(self, traindata):

        # Generate mu from training data
        # and deal with imputation
        # for every link
        linkN = traindata.shape[2]
        timeT = traindata.shape[1]
        num_impu_mu = 0
        num_impu_k = 0
        self._mu = np.array([[0.] * timeT] * linkN)
        for s in range(linkN):
            for t in range(timeT):
                non_imputation_idx = traindata[:,t,s] != -1
                if non_imputation_idx.sum() == 0:
                    num_impu_mu += 1
                    self._mu[s][t] = 20.
                else:
                    self._mu[s][t] = np.sum(traindata[non_imputation_idx,t,s], axis=0) / non_imputation_idx.sum()

        # Generate K from training data
        self._K = np.array([[[0.] * timeT] * timeT] * linkN)
        for s in range(linkN):
            for t1 in range(timeT):
                for t2 in range(timeT):
                    non_imputation_idx = (traindata[:,t1,s] != -1) & (traindata[:,t2,s] != -1)
                    if non_imputation_idx.sum() == 0:
                        num_impu_k += 1
                        self._K[s][t1][t2] = 5.
                    else:
                        k_temp = (traindata[non_imputation_idx,t1,s] - self._mu[s][t1])\
                             * (traindata[non_imputation_idx,t2,s] - self._mu[s][t2])
                        self._K[s][t1][t2] = np.sum(k_temp) / non_imputation_idx.sum()

        # Generate noise sigma
        self._sigma2 = np.array([[[0.] * timeT] * timeT] * linkN)
        for s in range(linkN):
            self._sigma2[s] = np.matrix(5. * np.identity(timeT, float))
        print '---------- imputation num: mu-%d, k-%d' % (num_impu_mu, num_impu_k)


    # test on single time index: randomly miss some links
    def gp(self, testdata, miss_idx, observe_idx):
        # imputate observe links with no records
        testdata_impu  = testdata[:].T
        non_imputation_idx = np.array(testdata_impu != -1)
        testdata_impu[non_imputation_idx] = self._mu[non_imputation_idx]

        result_s = defaultdict(lambda :0)

        for s in range(len(testdata[0])):
            for t in miss_idx:
                # posterior mu and k for every missing link
                # !!! a block of array is a[b,:][:,b], not a[b,b]!!!
                mu_post = self._mu[s][t] + np.dot( np.dot(self._K[s][observe_idx,t].T, \
                                          np.matrix(self._K[s][observe_idx,:][:,observe_idx]\
                                           +self._sigma2[s][observe_idx,:][:,observe_idx]).I),\
                                   (testdata_impu[s][observe_idx]-self._mu[s][observe_idx]))
                k_post = self._K[s][t][t] - np.dot( np.dot(self._K[s][observe_idx,t].T, \
                                          np.matrix(self._K[s][observe_idx,:][:,observe_idx]\
                                           +self._sigma2[s][observe_idx,:][:,observe_idx]).I),\
                                            self._K[s][observe_idx,t])

                result_s[(s,t)] = (testdata[t][s], mu_post.A[0][0], k_post.A[0][0])

        return result_s

class GP:
    # train: calculate prior mu and K
    def __init__(self, traindata):

        # Generate mu from training data
        # and deal with imputation
        # vector [(s,t)]= linkN * t + s
        linkN = traindata.shape[2]
        timeT = traindata.shape[1]
        num_impu_mu = 0
        num_impu_k_s = 0
        num_impu_k_t = 0
        self._mu = np.array([0.] * (linkN * timeT))
        for t in range(timeT):
            for s in range(linkN):
                non_imputation_idx = traindata[:,t,s] != -1
                if non_imputation_idx.sum() == 0:
                    num_impu_mu += 1
                    self._mu[linkN * t + s] = 20.
                else:
                    self._mu[linkN * t + s] = np.sum(traindata[non_imputation_idx,t,s], axis=0) \
                                     / non_imputation_idx.sum()

        # Generate K from training data
        # matrix [(s1,t1),(s2,t2)] = [linkN * t1 + s1, linkN * t2 + s2]
        # wrong...((s1,t1),(s2,t2)) = (timeT * linkN * linkN) * t1 + (linkN * linkN) * t2 + linkN * s1 + s2
        self._K = np.array([[0.] * (linkN * timeT)] * (linkN * timeT))

        # calc k(s1,s2)
        K_s = np.array([[0.] * linkN] * linkN)
        for s1 in range(linkN):
            for s2 in range(linkN):
                K_s_per_t  = np.array([0.] * timeT)
                for t in range(timeT):
                    non_imputation_idx = (traindata[:,t,s1] != -1) & (traindata[:,t,s2] != -1)
                    if non_imputation_idx.sum() == 0:
                        num_impu_k_s += 1
                        K_s_per_t[t] = 5.
                    else:
                        K_s_per_t[t] = np.mean((traindata[non_imputation_idx,t,s1] - self._mu[linkN * t + s1])\
                                    * (traindata[non_imputation_idx, t,s2] - self._mu[linkN * t + s2]))
                K_s[s1][s2] = np.mean(K_s_per_t)

        # calc k(t1,t2)
        K_t = np.array([[0.] * timeT] * timeT)
        for t1 in range(timeT):
            for t2 in range(timeT):
                K_t_per_s = np.array([0.] * linkN)
                for s in range(linkN):
                    non_imputation_idx = (traindata[:,t1,s] != -1) & (traindata[:,t2,s] != -1)
                    if non_imputation_idx.sum() == 0:
                        num_impu_k_t += 1
                        K_t_per_s[s] = 5.
                    else:
                        K_t_per_s[s] = np.mean((traindata[non_imputation_idx,t1,s] - self._mu[linkN * t1 + s])\
                                    * (traindata[non_imputation_idx, t2,s] - self._mu[linkN * t2 + s]))
                K_t[t1][t2] = np.mean(K_t_per_s)

        # k((s1,t1),(s2,t2)) = k(s1,s2)k(t1,t2)
        for t1 in range(timeT):
            for s1 in range(linkN):
                for t2 in range(timeT):
                    for s2 in range(linkN):
                        self._K[linkN * t1 + s1][linkN * t2 + s2] = \
                            K_s[s1][s2] * K_t[t1][t2]

        # Generate noise sigma
        self._sigma2 = np.matrix(5. * np.identity(linkN * timeT, float))
        print '---------- imputation num: mu-%d, k_s-%d, k_t-%d' % (num_impu_mu, num_impu_k_s, num_impu_k_t)

    # test on single time index: randomly miss some links
    # testdata = timeT * linkN
    def gp(self, testdata, miss_idx, observe_idx):
        timeT = len(testdata)
        linkN = len(miss_idx) + len(observe_idx)

        # transform testdata from matrix into vector
        testdata_vector = np.array([0.] * (linkN * timeT))
        for t in range(timeT):
            for s in range(linkN):
                testdata_vector[linkN * t + s] = testdata[t][s]

        # imputate observe links with no records
        non_imputation_idx = testdata_vector != -1
        testdata_vector_impu  = testdata_vector[:]
        testdata_vector_impu[non_imputation_idx] = self._mu[non_imputation_idx]

        result_s = defaultdict(lambda :0)

        for t in range(timeT):
            for s in miss_idx:
                # change oberve_idx and miss_idx to (linkN*t+s)
                observe_idx_flatten = list(linkN * t + np.array(observe_idx))
                miss_idx_flatten = list(linkN * t + np.array(miss_idx))

                # posterior mu and k for every missing link
                # !!! a block of array is a[b,:][:,b], not a[b,b]!!!
                mu_post = self._mu[linkN * t + s] \
                          + np.dot( np.dot(self._K[observe_idx_flatten, linkN * t + s].T, \
                            (self._K[observe_idx_flatten,:][:,observe_idx_flatten]\
                            +self._sigma2[observe_idx_flatten,:][:,observe_idx_flatten]).I),\
                            (testdata_vector_impu[observe_idx_flatten]-self._mu[observe_idx_flatten]))
                k_post = self._K[linkN * t + s][linkN * t + s] \
                         - np.dot( np.dot(self._K[observe_idx_flatten,linkN * t + s].T, \
                            (self._K[observe_idx_flatten,:][:,observe_idx_flatten]\
                            +self._sigma2[observe_idx_flatten,:][:,observe_idx_flatten]).I),\
                            self._K[observe_idx_flatten,linkN * t + s])

                result_s[(t,s)] = (testdata[t][s], mu_post.A[0][0], k_post.A[0][0])

        return result_s
