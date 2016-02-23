'''
Created on 20160203

@author: linlu 
'''

import numpy as np
from scipy import stats
miss_division = 5

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
        self._sigma2 = np.matrix(10 * np.identity(len(traindata[0][0]), float))
        
    def gaussian_process(self, test_sp):

        RMSE = list()
        MAPE = list()
        mu = list()
        k = list()
        speed = list()
        ksscore = list()


        testdata = test_sp[0]
        # Calculate the average speed over all the time slices
        ave_speed = list()
        for s in range(len(testdata[0])):
            ave_speed.append(np.sum(testdata[:,s]) / len(testdata))
        ave_speed = np.array(ave_speed)
        
        # Calculate mu and k for every missing link in every division
        # and rmse for every division
        miss_step = len(testdata[0]) / miss_division # randomly observed
        miss_index = miss_step
        for division in range(miss_division - 1):
            print 'miss_rate = %f begin...' % (1-(division+1)/float(miss_division))
            unobserve1 = miss_index
            unobserve2 = len(testdata[0])
            miss_index += miss_step
            
            mu_s = list()
            k_s = list()
            true_speed = list()
            rmse = 0
            mape = 0

            for s1 in range(unobserve1, unobserve2):
                # mu and k for every missing link
                mu_post = self._mu[s1] + np.dot( np.dot(self._K[:unobserve1,s1].T, \
                                          (self._K[:unobserve1,:unobserve1]\
                                           +self._sigma2[:unobserve1,:unobserve1]).I),\
                                   (ave_speed[:unobserve1]-self._mu[:unobserve1]))
                k_post = self._K[s1][s1] - np.dot( np.dot(self._K[:unobserve1,s1].T, \
                                          (self._K[:unobserve1,:unobserve1]\
                                           +self._sigma2[:unobserve1,:unobserve1]).I),\
                                            self._K[:unobserve1,s1])

                # average rmse for every link over all the time slices
                rmse_t = 0
                mape_t = 0
                zeros = 0
                for t in range(len(testdata)):
                    rmse_t += np.power(testdata[t][s1] - (mu_post.A)[0][0], 2)
                    if testdata[t][s1] == 0:
                        zeros += 1
                        continue
                    mape_t += np.absolute(testdata[t][s1]-(mu_post.A)[0][0]) / testdata[t][s1]
                rmse += rmse_t / len(testdata)
                mape += mape_t / (len(testdata)-zeros)
                
                mu_s.append((mu_post.A)[0][0])
                k_s.append((k_post.A)[0][0])
                true_speed.append(ave_speed[s1])
            # average rmse for every division over all the missing links    
            rmse = np.sqrt(rmse / (unobserve2 - unobserve1))
            mape = mape / (unobserve2 - unobserve1)
            
            RMSE.append(rmse)
            MAPE.append(mape)
            mu.append(mu_s)
            k.append(k_s)
            speed.append(true_speed)

        RMSE = np.array(RMSE)
        MAPE = np.array(MAPE)
        mu = np.array(mu)
        k = np.array(k)
        speed = np.array(speed)
        # RMSE[division], mu[division][s], k[division][s]
        return (RMSE, MAPE, mu, k, speed)
        
        