
import numpy as np
from scipy.optimize import minimize
import time
import random
from bayes_logistic import *

import matplotlib.pyplot as plt
#from statsmodels.tools.tools import add_constant
# This is the blackbox class, bandidts = types
class ContextualMAB:

#!!!! INTERCEPT ==0 !!!!
    # initialization
    def __init__(self):
        # we build two bandits
        self.weights = {}
        self.weights[0] = [0.7, -0.8, -0.3, -2.5]
        self.weights[1] = [0, -0.55, -0.2, -2]
        self.weights[2] = [0.9, 0, -0.15, -0.5]
        self.weights[3] = [1, 0.4, -0.1, -0.25]
        self.weights[4] = [1.1, 0.6, -0.05, -0.1]

    # method for acting on the bandits
    def draw(self, k, x):
        # probability dict
        prob_dict = {}

        # loop for each bandit
        for bandit in self.weights.keys():
            # linear function of external variable
            f_x = self.weights[bandit][0] + self.weights[bandit][1] * x[0] + self.weights[bandit][1] * x[1] + self.weights[bandit][2] * x[2]

            # generate reward with probability given by the logistic
            probability = 1 / (1 + np.exp(-f_x))

            # appending to dict
            prob_dict[bandit] = probability

        # give reward according to probability
        return np.random.choice([0, 1], p=[1 - prob_dict[k], prob_dict[k]]), max(prob_dict.values()) - prob_dict[k], \
        prob_dict[k]
    def prob(self, k, x):
        # probability dict
        prob_dict = {}

        # loop for each bandit
        for bandit in self.weights.keys():
            # linear function of external variable
            f_x = self.weights[bandit][0] + self.weights[bandit][1] * x[0] + self.weights[bandit][1] * x[1] + \
                  self.weights[bandit][2] * x[2]

            # generate reward with probability given by the logistic
            probability = 1 / (1 + np.exp(-f_x))

            # appending to dict
            prob_dict[bandit] = probability

        # give reward according to probability
        return [1 - prob_dict[k], prob_dict[k]]


class OnlineLogisticRegression:

    # initializing
    def __init__(self, lambda_, alpha, n_dim):

        # the only hyperparameter is the deviation on the prior (L2 regularizer)
        self.lambda_ = lambda_
        self.alpha = alpha

        # initializing parameters of the model
        self.n_dim = n_dim,

        self.m = np.zeros(self.n_dim)
        self.q = np.ones(self.n_dim) * self.lambda_

        # initializing weights
        self.w = np.random.normal(self.m, self.alpha * (self.q) ** (-1.0), size=self.n_dim)

        self.w_int = np.concatenate([self.w, np.zeros(1)])
        self.m_int = np.concatenate([self.m, np.zeros(1)])
        print(self.w, self.w_int)


    def simulate(self, weather, congestion, battery) -> float:
        pass

    # the loss function


    def loss(self, w, *args):
        X, y = args
        print(w[-1],w[:-1])
        print(0.5 * (self.q * (w[:-1] - self.m_int[:-1])).dot(w[:-1] - self.m_int[:-1]) + np.sum(
             [np.log(1 + np.exp(w[-1]+ w[:-1].dot(X[j]))) - y[j]*(w[-1]+ w[:-1].dot(X[j])) for j in range(y.shape[0])]))
        return 0.5 * (self.q * (w[:-1] - self.m_int[:-1])).dot(w[:-1] - self.m_int[:-1]) + np.sum(
             [np.log(1 + np.exp(w[-1] + w[:-1].dot(X[j]))) - y[j]*(w[-1]+ w[:-1].dot(X[j])) for j in range(y.shape[0])])

    # the gradient


    def grad(self, w, *args):
        X, y = args
        a1 = self.q * (w[:-1] - self.m) +  np.array(
            [ X[j] / (1. + np.exp( w[-1] + w[:-1].dot(X[j]))) - y[j]*X[j] for j in range(y.shape[0]) ]).sum(axis=0)

        a2 =  np.array( [ 1 / (1. + np.exp( w[-1] + w[:-1].dot(X[j]))) - y[j] for j in range(y.shape[0])]).sum(axis=0)
        return np.concatenate([a1, np.array([a2])])

    # method for sampling weights
    def get_weights(self):
        return np.random.normal(self.m, self.alpha * (self.q) ** (-1.0), size=self.n_dim)

    # fitting method
    def fit(self, X, y):
        intercept = np.ones([X.shape[0], 1])
        X_intercept = np.column_stack((X, intercept))
        """
        :param X: nparray (N,3)
        :param y: np ( N,1)
        :return: float
        """
        # step 1, find w
        self.w_int = minimize(self.loss, self.w_int, args=(X, y), jac=self.grad, method="L-BFGS-B",
                          options={'maxiter': 50, 'disp': False}).x
        self.m = self.w_int[:-1]
        self.m_int = self.w_int

        # step 2, update q
        P = (1 + np.exp(- self.m_int[-1] - X.dot(self.m))) ** (-1)
        self.q = self.q + (P * (1 - P)).dot(X ** 2)

    # probability output method, using weights sample
    def predict_proba(self, X, mode='sample'):
        intercept = np.ones([X.shape[0], 1])
        X_intercept = np.column_stack((X, intercept))
        # adding intercept to X
        #X = add_constant(X)

        # sampling weights after update
        self.w = self.get_weights()

        # using weight depending on mode
        if mode == 'sample':
            w = self.w  # weights are samples of posteriors
        elif mode == 'expected':
            w = self.m  # weights are expected values of posteriors
        else:
            raise Exception('mode not recognized!')

        # calculating probabilities
        proba = 1 / (1 + np.exp( -self.w_int[-1] -1 * X.dot(w)))
        return np.array([1 - proba, proba]).T



if __name__ == "__main__":
    np.random.seed(123)
    cmab3 = ContextualMAB()
    X1 = np.random.choice([0, 1], size=1000)
    X2 = np.random.choice([0, 1], size=1000)
    X3 = np.random.uniform(0, 1, 1000)
    X = np.column_stack((X1, X2, X3))
    y = np.array([cmab3.draw(0, X[i])[0] for i in range(0, len(X1))])
    results = []

    X1_r = np.random.choice([0, 1], size=3)
    X2_r = np.random.choice([0, 1], size=3)
    X3_r = np.random.uniform(0, 1, 3)
    X_r = np.column_stack((X1_r, X2_r, X3_r))
    yprueba = np.array([cmab3.prob(0, X_r[n]) for n in range(0, len(X1_r))])
    print(yprueba)

    for i in range(10, 11):
        b_log = EBLogisticRegression()
        b_log.fit(X, y)
        print(b_log.coef_)
        print((i))

        for _ in range(0, i):
            print(b_log.coef_)
            X1_new = np.random.choice([0, 1], size=1000)
            X2_new = np.random.choice([0, 1], size=1000)
            X3_new = np.random.uniform(0, 1, 1000)
            X_new = np.column_stack((X1_new, X2_new, X3_new))
            y_new = np.array([cmab3.draw(0, X_new[k])[0] for k in range(0, len(X1_new))])
            b_log = b_log.fit(X_new, y_new)

        b = b_log.predict_proba(X_r)
        print(b)
        mse = 0
        for j in range(0, len(yprueba)):
            mse += abs(b[j][0] - yprueba[j][0])
        mse = mse / len(yprueba)
        results.append(mse)

    print((results))
    plt.plot(results)
    plt.show()






    """"
    
    np.random.seed(123)
    # cmab3 is blackbox
    cmab3 = ContextualMAB()

    # These are the 3 variables, if the range is positive, BAD
    X1 = np.random.choice([-1, +1], size=1000)
    X2 = np.random.choice([-1, +1], size=1000)
    X3 = np.random.uniform(-1, 1, 1000)
    X = np_combined_array = np.column_stack((X1, X2, X3))
    x_res = X.reshape(-3, 3)
    # cmb3.draw is to simulate in blacbox
    y = np.array([cmab3.draw(0, X[i])[0] for i in range(0, len(X1))])

    # OLR object
    online_lrn3 = OnlineLogisticRegression(0.5, 1, 3)

    # start = time.time()
    online_lrn3.fit(x_res, y)
    # end = time.time()
    # print(end - start)

    # Now test the capacity of prediction with a sample of 10

    X1 = np.random.choice([-1, 1], size=5)
    X2 = np.random.choice([-1, 1], size=5)
    X3 = np.random.uniform(-1, 1, 5)
    X_test = np_combined_array = np.column_stack((X1, X2, X3))
    x_test = X.reshape(-3, 3)
    yprueba = np.array([cmab3.prob(0, X_test[i]) for i in range(0, len(X1))])

    print(yprueba)

    print(online_lrn3.predict_proba(X_test, "sample"))
    
    
    X1 = np.random.choice([-1, +1], size=20)
    X2 = np.random.choice([-1, +1], size=20)
    X3 = np.random.uniform(-1, 1, 20)
    X = np_combined_array = np.column_stack((X1, X2, X3))
    x_res = X.reshape(-3, 3)
    online_lrn3.fit(x_res, y)
    
    np.random.seed(12345)
    X1 = np.random.choice([0, 1], size=100)
    X2 = np.random.choice([0, 1], size=100)
    X3 = np.random.uniform(0, 1, 100)
    X = np.column_stack((X1, X2, X3))
    yprueba = np.array([cmab3.prob(1, X[i]) for i in range(0, len(X1))])

    b = b_log.predict_proba(X)
    #print(yprueba,b)
    mse= 0
    for i in range(0,len(b)):
        mse = (b[i][0] - yprueba[i][0])**2
    mse = mse/len(b)

    print(mse)


    # instance of our class
    np.random.seed(123)
    #cmab3 is blackbox
    cmab3 = ContextualMAB()

    #These are the 3 variables, if the range is positive, BAD
    X1 = np.random.choice([-1, +1], size=5000)
    X2 = np.random.choice([-1, +1], size=5000)
    X3 = np.random.uniform(-1, 1, 5000)
    X = np_combined_array = np.column_stack((X1, X2, X3))
    # cmb3.draw is to simulate in blacbox
    y = np.array([cmab3.draw(1, X[i])[0] for i in range(0, len(X1))])


    # OLR object
    online_lrn3 = OnlineLogisticRegression(0.5, 1, 3)
    x_res = X.reshape(-3, 3)
    #start = time.time()
    online_lrn3.fit(x_res, y)
    #end = time.time()
    #print(end - start)

    # Now test the capacity of prediction with a sample of 10

    X1 = np.random.choice([-1, 1], size=10)
    X2 = np.random.choice([-1, 1], size=10)
    X3 = np.random.uniform(-1, 1, 10)
    X = np_combined_array = np.column_stack((X1, X2, X3))
    yprueba = np.array([cmab3.prob(1, X[i]) for i in range(0, len(X1))])
    print(yprueba)
    print(online_lrn3.predict_proba(X,"sample"))




 cmab3 = ContextualMAB()
    X1 = np.random.choice([0, 1], size=10)
    X2 = np.random.choice([0, 1], size=10)
    X3 = np.random.uniform(0, 1, 10)
    X = np.column_stack((X1, X2, X3))
    y = np.array([cmab3.draw(0, X[i])[0] for i in range(0, len(X1))])
    results = []

    X1_r = np.random.choice([0, 1], size=1000)
    X2_r = np.random.choice([0, 1], size=1000)
    X3_r = np.random.uniform(0, 1, 1000)
    X_r = np.column_stack((X1_r, X2_r, X3_r))
    yprueba = np.array([cmab3.prob(0, X_r[n]) for n in range(0, len(X1_r))])


    for i in range(0, 20):
        b_log = EBLogisticRegression()
        b_log.fit(X, y)
        print((i))

        for _ in range(0, i):
            X1_new = np.random.choice([0, 1], size=10)
            X2_new = np.random.choice([0, 1], size=10)
            X3_new = np.random.uniform(0, 1, 10)
            X_new = np.column_stack((X1_new, X2_new, X3_new))
            y_new = np.array([cmab3.draw(0, X_new[k])[0] for k in range(0, len(X1_new))])
            b_log = b_log.fit(X_new, y_new)

        b = b_log.predict_proba(X_r)
        mse = 0
        for j in range(0, len(yprueba)):
            mse += abs(b[j][0] - yprueba[j][0])
        mse = mse / len(yprueba)
        results.append(mse)

    print((results))
    plt.plot(results)
    plt.show()
    
    
        def loss(self, w, *args):
        X, y = args
        print(w[-1],w[:-1])
        print(0.5 * (self.q * (w[:-1] - self.m_int[:-1])).dot(w[:-1] - self.m_int[:-1]) - np.sum(
            [np.log(1 + np.exp(-y[j]*w[-1] - y[j] * w[:-1].dot(X[j]))) for j in range(y.shape[0])]))
        return 0.5 * (self.q * (w[:-1] - self.m_int[:-1])).dot(w[:-1] - self.m_int[:-1]) - np.sum(
            [np.log(1 + np.exp(-y[j]*w[-1] - y[j] * w[:-1].dot(X[j]))) for j in range(y.shape[0])])
            
            
                def grad(self, w, *args):
        X, y = args
        a1 = self.q * (w[:-1] - self.m) + (-1) * np.array(
            [y[j] * X[j] / (1. + np.exp( y[j]*w[-1] + y[j] * w[:-1].dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)

        a2 = (-1) * np.array( [ y[j] / (1. + np.exp( y[j]*w[-1] + y[j] * w[:-1].dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)
        #print(a2)
        return np.concatenate([a1, np.array([a2])])

"""""

