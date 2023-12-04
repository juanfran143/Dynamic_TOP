
import numpy as np
from scipy.optimize import minimize
import time
import random


# This is the blackbox class, bandidts = types
class ContextualMAB:

#!!!! INTERCEpT ==0 !!!!
    # initialization
    def __init__(self):
        # we build two bandits
        self.weights = {}
        self.weights[0] = [0, -0.55, -0.2, -2]
        self.weights[1] = [0, -0.8, -0.3, -1.5]

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
        self.w = np.random.normal(self.m, self.alpha * self.q ** (-1.0), size=self.n_dim)

    # the loss function
    def loss(self, w, *args):
        x, y = args
        return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum(
            [np.log(1 + np.exp(-y[j] * w.dot(x[j]))) for j in range(y.shape[0])])

    # the gradient
    def grad(self, w, *args):
        x, y = args
        return self.q * (w - self.m) + (-1) * np.array(
            [y[j] * x[j] / (1. + np.exp(y[j] * w.dot(x[j]))) for j in range(y.shape[0])]).sum(axis=0)

    # method for sampling weights
    def get_weights(self):
        return np.random.normal(self.m, self.alpha * (self.q) ** (-1.0), size=self.n_dim)

    # fitting method
    def fit(self, x, y):
        x = x.reshape(-3, 3)
        # step 1, find w
        self.w = minimize(self.loss, self.w, args=(x, y), jac=self.grad, method="L-BFGS-B",
                          options={'maxiter': 20, 'disp': False}).x
        self.m = self.w

        # step 2, update q
        p = (1 + np.exp(1 - x.dot(self.m))) ** (-1)
        self.q = self.q + (p * (1 - p)).dot(x ** 2)

    # probability output method, using weights sample
    def predict_proba(self, x, mode='sample'):

        # adding intercept to x
        # x = add_constant(x)

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
        proba = 1 / (1 + np.exp(-1 * x.dot(w)))
        return np.array([1 - proba, proba]).T


if __name__ == "__main__":

    # instance of our class
    np.random.seed(123)
    #cmab3 is blackbox
    cmab3 = ContextualMAB()

    #These are the 3 variables, if the range is positive, BAD
    x1 = np.random.choice([-1, +1], size=1000)
    x2 = np.random.choice([-1, +1], size=1000)
    x3 = np.random.uniform(-1, 1, 1000)
    x = np_combined_array = np.column_stack((x1, x2, x3))
    # cmb3.draw is to simulate in blacbox
    y = np.array([cmab3.draw(0, x[i])[0] for i in range(0, len(x1))])


    # OLR object
    online_lrn3 = OnlineLogisticRegression(0.5, 1, 3)
    x_res = x.reshape(-3, 3)
    #start = time.time()
    online_lrn3.fit(x_res, y)
    #end = time.time()
    #print(end - start)

    # Now test the capacity of prediction with a sample of 10

    x1 = np.random.choice([-1, 1], size=10)
    x2 = np.random.choice([-1, 1], size=10)
    x3 = np.random.uniform(-1, 1, 10)
    x = np_combined_array = np.column_stack((x1, x2, x3))
    yprueba = np.array([cmab3.prob(0, x[i]) for i in range(0, len(x1))])
    print(yprueba)
    print(online_lrn3.predict_proba(x,"sample"))
