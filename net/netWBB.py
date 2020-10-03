# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.special import logsumexp
import numpy as np

from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model

import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
from scipy.stats import norm

import time


class netWBB:
    n_bootstrap_samples = 100     #number of bootstrap sampling
    NNs = []

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
        """


        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary
        print("running")
        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ])
            self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin = 2).T

        
        # TODO: implement WBB network
        input_shape = X_train.shape[1]
        output_shape = y_train_normalized.shape[1]
        max_index = X_train.shape[0]
        print("shape", input_shape, output_shape, max_index)

        lamda = 0.001
        Exp = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
        for m in range (self.n_bootstrap_samples):
            print("Trainning model: ", m)
            weights = [Exp.sample() for _ in range(X_train.shape[0] + 1)]

            # Network with a hidden layer and ReLU activation
            net = torch.nn.Sequential(
            torch.nn.Linear(input_shape, n_hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[0], output_shape),
            )
            
            
            BATCH_SIZE = 64
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

            # Minimizing the negative log-likelihood of our data with respect to θ is equivalent to 
            # minimizing the mean squared error between the observed y and our prediction thereof
            loss_func = torch.nn.MSELoss(reduction = 'none')
            torch_dataset = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_normalized))

            loader = Data.DataLoader(
            dataset=torch_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, num_workers=2,)

            start_time = time.time()
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                loss = 0

                index1 = 0;
                for step, (batch_x, batch_y) in enumerate(loader):  
                    
                    index2 = (step+1)*BATCH_SIZE   # set index of weight array to multiply with loss: weight*loss_func
                    index2 = min(index2,max_index) 
                    b_x = Variable(batch_x)
                    b_y = Variable(batch_y)               
                    prediction = net(b_x.float())    
                    
                    # print("weight ",(torch.FloatTensor(weights[index1:index2]).shape))
                    # print("loss ", loss_func(prediction, b_y.float()).shape)
                    loss += (torch.FloatTensor(weights[index1:index2]).unsqueeze(1) * loss_func(prediction, b_y.float())).sum()
                    # print((torch.FloatTensor(weights[index1:index2]).unsqueeze(1) * loss_func(prediction, b_y.float())).shape)
                    index1 = index2   

                # #add L1 regularization
                # l1 = 0
                # for p in net.parameters():
                #     l1 = l1 + p.abs().sum()       
                # loss += weights[-1] * lamda * l1

                #add L2 regularization
                l2 = 0
                for p in net.parameters():
                    l2 = l2 + 0.5 * (p ** 2).sum()      
                loss += (weights[-1] * lamda * l2).sum()
                
                loss.backward()         
                optimizer.step()        
                # print("EPOCH: ", epoch, " LOSS: ", loss)
            print("BATCH_SIZE: ", BATCH_SIZE, "time execution for a neural net: ", time.time() - start_time )
            self.NNs.append(net)


    def predict(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            
    
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin = 2)
        y_test = np.array(y_test, ndmin = 2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        NN_pred = np.array([self.NNs[m](torch.from_numpy(X_test).float()).detach().numpy() for m in range(self.n_bootstrap_samples)])
        NN_pred = NN_pred * self.std_y_train + self.mean_y_train

        standard_pred = NN_pred[0]
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5

        # MC_error
        MC_pred = np.mean(NN_pred, axis = 0)
        rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5

        variance_ll = 0.1
        tau = 1. / variance_ll
        # T = y_test.shape[0]
        T = self.n_bootstrap_samples
        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * tau * (y_test[None] - NN_pred)**2., 0) - np.log(T) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
        test_ll = np.mean(ll)

        # We are done!
        return rmse_standard_pred, rmse, test_ll
