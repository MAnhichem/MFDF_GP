# -*- coding: utf-8 -*-
"""
Python Mulfidelity Surrogate model from various information sources using Gaussian Processes 
based on GPflow. Light version limited to standard application.

Mehdi Anhichem
University of Liverpool
07/06/2021
"""
#==============================================================================
# IMPORT REQUIRED MODULES
#==============================================================================
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random


import gpflow
import tensorflow as tf
from gpflow.ci_utils import ci_niter
from gpflow.utilities import print_summary

#==============================================================================
# GPFLOW-BASED FUNCTIONS
#==============================================================================

def IntermediateSurrogate(X_training, Y_training, kernel, mean_function = None, name = 'information_source',
                          likelihood_variance = 1e-3, train_likelihood_var = True):
    ''' Function that trains an intermediate surrogate model for an information
        source using Gaussian Processes. 
    '''
    ''' Inputs:
        X_training: list of D arrays with training data inputs (need to be reshaped to (N,D)).
        Y_training: array with training data output (need to be reshaped to (N,1)).
        kernel: GPflow kernel function.
        mean_function: GPflow mean function.
        name: string with the name of the information source.
        likelihood_variance: float initial value of likelihood variance.
        train_likelihood_var: boolean on training likelihood variance.
    '''
    ''' Outputs:
        model: GPflow trained model.
    '''    
    ''' Format training data for the GP regression '''
    if len(X_training) == 1:
        X = X_training[0].reshape(-1,1)
    else:
        if all([len(X_training[k])==len(X_training[k+1]) for k in range(len(X_training)-1)]):
            X = np.column_stack([X_training]).T
    if all([len(X_training[k])==len(Y_training) for k in range(len(X_training))]):
        Y = Y_training.reshape(-1,1)
    ''' Train GP '''
    # Choose a Kernel
    # print_summary(kernel)
    # Construct a model
    model = gpflow.models.GPR(data = (X,Y), kernel = kernel, mean_function = mean_function)
    # Set likelihood variance
    model.likelihood.variance.assign(likelihood_variance)
    gpflow.set_trainable(model.likelihood.variance, train_likelihood_var)
    print_summary(model)
    # Optimize the model parameters
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options = dict(maxiter=100))
    print_summary(model)
    
    return model



def Predict(model, X_prediction):
    ''' 
    Function that uses the model to make some predictions at defined prediction points.
    '''
    ''' Inputs:
        model: GPflow trained model.
        X_prediction: array of the shape (N_prediction, D) made of prediction points.
    '''
    ''' Outputs:
        Y_mean: Tensor of the shape (N_prediction, 1) of the posterior mean distribution.
        Y_std: Tensor of the shape (N_prediction, 1) of the posterior standard deviation distribution.
    ''' 
    Y_mean, Y_var = model.predict_f(X_prediction)
    Y_std = tf.sqrt(Y_var)
    return Y_mean, Y_std
        
def FidelityStd(X_prediction, constant_fidelity):
    ''' Function that constructs a constant fidelity variance associated to
        the uncertainty of an information source.
    '''
    ''' Inputs:
        X_prediction: array of the shape (N_prediction, D) made of prediction points. 
        constant_fidelity: constant float value of the fidelity function over the prediction space. 
    '''
    ''' Outputs:
        Y_fid_std: tensor of the shape (N_prediction, 1) of the fidelity function.
    '''  
    Y_std_fid = constant_fidelity * np.ones((X_prediction.shape[0], 1))
    Y_std_fid = tf.convert_to_tensor(Y_std_fid)

    return Y_std_fid

def MultifidelityFusion(IntermediateSurrogates):
    ''' Function that constructs a multifidelity data fusion model from each intermediate surrogate.
    '''
    ''' Inputs:
        IntermediateSurrogates: list of 3-tensors lists corresponding to each intermediate surrogate
        prediction. The number of 3-arrays lists is the number of information sources.
    '''
    ''' Outputs:
        Y_mean_multi: tensor of the shape (N_prediction, 1) of the multifidelity mean.
        Y_std_multi: tensor of the shape (N_prediction, 1) of the multifidelity standard deviation.
    '''
    M = len(IntermediateSurrogates)
    ''' Construction of the total standard deviation '''
    IntermediateSurrogates_temp = []
    for IS in range(M):
        Y_std_tot_temp = np.sqrt(IntermediateSurrogates[IS][1]**2 + IntermediateSurrogates[IS][2]**2)
        IntermediateSurrogates_temp.append(np.array([IntermediateSurrogates[IS][0],Y_std_tot_temp]))
        
    ''' Construction of the multifidelity standard deviation '''
    sum_inv_var = 0     
    for IS in range(M):
        sum_inv_var = sum_inv_var + (1/(IntermediateSurrogates_temp[IS][1]**2))
    Y_std_multi = np.sqrt(1/(sum_inv_var))
    
    ''' Construction of the multifidelity mean '''    
    sum_mean = 0
    for IS in range(M):
        sum_mean = sum_mean + (IntermediateSurrogates_temp[IS][0]/(IntermediateSurrogates_temp[IS][1]**2))
    Y_mean_multi = (Y_std_multi**2) * sum_mean

    Y_mean_multi = tf.convert_to_tensor(Y_mean_multi)
    Y_std_multi = tf.convert_to_tensor(Y_std_multi)
        
    return Y_mean_multi, Y_std_multi   

#==============================================================================
# UTILITIES FUNCTIONS
#==============================================================================
def dataset_separation(X_t, Y_t, ratio_training_validation):
    ''' Fonction used to split a given dataset into two dataset.
    '''
    ''' Inputs:
        X_t: list of D arrays corresponding to the input space.
        Y_t: array corresponding to the scalar of interest.
        ratio_training_validation: float corresponding to the ratio of the length of the output training data set to the validation data set.
    '''
    ''' Outputs:
        X_training: list of D arrays corresponding to the training input space.
        Y_training: array corresponding to the training scalar of interest.
        X_validation: list of D arrays corresponding to the validation input space.
        Y_validation: array corresponding to the validation scalar of interest.
    '''
    N_training = int(len(Y_t)*ratio_training_validation)
    N_validation = len(Y_t)-N_training
    whole_range = range(len(Y_t))
    shuffled_training_range = random.sample(whole_range, len(whole_range))
    try:
        training_range = shuffled_training_range[0:N_training]
        validation_range = shuffled_training_range[N_training:]
    except len(validation_range)+len(training_range) != len(whole_range):
        print('Error in data separation')
    else:
        D = len(X_t)
        X_training = [[] for i in range(D)]
        X_validation = [[] for i in range(D)]
        Y_training = []
        Y_validation = []
        for n in training_range:
            for d in range(D):
                X_training[d].append(X_t[d][n])
            Y_training.append(Y_t[n])
        X_training = [np.array(X_training[d]) for d in range(D)]
        Y_training = np.array(Y_training)
        for n in validation_range:
            for d in range(D):
                X_validation[d].append(X_t[d][n])
            Y_validation.append(Y_t[n])
        X_validation = [np.array(X_validation[d]) for d in range(D)]
        Y_validation = np.array(Y_validation)
    
    return(X_training, X_validation, Y_training, Y_validation)

def PlotPredictionSpace2D(design_space):
    ''' Function that constructs an uniform set where the gaussian process will be 
    be interfered on. 
    ''' 
    ''' Inputs:
        design_space: list of 2 2-elements lists made from the design interval for each dimension.
    '''
    ''' Outputs:
        X_prediction: array of the shape (N_prediction, 2) used for prediction.
        X_plot1: array formed as a grid of values of dimension 1 for 3D-plot.
        X_plot2: array formed as a grid of values of dimension 2 for 3D-plot.
    '''    
    D = len(design_space)
    try:
        X_plot1, X_plot2 = np.mgrid[design_space[0][0]:design_space[0][1]:8, design_space[1][0]:design_space[1][1]:8]
        X_prediction_plot = np.column_stack([[X_plot1, X_plot2]]).T
        X_prediction_plot = X_prediction_plot.reshape(X_prediction_plot.shape[0]*X_prediction_plot.shape[1], 2)
        return X_prediction_plot, X_plot1, X_plot2
    except D > 2:
        return 'Dimesion of the design space is higher than 2'
        
    
