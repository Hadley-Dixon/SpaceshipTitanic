#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:10:45 2024

@author: hadleydixon
"""

# link to GitHub repo: https://github.com/Hadley-Dixon/SpaceshipTitanic

#import tensorflow as tf
#import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection
import sklearn.datasets
#import matplotlib.pyplot as plt

# split the labeled data into training and validation subsets.
df_labeled = pd.read_csv('/Users/hadleydixon/Desktop/BSDS200/Homework /Homework 7/spaceship-titanic/train.csv')
df_test = pd.read_csv('/Users/hadleydixon/Desktop/BSDS200/Homework /Homework 7/spaceship-titanic/test.csv')
df_train, df_val = sk.model_selection.train_test_split(df_labeled)

# goal: predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.