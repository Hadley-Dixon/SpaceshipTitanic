#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:49:43 2024

@author: hadleydixon
"""

# Link to GitHub repo: https://github.com/Hadley-Dixon/SpaceshipTitanic

#%%

import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection
import sklearn.tree
from sklearn import ensemble
#import sklearn.datasets
#import matplotlib.pyplot as plt
#import tensorflow as tf
#import tensorflow_decision_forests as tfdf
from xgboost import XGBClassifier

#%%

df_labeled = pd.read_csv('/Users/hadleydixon/Desktop/BSDS200/Homework /Homework 7/spaceship-titanic/train.csv')
df_test = pd.read_csv('/Users/hadleydixon/Desktop/BSDS200/Homework /Homework 7/spaceship-titanic/test.csv')

# Function combines train/test dataset to handle missing/categorical data
def whole(df1, df2):
    return pd.concat((df1, df2), ignore_index = True)

# Function seperates train/test dataset after special cases handles
def seperate(df, df1_len, df2_len):
    df1 = df.iloc[:df1_len]
    df2 = df.iloc[df1_len:df1_len + df2_len]
    return df1, df2

whole_df = whole(df_labeled, df_test)

# TODO: Impute values for missing data & handle categorical data correctly

# Cabin
cabin = whole_df["Cabin"].str.split("/", expand=True)
cabin.columns = ["Deck", "Room", "Side"]
whole_df = pd.concat([whole_df, cabin], axis=1)
whole_df.drop(columns=["Cabin"], inplace = True)

# Age
age = {"Age": whole_df["Age"].mean()}
whole_df = whole_df["Age"].fillna(age)


#columns = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
#X_train = df_train[columns]
#X_val = df_val[columns]

#y_train = df_train["Transported"]
#y_val = df_val["Transported"]

#%%

# Split the labeled data into training and validation subsets.
#df_train, df_val = sk.model_selection.train_test_split(df_labeled, train_size = 0.8)

# Model 1: Decision Tree
#tree = sk.tree.DecisionTreeClassifier().fit(X_train, y_train)
#y_pred1 = tree.predict(X_val)

#acc_val1 = np.mean(y_pred1 == y_val)
#print("Accuracy (Decision Tree):", acc_val1)

#%%

# Model 2: Random Forest
#clf = sk.ensemble.RandomForestClassifier(n_estimators = 100)
#clf.fit(X_train, y_train)
#y_pred2 = clf.predict(X_val)

#acc_val2 = np.mean(y_pred2 == y_val)
#print("Accuracy (Random Forest):", acc_val2)

# %%

# Model 3: Gradient Boosting
#gb = XGBClassifier(n_estimators = 200, max_depth = 3, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8)
#gb.fit(X_train, y_train)
#y_pred3 = gb.predict(X_val)

#acc_val3 = np.mean(y_pred3 == y_val)
#print("Accuracy (Gradient Boosting):", acc_val3)

# %%

# TODO: Kaggle submission
# goal: predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

