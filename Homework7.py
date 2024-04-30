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

#%%

# Impute values for missing data

whole_df = whole(df_labeled, df_test)

# Cabin
cabin = whole_df["Cabin"].str.split("/", expand=True)
cabin.columns = ["Deck", "Room", "Side"]
whole_df = pd.concat([whole_df, cabin], axis=1)
whole_df.drop(columns=["Cabin"], inplace = True)

# Judgement call to drop "Room" column from dataframe due to the risk of overfitting and the existence of better predictors
whole_df.drop(columns=["Room"], inplace = True)

# CryoSleep
numerical = ["RoomService", "FoodCourt", "Spa", "ShoppingMall", "VRDeck"]
vip = ["VIP"]
condition = (whole_df["CryoSleep"] == True)
whole_df.loc[~condition, numerical] = whole_df.loc[~condition, numerical].fillna(0)
whole_df.loc[~condition, vip] = whole_df.loc[~condition, vip].fillna(False)

columns = ["RoomService", "FoodCourt", "Spa", "ShoppingMall", "VRDeck"]
cryo = ["CryoSleep"]
condition = (whole_df[columns].eq(0).all(axis=1))
whole_df.loc[~condition, cryo] = whole_df.loc[~condition, cryo].fillna(True)

# Judgement call to drop "Name" column from dataframe due to it's lack of direct predictive power, the risk of overfitting, and the existence of better predictors
whole_df.drop(columns=["Name"], inplace = True)

# Age, RoomService, FoodCourt, Spa, ShoppingMall, VRDeck, VIP, Destination, Deck, Side, CryoSleep, HomePlanet

impute_vals = { "Age": whole_df["Age"].mean(),
               "RoomService": whole_df["RoomService"].mean(),
               "FoodCourt": whole_df["FoodCourt"].mean(),
               "Spa": whole_df["Spa"].mean(),
               "ShoppingMall": whole_df["ShoppingMall"].mean(),
               "VRDeck": whole_df["VRDeck"].mean(),
               "VIP": whole_df["VIP"].mode()[0],
               "Destination": whole_df["Destination"].mode()[0],
               "Deck": whole_df["Deck"].mode()[0],
               "Side": whole_df["Side"].mode()[0],
               "CryoSleep": whole_df["CryoSleep"].mode()[0],
               "HomePlanet": whole_df["HomePlanet"].mode()[0]}

columns = ["Age", "RoomService", "FoodCourt", "Spa", "ShoppingMall", "VRDeck", "VIP", "Destination", "Deck", "Side", "CryoSleep", "HomePlanet"]
whole_df[columns] = whole_df[columns].fillna(impute_vals)

#%%

# Handle categorical data correctly

# HomePlanet: Earth, Europa, Mars (neither)
whole_df["Earth"] = (whole_df["HomePlanet"] == "Earth").astype("float")
whole_df["Europa"] = (whole_df["HomePlanet"] == "Europa").astype("float")
whole_df.drop(columns=["HomePlanet"], inplace = True)

# Destination: TRAPPIST-1e, PSO J318.5-2, 55 Cancri e (neither)
whole_df["RAPPIST-1e"] = (whole_df["Destination"] == "TRAPPIST-1e").astype("float")
whole_df["PSO J318.5-2"] = (whole_df["Destination"] == "PSO J318.5-2").astype("float")
whole_df.drop(columns=["Destination"], inplace = True)

# Side: P, S (not P)
whole_df["P (Cabin Side)"] = (whole_df["Side"] == "P").astype("float")
whole_df.drop(columns=["Side"], inplace = True)

# Deck: B, F, A, G, E, D, C, T (neither)
whole_df["B (Cabin Deck)"] = (whole_df["Deck"] == "B").astype("float")
whole_df["F (Cabin Deck)"] = (whole_df["Deck"] == "F").astype("float")
whole_df["A (Cabin Deck)"] = (whole_df["Deck"] == "A").astype("float")
whole_df["G (Cabin Deck)"] = (whole_df["Deck"] == "G").astype("float")
whole_df["E (Cabin Deck)"] = (whole_df["Deck"] == "E").astype("float")
whole_df["D (Cabin Deck)"] = (whole_df["Deck"] == "D").astype("float")
whole_df["C (Cabin Deck)"] = (whole_df["Deck"] == "C").astype("float")
whole_df.drop(columns=["Deck"], inplace = True)

# CryoSleep, VIP 
whole_df["CryoSleep"] = whole_df["CryoSleep"].astype("float")
whole_df["VIP"] = whole_df["VIP"].astype("float")

#%%
labeled_len = len(df_labeled)
test_len = len(df_test)

df_labeled, df_test = seperate(whole_df, labeled_len, test_len)
df_test.drop(columns=["Transported"], inplace = True)
df_labeled.drop(columns=["PassengerId"], inplace = True)

# Split the labeled data into training and validation subsets.
df_train, df_val = sk.model_selection.train_test_split(df_labeled, train_size = 0.8)

df_train['Transported'] = df_train['Transported'].astype('category')
df_val['Transported'] = df_val['Transported'].astype('category')

X_train = df_train.loc[:, df_train.columns != "Transported"]
X_val = df_val.loc[:, df_val.columns != "Transported"]

y_train = df_train["Transported"]
y_val = df_val["Transported"]

#%%
  
# Model 1: Decision Tree
tree = sk.tree.DecisionTreeClassifier().fit(X_train, y_train)
y_pred1 = tree.predict(X_val)

acc_val1 = np.mean(y_pred1 == y_val)
print("-----------------------------------------")
print("Accuracy (Decision Tree):", acc_val1)
print("-----------------------------------------")

#%%

# Model 2: Random Forest
clf = sk.ensemble.RandomForestClassifier(n_estimators = 100)
clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_val)

acc_val2 = np.mean(y_pred2 == y_val)
print("-----------------------------------------")
print("Accuracy (Random Forest):", acc_val2)

#%%

# Model 3: Gradient Boosting
gb = XGBClassifier(n_estimators = 200, max_depth = 3, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8)
gb.fit(X_train, y_train)
y_pred3 = gb.predict(X_val)

acc_val3 = np.mean(y_pred3 == y_val)
print("-----------------------------------------")
print("Accuracy (Gradient Boosting):", acc_val3)
print("-----------------------------------------")

#%%

# Kaggle submission
ids = df_test['PassengerId']
df_test.drop(columns=["PassengerId"], inplace = True)

submission_predictions = gb.predict(df_test) == 1

col = {'PassengerId': ids,
        'Transported': submission_predictions}

kaggle_sub = pd.DataFrame(col)
kaggle_sub.to_csv('SpaceshipTitanic_submission.csv', index=False)
