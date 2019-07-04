#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:25:56 2018

@author: krisjan
"""
#%%
import os
#os.chdir('/Users/krisjan/repos/msft_malware/') #mac
os.chdir('/home/krisjan/repos/nb/nb_pl/') #linux

#%%
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from pandas import datetime as dt
from sklearn.preprocessing import LabelEncoder, RobustScaler
#%%
df= pd.read_hdf('data/pl_model.h5',key='df')
df = df.set_index(['ID','WEIGHTING'])
le_prev_e_grp = LabelEncoder()
df['PREV_E_GROUP_enc'] = le_prev_e_grp.fit_transform(df['PREV_E_GROUP'])
le_R_ACC_SUPP_GRD = LabelEncoder()
df['R_ACC_SUPP_GRD_enc'] = le_R_ACC_SUPP_GRD.fit_transform(df['R_ACC_SUPP_GRD'])
rs = RobustScaler()
df_data = df.drop(['VALIDATION', 'DEFAULT_FLAG', 'PREV_E_GROUP', 'R_ACC_SUPP_GRD'],axis=1)
df_data.fillna(-1,inplace=True)
df_data = pd.DataFrame(rs.fit_transform(df_data), index=df_data.index, columns=df_data.columns)
#%%
X_train = df_data.loc[df.VALIDATION == 0].copy()
y_train = df.loc[df.VALIDATION == 0].DEFAULT_FLAG
#%%
myseed = 0
cv_folds = 5
ttu = 12
first_guess = None #pd.DataFrame(final_var_bits).T
#%% VARIABLE SELECTION
#%% set up variables for genetic search
nr_cols = len(X_train.columns)
list_of_types = ['int'] * nr_cols
lower_bounds = [0] * nr_cols
upper_bounds = [2] * nr_cols
pop_size = 50
generations = 5

#%% define eval function
def logit_x_val_auc(var_bit_list):
    var_index = [i for i, e in enumerate(var_bit_list) if e == 1]
    clf = LogisticRegressionCV(cv=cv_folds, random_state=myseed, n_jobs = ttu, 
                               scoring = 'roc_auc').fit(X_train.iloc[:,var_index], y_train,
                                                  sample_weight=X_train.index.to_frame().WEIGHTING)
    auc_score = clf.score(X_train.iloc[:,var_index], y_train)
    print(auc_score)
    return auc_score

#%%
import genetic_algorithm as gen

#%%
best_gen_params, best_gen_scores = gen.evolve(list_of_types, lower_bounds, upper_bounds, 
                                          pop_size, first_guess, generations, logit_x_val_auc,
                                          mutation_prob=.05, mutation_str=.2, 
                                          perc_strangers=.05, perc_elites=.1)    

#%%  
#['CLOSEDGOOD',
# 'MONTHS_OLDEST_PP',
# 'OPEN_PL_TTL_BTB',
# 'OTC_PL_BOND_LAST_6M',
# 'PL_OP_LAST_24M',
# 'E_365_DAYS',
# 'NLR_W_PP_STATUS',
# 'CPA_P_3_PLUS_3M',
# 'CPA_NUM_0_A_12M',
# 'CPA_NUM_S_24M',
# 'BON_S_AVG_TOT',
# 'MEDIAN_TOT_S_GR_L2Y',
# 'AVG_PD_CUR_EMPL_L12M',
# 'LT_100_L3M',
# 'LT_500_L3M',
# 'DAYS_E_L3M',
# 'B_TO_S',
# 'NORM_SEC_LOW_S_L12M',
# 'NORM_AVG_D_AVL_L3M',
# 'ESCORE',
# 'NORM_PREV_NS',
# 'R_ACC_SUPP_GRD_enc']
final_var_bits = best_gen_params.loc[best_gen_scores.idxmax()]
final_var_index = [i for i, e in enumerate(final_var_bits) if e == 1]
col_filter = list(X_train.iloc[:,final_var_index].columns)

#%% BOOTSTRAPPING WITH THRESHOLD OPTIMIZATION
df_train = X_train
df_train_target = y_train

#%% bootstrapped predictor
def sigmoid(x):
    y = np.exp(x)/(1 + np.exp(x))
    return y
def bootstrap_logit(var_list):
    return sigmoid(np.dot(var_list, mean_coefs) + mean_intercept)    
#%%
from sklearn.model_selection import KFold
#%% set number of iterations
bt_strp_it = 150

#%% 
kf = KFold(n_splits=5,shuffle=True,random_state=myseed)

for train_index, test_index in kf.split(df_train):
    trainloc = df_train.iloc[train_index,:].index
    testloc = df_train.iloc[test_index,:].index
    
    for i in range(bt_strp_it):
        X_train, X_test, y_train, y_test = train_test_split(df_train.loc[trainloc, col_filter], 
                                                            df_train_target.loc[trainloc], test_size = .2)
    
        clf = LogisticRegression(random_state=myseed, solver='lbfgs', 
                                 n_jobs=ttu).fit(X_train, y_train, sample_weight=X_train.index.to_frame().WEIGHTING)
    
        predictions = clf.predict_proba(X_test)[:,1]
                
        if i == 0: 
            df_coefs = clf.coef_
            df_intercept = clf.intercept_
            df_auc = [roc_auc_score(y_test, predictions,sample_weight=X_test.index.to_frame().WEIGHTING)]
            
        else:
            df_coefs = np.vstack((df_coefs, clf.coef_))
            df_intercept = np.hstack((df_intercept, clf.intercept_))
            df_auc.append(roc_auc_score(y_test, predictions, sample_weight=X_test.index.to_frame().WEIGHTING))
    
        print('iteration:',i,',auc:',roc_auc_score(y_test, predictions, sample_weight=X_test.index.to_frame().WEIGHTING))
            
    df_coefs = pd.DataFrame(df_coefs, columns=X_train.columns)
    
    mean_coefs = df_coefs.mean()
    mean_intercept = np.mean(df_intercept)
    
    mean_auc = np.mean(df_auc)
    
    df_train.loc[testloc,'logit_enc'] = df_train.loc[testloc,col_filter].apply(bootstrap_logit, axis=1)
    print(roc_auc_score(df_train_target.loc[testloc], df_train.loc[testloc,'logit_enc'], sample_weight=df_train.loc[testloc].index.to_frame().WEIGHTING))
#%%
pd.DataFrame(df_train['logit_enc']).to_csv('data/20190703_1_logit_encode.csv')
#%% encode test
for i in range(bt_strp_it):
    X_train, X_test, y_train, y_test = train_test_split(df_train[col_filter], 
                                                        df_train_target, test_size = .2)

    clf = LogisticRegression(random_state=myseed, solver='lbfgs', 
                             n_jobs=ttu).fit(X_train, y_train, sample_weight=X_train.index.to_frame().WEIGHTING)

    predictions = clf.predict_proba(X_test)[:,1]
            
    if i == 0: 
        df_coefs = clf.coef_
        df_intercept = clf.intercept_
        df_auc = [roc_auc_score(y_test, predictions, sample_weight=X_test.index.to_frame().WEIGHTING)]
        
    else:
        df_coefs = np.vstack((df_coefs, clf.coef_))
        df_intercept = np.hstack((df_intercept, clf.intercept_))
        df_auc.append(roc_auc_score(y_test, predictions, sample_weight=X_test.index.to_frame().WEIGHTING))

    print('iteration:',i,',auc:',roc_auc_score(y_test, predictions, sample_weight=X_test.index.to_frame().WEIGHTING))
        
df_coefs = pd.DataFrame(df_coefs, columns=X_train.columns)

mean_coefs = df_coefs.mean()
mean_intercept = np.mean(df_intercept)

mean_auc = np.mean(df_auc)
#%%
#mean_coefs
#CLOSEDGOOD             -0.191186
#MONTHS_OLDEST_PP       -0.300551
#OPEN_PL_TTL_BTB         0.137284
#OTC_PL_BOND_LAST_6M     0.146522
#PL_OP_LAST_24M          0.055801
#E_365_DAYS              0.180543
#NLR_W_PP_STATUS         0.122140
#CPA_P_3_PLUS_3M         0.176290
#CPA_NUM_0_A_12M        -0.162381
#CPA_NUM_S_24M          -0.231133
#BON_S_AVG_TOT          -0.259799
#MEDIAN_TOT_S_GR_L2Y    -0.121423
#AVG_PD_CUR_EMPL_L12M    0.033585
#LT_100_L3M              0.813155
#LT_500_L3M             -0.437640
#DAYS_E_L3M              0.037973
#B_TO_S                  0.086512
#NORM_SEC_LOW_S_L12M    -0.124982
#NORM_AVG_D_AVL_L3M     -0.016394
#ESCORE                 -0.062382
#NORM_PREV_NS           -0.115108
#R_ACC_SUPP_GRD_enc     -0.020371
#mean_intercept = -2.5064467274709736
#%%
X_test = df_data.loc[df.VALIDATION == 1,col_filter].copy()
y_test = df.loc[df.VALIDATION == 1].DEFAULT_FLAG 

#%% TEST On 1 
#single prediction
bootstrap_logit(X_test.iloc[0,:])
#%% multi-test

df_out = pd.DataFrame(X_test.apply(bootstrap_logit, axis=1))
df_out.rename(columns={0:'logit_enc'},inplace=True)
roc_auc_score(y_test, df_out.logit_enc, sample_weight=df_out.index.to_frame().WEIGHTING)
# 0.8248691924450222
df_out.to_csv('data/20190703_1_logit_encode_test.csv')
#%%
X_val = df_data.loc[df.VALIDATION == 2,col_filter].copy()
y_val = df.loc[df.VALIDATION == 2].DEFAULT_FLAG 

df_val = pd.DataFrame(X_val.apply(bootstrap_logit, axis=1))
df_val.rename(columns={0:'logit_enc'},inplace=True)
df_val.to_csv('data/20190703_1_logit_encode_val.csv')
