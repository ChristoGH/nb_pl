#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:08:50 2019

@author: krisjan
"""

#%%
import os
#os.chdir('/Users/krisjan/repos/msft_malware/') #mac
os.chdir('/home/krisjan/repos/nb/nb_pl/') #linux
#%%
import pandas as pd
from pandas import datetime as dt
from sklearn.preprocessing import LabelEncoder
#%%
df= pd.read_hdf('data/pl_model.h5',key='df')
df = df.set_index('ID')
le_prev_e_grp = LabelEncoder()
df['PREV_E_GROUP_enc'] = le_prev_e_grp.fit_transform(df['PREV_E_GROUP'])
le_R_ACC_SUPP_GRD = LabelEncoder()
df['R_ACC_SUPP_GRD_enc'] = le_R_ACC_SUPP_GRD.fit_transform(df['R_ACC_SUPP_GRD'])
#%%
df_train = df.loc[df.VALIDATION == 0].copy()
#%%
import xgboost as xgb
#import pbar
#%%
#%% get train_df from dataprep for final train
col_filter = list(df_train.drop(['VALIDATION', 'DEFAULT_FLAG', 'PREV_E_GROUP', 'R_ACC_SUPP_GRD'],axis=1).columns)#+list_discarded
X_train = df_train[col_filter]
y_train = df_train.DEFAULT_FLAG
#%%
myseed = 7
cv_folds = 5
ttu = 12

max_trees = 1000
#%%
#%% objective function
#%% objective function
date = str(pd.datetime.now().year) + str(pd.datetime.now().month)+str(pd.datetime.now().day)

def xgb_x_val_auc(param_list):

    md = int(param_list[0])
    mcw = int(param_list[1])
    gam = param_list[2]
    ss = param_list[3]
    csbt = param_list[4]
    spw = param_list[5]
    lr = param_list[6]
    ra = param_list[7]
    rl = param_list[8]
    
    xgb_param = {'base_score': 0.5,
                         'booster': 'gbtree',
                         'colsample_bylevel': 1,
                         'colsample_bytree': csbt,
                         'gamma': gam,
                         'eta': lr,
                         'max_depth': md,
                         'min_child_weight': mcw,
                         'missing': None,
                         'tree_method':'hist',
    #                     'n_estimators': max_trees,
                         'nthread': ttu,
                         'objective': 'binary:logistic',
                         'reg_alpha': ra,
                         'reg_lambda': rl,
                         'scale_pos_weight': spw,
                         'random_state': myseed,
                         'subsample': ss,
                         'silent':1}

    xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    cvresult = xgb.cv(xgb_param, xgtrain, nfold=cv_folds, num_boost_round= max_trees,
            metrics='auc', early_stopping_rounds=50, seed=myseed)
    cur_res = cvresult.tail(1).values[0][2]
    with open('train_log/xgb_'+date+'.txt', 'a') as f:
        print('-----------', file=f)
        print(list(param_list), file=f)
        print('ntrees: ',cvresult.shape[0], file=f)
        print(cvresult.tail(1), file=f)
    return cur_res

#%%
import genetic_algorithm as gen
#%% seed starting individual

md = 5
mcw = 1
gam = 0
ss = .8
csbt = .8
spw = 1
lr = .01
ra = 0
rl = 1

first_guess = pd.DataFrame([[md,mcw,gam,ss,csbt,spw,lr,ra,rl]])
    
#%%
list_of_types = ['int', 'int', 'float', 'float','float', 'float', 'float', 'float', 'float']

lower_bounds = [1,1,0,.4,.4,1,.001,0,1]
upper_bounds = [20,20,10,1,1,10,.5,100,10]
pop_size = 50
generations = 5

#%% Genetic Search for Parameters

best_gen_params, best_gen_scores = gen.evolve(list_of_types, lower_bounds, upper_bounds, 
                                          pop_size, first_guess, generations, xgb_x_val_auc,
                                          mutation_prob=.05, mutation_str=.2, 
                                          perc_strangers=.05, perc_elites=.1)

#%% results
#[1.0, 17.0, 6.595721341495928, 0.8365994012639792, 0.9800024451352348, 
# 3.070288372175167, 0.3269462667051828, 38.189157440980615, 6.248247624973247]
#ntrees:  1633
#      train-auc-mean  train-auc-std  test-auc-mean  test-auc-std
#1632        0.915843         0.0004        0.89771      0.001576
# [2.0, 16.0, 5.824753838204659, 0.7863037966394404, 0.5549698410581101, 
# 5.332480165275143, 0.3316496007509376, 34.82269238874623, 3.66051193238583]
# ntrees:  645
# train-auc-mean  train-auc-std  test-auc-mean  test-auc-std
#     0.929277        0.00034       0.892569      0.002264

md,mcw,gam,ss,csbt,spw,lr,ra,rl = [2, 16, 5.824753838204659, 0.7863037966394404, 
                                   0.5549698410581101, 5.332480165275143, 0.3316496007509376, 
                                   34.82269238874623, 3.66051193238583]

n_trees = 645
#%%
xgb_x_val_auc([md,mcw,gam,ss,csbt,spw,lr,ra,rl])
#%% final training

y_train = df_train.target
X_train = df_train[col_filter]
                     
xgb_param = {'base_score': 0.5,
                         'booster': 'gbtree',
                         'colsample_bylevel': 1,
                         'colsample_bytree': csbt,
                         'gamma': gam,
                         'learning_rate': lr,
                         'max_delta_step': 0,
                         'max_depth': md,
                         'min_child_weight': mcw,
                         'missing': None,
    #                     'n_estimators': max_trees,
                         'n_jobs': ttu,
                         'objective': 'binary:logistic',
                         'reg_alpha': ra,
                         'reg_lambda': rl,
                         'scale_pos_weight': spw,
                         'random_state': myseed,
                         'subsample': ss,
                         'silent':1}

xgtrain = xgb.DMatrix(X_train.values, label=y_train.values, silent=True)                         
estimator = xgb.train( xgb_param, xgtrain, verbose_eval=True)

estimator.save_model('data/20190310_xgb.model')

#%%
from sklearn.model_selection import KFold
#%% encoding
xgb_param = {'base_score': 0.5,
                         'booster': 'gbtree',
                         'colsample_bylevel': 1,
                         'colsample_bytree': csbt,
                         'gamma': gam,
                         'learning_rate': lr,
                         'max_delta_step': 0,
                         'max_depth': md,
                         'min_child_weight': mcw,
                         'missing': None,
    #                     'n_estimators': max_trees,
                         'n_jobs': ttu,
                         'objective': 'binary:logistic',
                         'reg_alpha': ra,
                         'reg_lambda': rl,
                         'scale_pos_weight': spw,
                         'random_state': myseed,
                         'subsample': ss,
                         'silent':1}

kf = KFold(n_splits=5,shuffle=True)

for train_index, test_index in kf.split(df_train):
    trainloc = df_train.iloc[train_index,:].index
    testloc = df_train.iloc[test_index,:].index
    xgtrain = xgb.DMatrix(df_train.loc[trainloc,col_filter].values, 
                           df_train.loc[trainloc,'target'].values) 
    estimator = xgb.train( xgb_param, xgtrain, verbose_eval=True)
    df_train.loc[testloc,'xgb_enc'] = estimator.predict(xgb.DMatrix(df_train.loc[testloc,col_filter].values))

pd.DataFrame(df_train.xgb_enc).to_csv('data/xgb_encode.csv')

#%%
df_test = pd.read_csv('data/test.csv')
df_test = df_test.set_index('ID_code')
df_test['target'] = estimator.predict(xgb.DMatrix(df_test.values))
#%%
pd.DataFrame(df_test.target).to_csv('data/20190310_xgb_model.csv')
