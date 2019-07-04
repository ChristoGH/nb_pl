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
df.sort_values(['ID','WEIGHTING'],inplace=True)
df.index = range(len(df))
le_prev_e_grp = LabelEncoder()
df['PREV_E_GROUP_enc'] = le_prev_e_grp.fit_transform(df['PREV_E_GROUP'])
le_R_ACC_SUPP_GRD = LabelEncoder()
df['R_ACC_SUPP_GRD_enc'] = le_R_ACC_SUPP_GRD.fit_transform(df['R_ACC_SUPP_GRD'])
df_logit_enc = pd.concat([pd.read_csv('data/20190703_1_logit_encode.csv'),
                        pd.read_csv('data/20190703_1_logit_encode_test.csv'),
                        pd.read_csv('data/20190703_1_logit_encode_val.csv')])
df_logit_enc.sort_values(['ID','WEIGHTING'],inplace=True)
df_logit_enc.index = range(len(df_logit_enc))
df_logit_enc.drop(['ID','WEIGHTING'],axis=1,inplace=True)
df = df.merge(df_logit_enc, left_index=True,right_index=True)
df = df.set_index(['ID','WEIGHTING'])
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
myseed = 0
cv_folds = 5
ttu = 12

max_trees = 5000
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

    xgtrain = xgb.DMatrix(X_train.values, label=y_train.values, weight=X_train.index.to_frame().WEIGHTING.values)
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


md,mcw,gam,ss,csbt,spw,lr,ra,rl = [5, 1, 0.0, 0.8, 0.407583809701451, 
                                   2.023344245301038, 0.012291901733646476, 
                                   14.449175235085729, 9.311629606262882]

n_trees = 1650
# 0.7666 cross val
# 0.7809610579546018 test
#%%
xgb_x_val_auc([md,mcw,gam,ss,csbt,spw,lr,ra,rl])
#%% final training

y_train = df_train.DEFAULT_FLAG
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
#                         'n_estimators': n_trees,
                         'n_jobs': ttu,
                         'objective': 'binary:logistic',
                         'reg_alpha': ra,
                         'reg_lambda': rl,
                         'scale_pos_weight': spw,
                         'random_state': myseed,
                         'subsample': ss,
                         'silent':1}

xgtrain = xgb.DMatrix(X_train.values, label=y_train.values, silent=True, weight=X_train.index.to_frame().WEIGHTING.values)                         
estimator = xgb.train( xgb_param, xgtrain, verbose_eval=True, num_boost_round=n_trees)
#%%
estimator.save_model('data/20190704_xgb_logit_stack.model')

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

kf = KFold(n_splits=5,shuffle=True,random_state=myseed)

for train_index, test_index in kf.split(df_train):
    trainloc = df_train.iloc[train_index,:].index
    testloc = df_train.iloc[test_index,:].index
    xgtrain = xgb.DMatrix(df_train.loc[trainloc,col_filter].values, 
                           df_train.loc[trainloc,'DEFAULT_FLAG'].values, 
                           weight=df_train.loc[trainloc].index.to_frame().WEIGHTING.values) 
    estimator = xgb.train( xgb_param, xgtrain, verbose_eval=True, num_boost_round=n_trees)
    df_train.loc[testloc,'xgb_enc'] = estimator.predict(xgb.DMatrix(df_train.loc[testloc,col_filter].values))

pd.DataFrame(df_train.xgb_enc).to_csv('data/20190701_xgb_encode.csv')

#%%
from sklearn.metrics import roc_auc_score
#%%
df_test = df.loc[df.VALIDATION == 1].copy()
X_test = df_test[col_filter]
y_test = df_test.DEFAULT_FLAG

df_test['xgb_enc'] = estimator.predict(xgb.DMatrix(X_test.values))
roc_auc_score(y_test, df_test.xgb_enc.values, sample_weight=X_test.index.to_frame().WEIGHTING.values)
# 0.8251559597994405

pd.DataFrame(df_test.xgb_enc).to_csv('data/20190704_xgb_logit_stack_test.csv')
#%%
#%%
df_val = df.loc[df.VALIDATION == 2].copy()
X_val = df_val[col_filter]

df_val['xgb_enc'] = estimator.predict(xgb.DMatrix(X_val.values))

pd.DataFrame(df_val.xgb_enc).to_csv('data/20190704_xgb_logit_stack_val.csv')
