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
#os.chdir('/home/lnr-ai/krisjan/santander')
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
df_xgb_enc = pd.concat([pd.read_csv('data/20190701_xgb_encode.csv',index_col=0),
                        pd.read_csv('data/20190701_xgb_encode_test.csv',index_col=0)])
df = df.merge(df_xgb_enc, left_index=True,right_index=True)
#%%
df_train = df.loc[df.VALIDATION == 0].copy()

#%% check distributions between train and test
#%% this is to check the match between feature distributions in test and train
# i.e. to discard features where the distribution is very different...don't think it's relevant for this dataset
#
#df_test = pd.read_csv('data/test.csv')
#df_test = df_test.set_index('ID_code')
##%% 
#
#from scipy.stats import ks_2samp
#from tqdm import tqdm
#list_p_value =[]
#
#for col in tqdm(df_test.columns):
#    list_p_value.append(ks_2samp(df_test[col] , df_train[col])[1])
#
#Se = pd.Series(list_p_value, index = df_test.columns).sort_values() 
#list_discarded = list(Se[Se < .05].index)
#%% remove outliers?
#from sklearn.neighbors import LocalOutlierFactor
#out_clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=12)
#y_pred = out_clf.fit_predict(df_train)
#X_scores = out_clf.negative_outlier_factor_
#%%
#%% get train_df from dataprep for final train
col_filter = list(df_train.drop(['VALIDATION', 'DEFAULT_FLAG', 'PREV_E_GROUP', 'R_ACC_SUPP_GRD'],axis=1).columns)#+list_discarded
X_train = df_train[col_filter]
y_train = df_train.DEFAULT_FLAG
#%%
import lightgbm as lgb
#import pbar

#%%
myseed = 0
cv_folds = 5
ttu = 12

max_trees = 10000
#%%
#%% objective function
date = str(dt.now().year) + str(dt.now().month)+str(dt.now().day)

def lgb_x_val_auc(param_list):
    
    lr = param_list[0]
    spw = param_list[1]
    mb = int(param_list[2])
    nl = int(param_list[3])
    mcw = int(param_list[4])
    ss = param_list[5]
    csbt = param_list[6]
    alpha = param_list[7]
    mgts = param_list[8]
    mdil = int(param_list[9])
    rl = param_list[10]
    bfreq = int(param_list[11])
    
    lgb_param = {'boosting_type': 'gbdt',
                     'num_leaves': nl,
                     'min_data_in_leaf': mdil,
                     'max_depth': -1,
                     'learning_rate': lr,
                     'min_gain_to_split': mgts,
#                     'n_estimators': num_trees,
                     'max_bin': mb,
                     'objective': 'binary',
                     'min_child_weight': mcw,
                     'subsample': ss,
                     'colsample_bytree': csbt,
                     'reg_alpha': alpha,
                     'reg_lambda': rl,
                     'seed': myseed,
                     'num_threads': ttu,
                     'scale_pos_weight': spw,
                     'bagging_freq':bfreq,
                     'missing': None,
                     'verbose':-1}

    lgbtrain = lgb.Dataset(data = X_train.values, label = y_train.values)
    cvresult = lgb.cv(lgb_param, lgbtrain, nfold=cv_folds, num_boost_round= max_trees,
            metrics='auc', early_stopping_rounds=50, verbose_eval=False,seed=myseed)
    cvresult = pd.DataFrame.from_dict(cvresult, orient='columns', dtype=None)
    cur_res = cvresult.tail(1).values[0][0]
    with open('train_log/lgb_'+date+'.txt', 'a') as f:
        print('-----------', file=f)
        print(list(param_list), file=f)
        print('ntrees: ',cvresult.shape[0], file=f)
        print('auc:',cur_res, file=f)
    return cur_res

#%%
import genetic_algorithm as gen
#%% seed starting individual

lr = .01
spw=1
mb=255
nl =13
mcw=10
ss=0.33
csbt=0.05
alpha=0
mgts=1
mdil=80
rl=1
bfreq=5

first_guess = pd.DataFrame([[lr, spw, mb, nl, mcw, ss, csbt, alpha, mgts, mdil, rl, bfreq]])
    

#%% test
lgb_x_val_auc([lr, spw, mb, nl, mcw, ss, csbt, alpha, mgts, mdil, rl, bfreq])
#%%
list_of_types = ['float', 'float', 'int', 'int', 'int', 'float', 'float','float',
                 'float', 'int', 'float','int']

lower_bounds = [.001, 1, 20, 7, 1, .4, .4, 0, 0, 5, 1,0]
upper_bounds = [.5, 100, 511, 2047, 20, 1, 1, 100, 5, 120, 5,50]

#%%
pop_size = 50
generations = 5
#%% Genetic Search for Parameters

best_gen_params, best_gen_scores = gen.evolve(list_of_types, lower_bounds, upper_bounds, 
                                          pop_size, first_guess, generations, lgb_x_val_auc,
                                          mutation_prob=.05, mutation_str=.2, 
                                          perc_strangers=.05, perc_elites=.1 
                                          )#,old_scores=first_guess_scores, save_gens=True)

#%% results
# first run
# [0.013156492593887134, 5.174045074918895, 306.0, 1899.0, 12.0, 0.9638178867024068, 0.7554975891702108, 
#  13.59002239949685, 4.3293546609116405, 65.0, 3.4143119743501202, 9.0]
# n_trees = 724
# test auc: 0.8166
#-----
#[0.018534055158779598,9.513380418536581,414.0,  417.6600000000001,19.0,0.4269142120003584,
# 0.7862978350438524,10.949163100217227,2.839248368346823,7.0,0.8184855809672315,0.0]
# n_trees = 542
# test auc: 0.8126
lr, spw, mb, nl, mcw, ss, csbt, alpha, mgts, mdil, rl, bfreq = [0.011878857605065265, 10.851273008316747, 287.0, 739.0, 3.0, 0.9494528058727572, 0.9679354320100699, 3.944885445823133, 0.5841200149641762, 7.4, 2.4248606975818556, 15.0]
n_trees = 1515
# 0.8084017756623665 encoding of test is not correct
#%% final training

y_train = df_train.DEFAULT_FLAG
X_train = df_train[col_filter]
                     
lgb_param = {'boosting_type': 'gbdt',
             'num_leaves': int(nl),
             'min_data_in_leaf': int(mdil),
             'max_depth': -1,
             'learning_rate': lr,
             'min_gain_to_split': mgts,
#                     'n_estimators': num_trees,
             'max_bin': int(mb),
             'objective': 'binary',
             'min_child_weight': mcw,
             'subsample': ss,
             'colsample_bytree': csbt,
             'reg_alpha': alpha,
             'reg_lambda': rl,
             'seed': myseed,
             'num_threads': ttu,
             'scale_pos_weight': spw,
             'bagging_freq':int(bfreq),
             'missing': None,
             'verbose':1}

lgbtrain = lgb.Dataset(data = X_train.values, label = y_train.values)                         
estimator = lgb.train(lgb_param, lgbtrain, num_boost_round=n_trees) 
#%%
estimator.save_model('data/20190702_lgb.model')
#%%
from sklearn.metrics import roc_auc_score
#%%
df_test = df.loc[df.VALIDATION == 1].copy()
X_test = df_test[col_filter]
y_test = df_test.DEFAULT_FLAG
y_predicted = estimator.predict(X_test)
roc_auc_score(y_test, y_predicted)
#%%
df_pred = df.loc[df.VALIDATION == 2].copy()
X_pred = df_pred[col_filter]
df_pred['prediction_score'] = estimator.predict(X_pred)

#%%
pd.DataFrame(df_pred['prediction_score']).to_csv('data/20190626_lgb_model.csv')
