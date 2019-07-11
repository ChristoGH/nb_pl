#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:51:26 2019

@author: krisjan
"""

import os
#os.chdir('/Users/krisjan/repos/msft_malware/') #mac
#os.chdir('/home/krisjan/repos/nb/nb_pl/') #linux
os.chdir('/home/lnr-ai/krisjan/nb_pl') # rog
#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
#%%
df_lgb_test = pd.read_csv('data/20190703_lgb_logit_stack_model_test.csv')
df_lgb_val = pd.read_csv('data/20190703_lgb_logit_stack_model_val.csv')
df_lgb_test = df_lgb_test.set_index(['ID','WEIGHTING'])
df_lgb_val = df_lgb_val.set_index(['ID','WEIGHTING'])
df_lgb_val.rename(columns={'prediction_score':'lgb_pred'}, inplace=True)

mm_lgb = MinMaxScaler()
mm_lgb.fit(pd.concat([df_lgb_test,df_lgb_val]))

#%%
df_xgb_test = pd.read_csv('data/20190704_xgb_logit_stack_test.csv')
df_xgb_val = pd.read_csv('data/20190704_xgb_logit_stack_val.csv')
df_xgb_test = df_xgb_test.set_index(['ID','WEIGHTING'])
df_xgb_val = df_xgb_val.set_index(['ID','WEIGHTING'])

mm_xgb = MinMaxScaler()
mm_xgb.fit(pd.concat([df_xgb_test,df_xgb_val]))

#%%
df_sgd_test = pd.read_csv('data/20190704_sgd_encode_test.csv')

df_sgd_test = df_sgd_test.set_index(['ID','WEIGHTING'])

mm_sgd = MinMaxScaler()
mm_sgd.fit(df_sgd_test)
#%%
df_scores_test = pd.DataFrame(mm_lgb.transform(df_lgb_test),index=df_lgb_test.index,columns=df_lgb_test.columns)
df_scores_test = df_scores_test.merge(df_xgb_test, left_index=True, right_index=True)
df_scores_test['xgb_enc'] = mm_xgb.transform(pd.DataFrame(df_scores_test['xgb_enc']))
df_scores_test = df_scores_test.merge(df_sgd_test, left_index=True, right_index=True)
df_scores_test['svm_enc'] = mm_sgd.transform(pd.DataFrame(df_scores_test['svm_enc']))
df_scores_test.sort_index(inplace=True)
#%%
df = pd.read_hdf('data/pl_model.h5',key='df')[['ID','VALIDATION','WEIGHTING','DEFAULT_FLAG']]
df=df.set_index(['ID','WEIGHTING'])
y_test = df.loc[df.VALIDATION == 1, 'DEFAULT_FLAG']
y_test.sort_index(inplace=True)
#%%
roc_auc_score(y_test, df_scores_test.mean(axis=1),sample_weight=df_scores_test.index.to_frame().WEIGHTING)
# 0.7808654181519022 not good enough