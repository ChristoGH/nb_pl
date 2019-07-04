#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:51:26 2019

@author: krisjan
"""

import os
#os.chdir('/Users/krisjan/repos/msft_malware/') #mac
os.chdir('/home/krisjan/repos/nb/nb_pl/') #linux
#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
#%%
df_logit_test = pd.read_csv('data/20190703_logit_encode_test.csv',index_col=0)
df_logit_val = pd.read_csv('data/20190703_logit_encode_val.csv',index_col=0)

mm_logit = MinMaxScaler()
mm_logit.fit(pd.concat([df_logit_test,df_logit_val]))

#%%
df_xgb_test = pd.read_csv('data/20190703_xgb_encode_test.csv',index_col=0)
df_xgb_val = pd.read_csv('data/20190703_xgb_encode_val.csv',index_col=0)

mm_xgb = MinMaxScaler()
mm_xgb.fit(pd.concat([df_xgb_test,df_xgb_val]))

#%%
df_scores_test = pd.DataFrame(mm_logit.transform(df_logit_test),index=df_logit_test.index,columns=df_logit_test.columns)
df_scores_test.sort_index(inplace=True)
#%%
roc_auc_score(y_test, )