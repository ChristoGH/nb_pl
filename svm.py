#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:19:35 2019

@author: lnr-ai
"""

#%%
import os
#os.chdir('/Users/krisjan/repos/msft_malware/') #mac
#os.chdir('/home/krisjan/repos/nb/nb_pl/') #linux
os.chdir('/home/lnr-ai/krisjan/nb_pl') # rog
#%%
import pandas as pd
from sklearn import pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import roc_auc_score
import numpy as np
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
kernel_func = Nystroem(n_components=500)
sgd_clf = SGDClassifier(loss='modified_huber',n_jobs=6)
sgd_pipe = pipeline.Pipeline([("feature_map", kernel_func),
                                        ("sgd", sgd_clf)])
sgd_pipe.fit(X_train, y_train, sgd__sample_weight=X_train.index.to_frame().WEIGHTING)
#%%
svc_1 = SVC(gamma='auto')
svc_1.fit(X_train, y_train, sample_weight=X_train.index.to_frame().WEIGHTING)
#%%
X_test = df_data.loc[df.VALIDATION == 1].copy()
y_test = df.loc[df.VALIDATION == 1].DEFAULT_FLAG 

df_out = pd.DataFrame(sgd_pipe.predict_proba(X_test)[:,1],index=X_test.index)
df_out.rename(columns={0:'svm_enc'},inplace=True)
roc_auc_score(y_test, df_out.svm_enc, sample_weight=df_out.index.to_frame().WEIGHTING)
# 0.8248691924450222
df_out.to_csv('data/20190704_sgd_encode_test.csv')
