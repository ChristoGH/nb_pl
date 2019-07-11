#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 18:40:08 2019

@author: krisjan
"""

#%%
import os
os.chdir('/home/krisjan/repos/nb/nb_pl/') #amd
#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
#from sklearn.externals import joblib
#%%
df= pd.read_hdf('data/pl_model.h5',key='df')
df = df.set_index(['ID','WEIGHTING'])
le_prev_e_grp = LabelEncoder()
df['PREV_E_GROUP_enc'] = le_prev_e_grp.fit_transform(df['PREV_E_GROUP'])
le_R_ACC_SUPP_GRD = LabelEncoder()
df['R_ACC_SUPP_GRD_enc'] = le_R_ACC_SUPP_GRD.fit_transform(df['R_ACC_SUPP_GRD'])
df_data = df.drop(['VALIDATION', 'DEFAULT_FLAG', 'PREV_E_GROUP', 'R_ACC_SUPP_GRD'],axis=1)
df_data.fillna(-1,inplace=True)
rs = RobustScaler()
df_data = pd.DataFrame(rs.fit_transform(df_data), index=df_data.index, columns=df_data.columns)

#%%
myseed = 888
df_train = df_data.loc[df.VALIDATION == 0].copy()
target_train = df.loc[df.VALIDATION == 0].DEFAULT_FLAG

nr_cols = len(df_train.columns)

#%%
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
class roc_callback(Callback):
    def __init__(self,training_data,validation_data,weights):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.tr_weight = weights[0]
        self.val_weight = weights[1]


    def on_train_begin(self, logs={}):
        self.auc = []
        self.auc_val = []
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred, sample_weight=self.tr_weight)
        self.auc.append(roc)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val, sample_weight=self.val_weight)
        self.auc_val.append(roc_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
#%%
def dl_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(nr_cols*8, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(.35))
    model.add(tf.keras.layers.Dense(nr_cols*4, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(.25))
    model.add(tf.keras.layers.Dense(nr_cols, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(.05))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer = tf.train.AdamOptimizer(.001), loss = 'binary_crossentropy')
    return model 

#%%
X_train, X_test, y_train, y_test = train_test_split(df_train,target_train,test_size=0.2)

#%%
roc_history = roc_callback(training_data=(X_train,y_train),
                           validation_data=(X_test,y_test),
                           weights = (X_train.index.to_frame().WEIGHTING,
                                      X_test.index.to_frame().WEIGHTING))
model = dl_model()
history = model.fit(X_train.values, y_train.values, batch_size=512, epochs = 30, 
                    sample_weight=X_train.index.to_frame().WEIGHTING.values,
                    validation_data=(X_test,y_test),callbacks=[roc_history]) #,callbacks=[roc_history]
#%%
#%%
cumsum_vec = np.cumsum(np.insert(roc_history.auc_val, 0, 0))
#ma_vec = (cumsum_vec[5:] - cumsum_vec[:-5]) / 5
# summarize history for accuracy

plt.figure(figsize=(8, 6))
plt.plot(roc_history.auc)
plt.plot(roc_history.auc_val)
#plt.plot(ma_vec)
plt.title('model auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()


#%% train on full
model = dl_model()
model.fit(df_train[train_cols].values, target_train.values, batch_size=512, epochs = 30, 
          sample_weight=df_train.index.to_frame().WEIGHTING.values)
# save
yaml_obj = model.to_yaml()
with open("dl_20190710.yaml", "w") as yaml_file:
    yaml_file.write(yaml_obj)
# serialize weights to HDF5
model.save_weights("dl_20190710.h5")
#%%predict on test
df_test = df_data.loc[df.VALIDATION == 1].copy()
target_test = df.loc[df.VALIDATION == 1].DEFAULT_FLAG
df_test['dl_enc'] = model.predict(df_test.values)
roc_auc_score(target_test, df_test.target.values, sample_weight=df_test.index.to_frame().WEIGHTING.values)

pd.DataFrame(df_test.dl_enc).to_csv('data/201710_dl_encode_test.csv')
#%%predict on val
df_val = df_data.loc[df.VALIDATION == 2].copy()

df_val['dl_enc'] = model.predict(df_val.values)

pd.DataFrame(df_val.dl_enc).to_csv('data/201710_dl_encode_val.csv')
#%% dl encoding
from sklearn.model_selection import KFold
#%% encoding
train_cols=list(df_train.columns)
kf = KFold(n_splits=5,shuffle=True)

for train_index, test_index in kf.split(df_train):
    trainloc = df_train.iloc[train_index,:].index
    testloc = df_train.iloc[test_index,:].index
    model = dl_model()
    model.fit(df_train.loc[trainloc,train_cols].values, target_train.loc[trainloc].values, 
              sample_weight=df_train.loc[trainloc].index.to_frame().WEIGHTING.values,
              batch_size=512, epochs = 30)
    df_train.loc[testloc,'dl_enc'] = model.predict(df_train.loc[testloc,train_cols].values)

pd.DataFrame(df_train.dl_enc).to_csv('data/201710_dl_encode_train.csv')

