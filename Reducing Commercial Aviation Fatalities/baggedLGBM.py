# -- coding: utf-8 --
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import timeit


start = timeit.default_timer() #to measure runtime

train_df = pd.read_csv(r"C:\Users\wnp387\Desktop\train_no_outliers.csv")

test_df = pd.read_csv(r"C:\Users\wnp387\Desktop\test.csv")

stop = timeit.default_timer()
print('\nRunTime:', stop - start, 'sec')


#%%






#lgbm
import lightgbm as lgb
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import timeit


start = timeit.default_timer() #to measure runtime

features_n = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3",
              "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4",
              "eeg_c4" , "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg",
              "r", "gsr"]
train_df['pilot'] = 100 * train_df['seat'] + train_df['crew']
test_df['pilot'] = 100 * test_df['seat'] + test_df['crew']

# apply min/max scalar for each pilot
def normalize_by_pilots(df):
    pilots = df["pilot"].unique()
    for pilot in tqdm(pilots):
        ids = df[df["pilot"] == pilot].index
        scaler = MinMaxScaler()
        df.loc[ids, features_n] = scaler.fit_transform(df.loc[ids, features_n])
        
    return df


train_df = normalize_by_pilots(train_df)
test_df = normalize_by_pilots(test_df)

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=420)
print(f"Training on {train_df.shape[0]} samples.")


features = ["crew"] + features_n #neglecting seat
      
def run_lgb(df_train, df_test):
    # Classes as integers
    dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    try:
        df_train["event"] = df_train["event"].apply(lambda x: dic[x])
        df_test["event"] = df_test["event"].apply(lambda x: dic[x])
    except: 
        pass
    
    params = {"objective" : "multiclass",
              "num_class": 4,
              "metric" : "multi_error",
              "num_leaves" : 30,
              "min_child_weight" : 50,
              "learning_rate" : 0.1,
              "bagging_fraction" : 0.7,
              "feature_fraction" : 0.7,
              "bagging_seed" : 420,
              "verbosity" : -1
             }
    
    lg_train = lgb.Dataset(df_train[features], label=(df_train["event"]))
    lg_test = lgb.Dataset(df_test[features], label=(df_test["event"]))
    model = lgb.train(params, lg_train, 1000, valid_sets=[lg_test], early_stopping_rounds=15, verbose_eval=100)
    
    return model
model = run_lgb(train_df, val_df)

pred_val = model.predict(val_df[features], num_iteration=model.best_iteration)
print("Log loss on validation data :", round(log_loss(np.array(val_df["event"].values), pred_val), 3))



pred_test = model.predict(test_df[features], num_iteration=model.best_iteration)

submission = pd.DataFrame(np.concatenate((np.arange(len(test_df))[:, np.newaxis], pred_test), axis=1), columns=['id', 'A', 'B', 'C', 'D'])
submission['id'] = submission['id'].astype(int)

submission.to_csv("submission.csv", index=False)


stop = timeit.default_timer()
print('\nRunTime:', stop - start, 'sec')

import pyaudio  
import wave  

#define stream chunk   
chunk = 1024  

#open a wav format music  
f = wave.open(r"C:\Users\wnp387\Desktop\chirp_1to200Hz_0.5sec.wav","rb")  
#instantiate PyAudio  
p = pyaudio.PyAudio()  
#open stream  
stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
#read data  
data = f.readframes(chunk)  

#play stream  
while data:  
    stream.write(data)  
    data = f.readframes(chunk)  

#stop stream  
stream.stop_stream()  
stream.close()  

#close PyAudio  
p.terminate()  


#%%



from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.gridspec as gridspec


clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
#clf2 = KNeighborsClassifier(n_neighbors=1)    

from sklearn.decomposition import PCA












bagging1 = BaggingClassifier(base_estimator=clf1, n_estimators=10, max_samples=0.8, max_features=0.8)
#bagging2 = BaggingClassifier(base_estimator=clf2, n_estimators=10, max_samples=0.8, max_features=0.8), 'Bagging K-NN', bagging2


label = ['Decision Tree', 'Bagging Tree']
clf_list = [clf1, bagging1]


dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
dic1 = {'CA':0,'DA':1,'SS':3,'LOFT':4}
train_df["event"] = train_df["event"].apply(lambda x: dic[x])
train_df['experiment'] = train_df['experiment'].apply(lambda x: dic1[x])
test_df['experiment'] = test_df['experiment'].apply(lambda x: dic1[x])
#train_df.drop(['experiment'] , axis=1 ,inplace=True)
#test_df.drop(['experiment'] , axis=1 ,inplace=True)
#%%
y = train_df['event']
train_df.drop(['event'] , axis=1 ,inplace=True) #X


#train_df.drop(['id'], axis=1, inplace=True)

test_id = test_df['id']
test_df.drop(['id'], axis=1, inplace=True)
#%%

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
train_df = sc.fit_transform(train_df)  
X_test = sc.transform(test_df)  


pca = PCA(n_components=1)  
train_df = pca.fit_transform(train_df)  
X_test = pca.transform(X_test) 
explained_variance = pca.explained_variance_ratio_  

#%%

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42)
#i=0
#for clf in clf_list :
              
bagging1.fit(train_df ,y)
pred = bagging1.predict_proba(X_test)

#print("%s: "%(label[i]), accuracy_score(pred, y_test))
#    i=i+1
#    cnf_matrix = confusion_matrix(y_test, pred)
#    sns.heatmap(cnf_matrix)
#%%
sub = pd.DataFrame(pred,columns=['A', 'B', 'C', 'D'])
sub['id'] = test_id
cols = sub.columns.tolist()
sub.to_csv("Bagging_withnothing.csv", index=False)