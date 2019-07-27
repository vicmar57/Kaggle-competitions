# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:51:57 2019

@author: WNP387
"""

import pandas as pd
import timeit
import numpy as np


#https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df



start = timeit.default_timer() #to measure runtime


train = pd.read_csv(r"C:\Users\wnp387\Desktop\New folder\train.csv")
train = reduce_mem_usage(train)
print(len(train))
train1 = train.infer_objects() #convert all data types in DF to numeric
#print(train1.dtypes)

train_labels = train1[['event','crew','experiment','time','seat']] #get labels
train_data = train1.drop(['event','crew','experiment','time','seat'],axis=1) #remove labels from data


from scipy import stats #clear outliers!!!
zscore = np.abs(stats.zscore(train_data)) # clear outliers from every label's DF
train_data = train_data[(zscore < 3).all(axis=1)] # 95% of valid data! - 3 STDs from normal distr.
train_labels = train_labels[(zscore < 3).all(axis=1)] # 95% of valid data! - 3 STDs from normal distr.


train.to_csv('new.csv',index=False)

['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.']


print(len(train_data))

list_df = [train_data,train_labels]

train_fin   = pd.concat(list_df, axis=1)   #last 20% of data for label i

train_fin.to_csv('new.csv',index=False)


labelA = train_fin[train_fin['event'] == 'A']
labelB = train_fin[train_fin['event'] == 'B']
labelC = train_fin[train_fin['event'] == 'C']
labelD = train_fin[train_fin['event'] == 'D']

len_data = len(train_fin)

per_a_in_data = len(labelA)/len_data
per_b_in_data = len(labelB)/len_data
per_c_in_data = len(labelC)/len_data
per_d_in_data = len(labelD)/len_data

A_50k = labelA[0:(int)(50000*per_a_in_data)]
B_50k = labelB[0:(int)(50000*per_b_in_data)]
C_50k = labelC[0:(int)(50000*per_c_in_data)]
D_50k = labelD[0:(int)(50000*per_d_in_data)]

tO_MATLAB = pd.concat([A_50k,B_50k,C_50k,D_50k], axis=0)   #last 20% of data for label i
tO_MATLAB.to_csv('TO_MATLAB.csv',index=False)

one_per_of_data = (int)(0.01*len_data)


stop = timeit.default_timer()
print('\nRunTime:', stop - start, 'sec')














#X = train_data
#y = train_labels
#
## Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
#
#
#from sklearn.preprocessing import StandardScaler
#
#sc = StandardScaler()  
#X_train = sc.fit_transform(X_train)  
#X_test = sc.transform(X_test)  
#
#
#from sklearn.decomposition import PCA
#
#pca = PCA(n_components=1)  
#X_train = pca.fit_transform(X_train)  
#X_test = pca.transform(X_test) 
#
#
#explained_variance = pca.explained_variance_ratio_  
#
#
#
#
#from sklearn.ensemble import RandomForestClassifier
#
#classifier = RandomForestClassifier(max_depth=2, random_state=0)  
#classifier.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred = classifier.predict(X_test) 
#
#
#
#from sklearn.metrics import confusion_matrix  
#from sklearn.metrics import accuracy_score
#
#cm = confusion_matrix(y_test, y_pred)  
#print(cm)  
#print('Accuracy' + accuracy_score(y_test, y_pred)) 

