
import pandas as pd
import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


##https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
#def reduce_mem_usage(df):
#    """ iterate through all the columns of a dataframe and modify the data type
#        to reduce memory usage.        
#    """
#    start_mem = df.memory_usage().sum() / 1024**2
#    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
#    
#    for col in df.columns:
#        col_type = df[col].dtype
#        
#        if col_type != object:
#            c_min = df[col].min()
#            c_max = df[col].max()
#            if str(col_type)[:3] == 'int':
#                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                    df[col] = df[col].astype(np.int8)
#                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                    df[col] = df[col].astype(np.int16)
#                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                    df[col] = df[col].astype(np.int32)
#                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                    df[col] = df[col].astype(np.int64)  
#            else:
#                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                    df[col] = df[col].astype(np.float16)
#                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                    df[col] = df[col].astype(np.float32)
#                else:
#                    df[col] = df[col].astype(np.float64)
#        else:
#            df[col] = df[col].astype('category')
#
#    end_mem = df.memory_usage().sum() / 1024**2
#    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
#    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
#    
#    return df










#%%
import pandas as pd

#path = r"C:\Users\wnp387\Desktop\train_no_outliers' 
#allFolderPaths = glob.glob(path + "/input/")
#print(allFolderPaths) # print all files with a number in current dir.

start = timeit.default_timer() #to measure runtime

train_csv = pd.read_csv(r"C:\Users\wnp387\Desktop\train_no_outliers.csv")

test_csv = pd.read_csv(r"C:\Users\wnp387\Desktop\test.csv")

stop = timeit.default_timer()
print('\nRunTime:', stop - start, 'sec')

#%%
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingClassifier


start = timeit.default_timer() #to measure runtime
train_df = train_csv
test_df = test_csv

clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)


bagging1 = BaggingClassifier(base_estimator=clf1, n_estimators=10, max_samples=0.8, max_features=0.8)
#bagging2 = BaggingClassifier(base_estimator=clf2, n_estimators=10, max_samples=0.8, max_features=0.8), 'Bagging K-NN', bagging2
randFor = RandomForestClassifier()
boosTree = GradientBoostingClassifier()
label = ['Decision Tree', 'Bagging Tree']
clf_list = [clf1, bagging1]


dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
dic1 = {'CA':0,'DA':1,'SS':3,'LOFT':4}
train_df["event"] = train_df["event"].apply(lambda x: dic[x])
#train_df['experiment'] = train_df['experiment'].apply(lambda x: dic1[x])
train_df.drop('experiment',axis=1,inplace=True) #remove labels from data

#test_df['experiment'] = test_df['experiment'].apply(lambda x: dic1[x])
test_df.drop('experiment',axis=1,inplace=True) #remove labels from data


y = train_df['event']
train_df.drop(['event'] , axis=1 ,inplace=True) #X


#train_df.drop(['id'], axis=1, inplace=True)

test_id = test_df['id']
test_df.drop(['id'], axis=1, inplace=True)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
train_df = sc.fit_transform(train_df)  
X_test = sc.transform(test_df) 

from sklearn.decomposition import PCA
pca = PCA(n_components=2)  
train_df = pca.fit_transform(train_df)  
X_test = pca.transform(X_test) 
#explained_variance = pca.explained_variance_ratio_  



from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42)
#i=0
#for clf in clf_list :
              
randFor.fit(train_df ,y)
#boosTree.fit(train_df ,y)

pred = randFor.predict_proba(X_test)

#print("%s: "%(label[i]), accuracy_score(pred, y_test))
#    i=i+1
#    cnf_matrix = confusion_matrix(y_test, pred)
#    sns.heatmap(cnf_matrix)

sub = pd.DataFrame(pred,columns=['A', 'B', 'C', 'D'])
sub['id'] = test_id
cols = sub.columns.tolist()
sub.to_csv("randFor_PCA_No_Outliers.csv", index=False)









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



stop = timeit.default_timer()
print('\nRunTime:', stop - start, 'sec')



#%%


sub.to_csv("randFor_PCA_No_Outlirs.csv", index=False)


#%%

























import pandas as pd
import timeit
import matplotlib.pyplot as plt
import seaborn as sns




start = timeit.default_timer() #to measure runtime

train = pd.read_csv(r"C:\Users\wnp387\Desktop\New folder\train.csv")
test = pd.read_csv(r"C:\Users\wnp387\Desktop\New folder\test.csv")

#train = reduce_mem_usage(train)
print(len(train))
train = train.infer_objects() #convert all data types in DF to numeric
#print(train1.dtypes)

train_labels = train[['event','crew','experiment','time','seat']] #get labels
train_data = train.drop(['event','crew','experiment','seat'],axis=1) #remove labels from data


#from scipy import stats #clear outliers!!!
#zscore = np.abs(stats.zscore(train_data)) # clear outliers from every label's DF
#train_data = train_data[(zscore < 3).all(axis=1)] # 95% of valid data! - 3 STDs from normal distr.
#train_labels = train_labels[(zscore < 3).all(axis=1)] # 95% of valid data! - 3 STDs from normal distr.

dataA  = train_data[train['event'] == 'A']
labelA = train_labels[train['event'] == 'A']
dataB  = train_data[train['event'] == 'B']
labelB = train_labels[train['event'] == 'B']
dataC  = train_data[train['event'] == 'C']
labelC = train_labels[train['event'] == 'C']
dataD  = train_data[train['event'] == 'D']
labelD = train_labels[train['event'] == 'D']

all_data = [dataA,dataB,dataC,dataD]
all_labels = [labelA,labelB,labelC,labelD]




#plot distribution of some features.
labelD = pd.concat([dataD,labelD], axis = 1)
labelD['id'] = range(1, len(dataD) + 1)
plot = sns.pairplot(x_vars=['time'], y_vars=['r'], data=labelD, hue="event", size = 10)
plot.fig.suptitle("respiration signal vs id in label D")







lenA_w_OL  = len(dataA)
lenB_w_OL  = len(dataB)
lenC_w_OL  = len(dataC)
lenD_w_OL  = len(dataD)

from scipy import stats #clear outliers!!!
for i in range(len(all_data)):
    zscore = np.abs(stats.zscore(all_data[i])) # clear outliers from every label's DF
    all_data[i] = all_data[i][(zscore < 3).all(axis=1)] # 95% of valid data! - 3 STDs from normal distr.
    all_labels[i] = all_labels[i][(zscore < 3).all(axis=1)] # 95% of valid data! - 3 STDs from normal distr.


all_label_columns = [0]*len(all_data)
for i in range(len(all_data)):
    all_label_columns[i] = pd.concat([all_data[i],all_labels[i]], axis=1) #concat columns
    
    
train_no_outliers = pd.concat(all_label_columns, axis=0) #concat rows
print(len(train_no_outliers))

lenA_no_OL  = len(train_no_outliers[train_no_outliers['event'] == 'A'])
lenB_no_OL  = len(train_no_outliers[train_no_outliers['event'] == 'B'])
lenC_no_OL  = len(train_no_outliers[train_no_outliers['event'] == 'C'])
lenD_no_OL  = len(train_no_outliers[train_no_outliers['event'] == 'D'])

    
stop = timeit.default_timer()
print('\nRunTime:', stop - start, 'sec')

train_no_outliers.to_csv('train_no_outliers.csv',index=False)


#plt.figure(figsize=(15,10))
#sns.countplot('event', hue='crew', data=train_no_outliers)
#plt.xlabel("crew and state of the pilot", fontsize=12)
#plt.ylabel("Count (log)", fontsize=12)
##plt.yscale('log')
#plt.title("crew vs event", fontsize=15)
#plt.show()

#train.to_csv('new.csv',index=False)
#
#['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.']
#
#
#print(len(train_data))
#
#list_df = [train_data,train_labels]
#
#train_fin   = pd.concat(list_df, axis=1)   #last 20% of data for label i
#
#train_fin.to_csv('new.csv',index=False)
#
#
#labelA = train_fin[train_fin['event'] == 'A']
#labelB = train_fin[train_fin['event'] == 'B']
#labelC = train_fin[train_fin['event'] == 'C']
#labelD = train_fin[train_fin['event'] == 'D']
#
#len_data = len(train_fin)
#
#per_a_in_data = len(labelA)/len_data
#per_b_in_data = len(labelB)/len_data
#per_c_in_data = len(labelC)/len_data
#per_d_in_data = len(labelD)/len_data
#
#A_50k = labelA[0:(int)(50000*per_a_in_data)]
#B_50k = labelB[0:(int)(50000*per_b_in_data)]
#C_50k = labelC[0:(int)(50000*per_c_in_data)]
#D_50k = labelD[0:(int)(50000*per_d_in_data)]
#
#tO_MATLAB = pd.concat([A_50k,B_50k,C_50k,D_50k], axis=0)   #last 20% of data for label i
#tO_MATLAB.to_csv('TO_MATLAB.csv',index=False)
#
#one_per_of_data = (int)(0.01*len_data)
#
#
#stop = timeit.default_timer()
#print('\nRunTime:', stop - start, 'sec')














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

