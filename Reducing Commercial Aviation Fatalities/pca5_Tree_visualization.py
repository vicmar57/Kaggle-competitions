# -- coding: utf-8 --
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import timeit


start = timeit.default_timer() #to measure runtime

train_df = pd.read_csv(r"C:\Users\wnp387\Desktop\train.csv")

test_df = pd.read_csv(r"C:\Users\wnp387\Desktop\test.csv")

stop = timeit.default_timer()
print('\nRunTime:', stop - start, 'sec')

#%%
start = timeit.default_timer() #to measure runtime

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.gridspec as gridspec
import itertools


clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=2)
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


pca = PCA(n_components=2)  
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
sub.to_csv("Baggingdepth2PCA2.csv", index=False)
#%%


# -- coding: utf-8 --
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import timeit


start = timeit.default_timer() #to measure runtime

train_df = pd.read_csv(r"C:\Users\wnp387\Desktop\train.csv")

test_df = pd.read_csv(r"C:\Users\wnp387\Desktop\test.csv")

stop = timeit.default_timer()
print('\nRunTime:', stop - start, 'sec')

#%%
start = timeit.default_timer() #to measure runtime

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.gridspec as gridspec
import itertools


train_df = pd.read_csv(r"C:\Users\wnp387\Desktop\train.csv")
y = train_df['event']

clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=2)

clf1.fit(train_df ,y)

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf1, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

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