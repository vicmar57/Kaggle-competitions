import gc
#load test files
df_test = pd.read_csv('../input/test.csv')
#downcast the train data
test_int = df_test.select_dtypes(include=['int'])
t_converted_int = test_int.apply(pd.to_numeric,downcast='unsigned')
test_float = df_test.select_dtypes(include=['float'])
t_converted_float = test_float.apply(pd.to_numeric,downcast='float')

t_converted_obj = pd.DataFrame()
t_converted_obj.loc[:,'experiment'] = df_test['experiment'].astype('category')

df_test[t_converted_int.columns] = t_converted_int
df_test[t_converted_float.columns] = t_converted_float
df_test[t_converted_obj.columns] = t_converted_obj


del [[t_converted_int,t_converted_float,t_converted_obj,test_int,test_float,]]
gc.collect()

#load train files
df_train_orig = pd.read_csv('../input/train.csv')
df_train_orig.info(memory_usage='deep')

#downcast the train data
train_int = df_train_orig.select_dtypes(include=['int'])
converted_int = train_int.apply(pd.to_numeric,downcast='unsigned')
train_float = df_train_orig.select_dtypes(include=['float'])
converted_float = train_float.apply(pd.to_numeric,downcast='float')

converted_obj = pd.DataFrame()
converted_obj.loc[:,'experiment'] = df_train_orig['experiment'].astype('category')
converted_obj.loc[:,'event'] = df_train_orig['event'].astype('category')

df_train = df_train_orig.copy()

df_train[converted_int.columns] = converted_int
df_train[converted_float.columns] = converted_float
df_train[converted_obj.columns] = converted_obj


del [[converted_int,converted_float,converted_obj,df_train_orig]]
gc.collect()

#experiment | CA=0,DA=1,SS=2
#event | A=0,B=1,C=2,D=3


