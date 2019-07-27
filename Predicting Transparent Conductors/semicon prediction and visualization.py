
from sklearn import linear_model
from sklearn.model_selection import cross_validate
import matplotlib.colors as colors
import matplotlib.cm as cmx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn import model_selection
import seaborn as sns

#%%
######################################### functions ############################################


def meta_features(df):
    df['per_al_ga'] = df['per_al'] * df['per_ga']
    df['per_al_in'] = df['per_al'] * df['per_in']
#    df['per_ga_in'] = df['per_ga'] * df['per_in']
#    
#    df['l_vec_1_ang_/_2'] = df['l_vec_1_ang'] / df['l_vec_2_ang']
#    df['l_vec_2_ang_/_3'] = df['l_vec_2_ang'] / df['l_vec_3_ang']
#    df['l_vec_3_ang_/_1'] = df['l_vec_3_ang'] / df['l_vec_1_ang']
#    
#    df['lattice_angle_alpha_beta_degree'] = df['l_ang_alpha_deg'] * df['l_ang_beta_deg']
#    df['lattice_angle_beta_gamma_degree'] = df['l_ang_beta_deg'] * df['l_ang_gamma_deg']
#    df['lattice_angle_gamma_alpha_degree'] = df['l_ang_gamma_deg'] * df['l_ang_alpha_deg']
#    
#    
#    df['per_al_ga_in/lv1'] = (df['per_al'] + df['per_ga'] + df['per_in']) / df['l_vec_1_ang']
#    df['per_al_ga_in/lv2'] = (df['per_al'] + df['per_ga'] + df['per_in']) / df['l_vec_2_ang']
#    df['per_al_ga_in/lv3'] = (df['per_al'] + df['per_ga'] + df['per_in']) / df['l_vec_3_ang']
#    
#    df['per_al_ga_in*alpha'] = (df['per_al'] + df['per_ga'] + df['per_in']) * df['l_ang_alpha_deg']
#    df['per_al_ga_in*beta'] = (df['per_al'] + df['per_ga'] + df['per_in']) * df['l_ang_beta_deg']
#    df['per_al_ga_in*gamma'] = (df['per_al'] + df['per_ga'] + df['per_in']) * df['l_ang_gamma_deg']
#    
#    df['lattice_vector_A_B_G_1'] = np.sqrt(df['l_ang_alpha_deg'] * df['l_ang_beta_deg'] * df['l_ang_gamma_deg']) / df['l_vec_1_ang']
#    df['lattice_vector_A_B_G_2'] = np.sqrt(df['l_ang_alpha_deg'] * df['l_ang_beta_deg'] * df['l_ang_gamma_deg']) / df['l_vec_2_ang']
#    df['lattice_vector_A_B_G_3'] = np.sqrt(df['l_ang_alpha_deg'] * df['l_ang_beta_deg'] * df['l_ang_gamma_deg']) / df['l_vec_3_ang']
#    
#    df['lattice_123_A_B'] = (df['l_vec_1_ang'] + df['l_vec_2_ang'] + df['l_vec_3_ang']) / (df['l_ang_alpha_deg'] * df['l_ang_beta_deg'])
#    df['lattice_123_B_G'] = (df['l_vec_1_ang'] + df['l_vec_2_ang'] + df['l_vec_3_ang']) / (df['l_ang_beta_deg'] * df['l_ang_gamma_deg'])
#    df['lattice_123_G_A'] = (df['l_vec_1_ang'] + df['l_vec_2_ang'] + df['l_vec_3_ang']) / (df['l_ang_gamma_deg'] * df['l_ang_alpha_deg'])
#    
#    df['NTA_al_ga_in'] = df['Natoms'] * (df['per_al'] + df['per_ga'] + df['per_in'])
#    df['NTA_1_2_3'] = df['Natoms'] * (df['l_vec_1_ang'] + df['l_vec_2_ang'] + df['l_vec_3_ang'])
#    df['NTA_A_B_G'] = df['Natoms'] * (df['l_ang_alpha_deg'] + df['l_ang_beta_deg'] + df['l_ang_gamma_deg'])


def vol_and_atom_density(df):
    """
    Returns:
        volume (float) of the parallelepiped unit cell
    """
    
    # Degree to radian
    df['alpha_rad'] = np.radians(df['l_ang_alpha_deg'])
    df['beta_rad'] = np.radians(df['l_ang_beta_deg'])
    df['gamma_rad'] = np.radians(df['l_ang_gamma_deg'])
    
    #drop redundant data (degree data is same as radian data)
    df = df.drop(['l_ang_alpha_deg','l_ang_beta_deg','l_ang_gamma_deg'],axis = 1)

    volume = df['l_vec_1_ang']*df['l_vec_2_ang']*df['l_vec_3_ang']*np.sqrt(
    1 + 2*np.cos(df['alpha_rad'])*np.cos(df['beta_rad'])*np.cos(df['gamma_rad'])
    -np.cos(df['alpha_rad'])**2 -np.cos(df['beta_rad'])**2 -np.cos(df['gamma_rad'])**2)
    
    df['volume'] = volume
    df['density'] = df['Natoms'] / df['volume']
    return df
    
    
def rmsle(y_true,y_pred): #kaggle's score
    assert len(y_true) == len(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5


#%%

def bestClassifier(train_df, feat1, feat2, idx): #kaggle's score
        
    labels_1 = feat1
    labels_2 = feat2
    features = train_df
    
    # prepare configuration for cross validation test harness
    seed = 2300
    
    # prepare models
    models = []
    models.append(('ElastiNet', linear_model.ElasticNet(max_iter=10000000)))
    models.append(('BayesRidg', linear_model.BayesianRidge()))
    models.append(('LassoLars', linear_model.LassoLars(max_iter=10000000)))
    models.append(('_RidgeReg', linear_model.Ridge(max_iter=10000000)))
    models.append(('_LassoReg', linear_model.Lasso(max_iter=10000000)))
    models.append(('LinearReg', linear_model.LinearRegression()))
    
    # evaluate each model in turn
    results1 = []
    results2 = []
    names = []
    
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=5, random_state=seed)
    	cv_results1 = model_selection.cross_val_score(model, features, labels_1, cv=kfold, scoring=rmsle_score)
    	results1.append(cv_results1)
    	names.append(name)
        
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=5, random_state=seed)
    	cv_results2 = model_selection.cross_val_score(model, features, labels_2, cv=kfold, scoring=rmsle_score)
    	results2.append(cv_results2)
    	names.append(name)
        
    means1 = [np.mean(i) for i in results1]
    means2 = [np.mean(i) for i in results2]
    
    means = []
    
    for i in list(range(len(means1))):
        means.append((means1[i] + means2[i])/2)
    
    print('SG no.:' + str(idx) + ' lowest RMSLE reg:' , names[means.index(min(means))], min(means))




######################################### end of functions ####################################

#%% read csv data
    
train = pd.read_csv(r"C:\Users\wnp387\Documents\Python Scripts\kaggle_comp_1\train.csv")
test = pd.read_csv(r"C:\Users\wnp387\Documents\Python Scripts\kaggle_comp_1\test.csv")

rmsle_score = make_scorer(rmsle, greater_is_better = True)

train.rename(columns={'number_of_total_atoms':'Natoms',
                      'percent_atom_al':'per_al',
                      'percent_atom_ga':'per_ga',
                      'percent_atom_in':'per_in',
                      'lattice_vector_1_ang':'l_vec_1_ang',
                      'lattice_vector_2_ang':'l_vec_2_ang',
                      'lattice_vector_3_ang':'l_vec_3_ang',
                      'lattice_angle_alpha_degree':'l_ang_alpha_deg',
                      'lattice_angle_beta_degree':'l_ang_beta_deg',
                      'lattice_angle_gamma_degree':'l_ang_gamma_deg',
                       }, inplace=True)
    
test.rename(columns={'number_of_total_atoms':'Natoms',
                      'percent_atom_al':'per_al',
                      'percent_atom_ga':'per_ga',
                      'percent_atom_in':'per_in',
                      'lattice_vector_1_ang':'l_vec_1_ang',
                      'lattice_vector_2_ang':'l_vec_2_ang',
                      'lattice_vector_3_ang':'l_vec_3_ang',
                      'lattice_angle_alpha_degree':'l_ang_alpha_deg',
                      'lattice_angle_beta_degree':'l_ang_beta_deg',
                      'lattice_angle_gamma_degree':'l_ang_gamma_deg',
                       }, inplace=True)

#%% plots and table
plt.close('all')

#no. of substances of each spacegroup
yy = pd.value_counts(train['spacegroup'])
fig, ax = plt.subplots()
ax = sns.barplot(x=yy.index, y=yy, data=train)
ax.set_xticklabels(ax.get_xticklabels())
ax.set(xlabel='Spacegroup', ylabel='num substances from the spacegroup')
ax.set_title('Distribution of space groups')

#formation_energy_ev_natom vs bandgap_energy_ev scatterplot
fig , ax = plt.subplots()
plt.scatter(train['formation_energy_ev_natom'],train['bandgap_energy_ev'],color=['r','b'])
plt.xlabel('formation_energy_ev_natom', fontsize=18); plt.ylabel('bandgap_energy_ev', fontsize=16)
ax.set_title('formation_energy_ev_natom vs bandgap_energy_ev scatterplot')

#Correlation matrix plot
cor_mat = train.corr()
fig , ax = plt.subplots()
sns.heatmap(cor_mat,center= 0,annot=True)
ax.set_title('Correlation matrix')

plot = sns.pairplot(x_vars=['per_al'], y_vars=['bandgap_energy_ev'], data=train, hue="spacegroup", size = 10)
plot.fig.suptitle("bandgap_energy_ev vs percent of aluminium")

plot = sns.pairplot(x_vars=['id'], y_vars=['bandgap_energy_ev'], data=train, hue="spacegroup", size = 10)
plot.fig.suptitle("bandgap_energy_ev vs id")

plot = sns.pairplot(x_vars=['id'], y_vars=['formation_energy_ev_natom'], data=train, hue="spacegroup", size = 10)
plot.fig.suptitle("formation_energy_ev_natom vs id")

#table of spacegroup vs no. of tot atoms
print('\n', pd.crosstab(train['Natoms'],train['spacegroup']), '\n')

#table of spacegroup vs no. of tot atoms - mean of other features
sg_v_NAtoms = train.groupby(['spacegroup', 'Natoms']).mean()


normalized = train[['id', 'bandgap_energy_ev', 'formation_energy_ev_natom', 'spacegroup']]
normalized['bandgap_energy_ev'] = np.log1p(normalized['bandgap_energy_ev'])
normalized['formation_energy_ev_natom'] = np.log1p(normalized['formation_energy_ev_natom'])
plot = sns.lmplot(x = 'id', y = "bandgap_energy_ev", data = normalized, hue="spacegroup")
plot.fig.suptitle("bandgap_energy_ev vs id, normalized bandgap_energy_ev, with outliers")

plot = sns.lmplot(x = 'id', y = 'formation_energy_ev_natom', data = normalized, hue="spacegroup")
plot.fig.suptitle("formation_energy_ev_natom vs id, normalized formation_energy_ev_natom, with outliers")
#%% extract meta features

meta_features(train)  
meta_features(test) 
train = vol_and_atom_density(train)
test = vol_and_atom_density(test)

plot = sns.pairplot(x_vars=['density'], y_vars=['bandgap_energy_ev'], data=train, hue="spacegroup", size = 10)
plot.fig.suptitle("bandgap_energy_ev vs substance density")

##Correlation matrix plot
#cor_mat = train.corr()
#fig , ax = plt.subplots()
#sns.heatmap(cor_mat,center= 0,annot=True)
#ax.set_title('Correlation matrix')


#%% PCA

vector = np.vstack((train[['l_vec_1_ang', 'l_vec_2_ang','l_vec_3_ang']].values,
                    test[['l_vec_1_ang', 'l_vec_2_ang','l_vec_3_ang']].values))
pca = PCA().fit(vector)
train['vector_pca0'] = pca.transform(train[['l_vec_1_ang', 'l_vec_2_ang','l_vec_3_ang']])[:, 0]
test['vector_pca0'] = pca.transform(test[['l_vec_1_ang', 'l_vec_2_ang','l_vec_3_ang']])[:, 0]

#%% 

from scipy import stats
zscore = np.abs(stats.zscore(train))
outliers_clean = train[(zscore < 3).all(axis=1)]

plot = sns.pairplot(x_vars=['id'], y_vars=['formation_energy_ev_natom'], data=outliers_clean, hue="spacegroup", size = 10)
plot.fig.suptitle("formation_energy_ev_natom vs id after outlier clean")

plot = sns.pairplot(x_vars=['id'], y_vars=['bandgap_energy_ev'], data=outliers_clean, hue="spacegroup", size = 10)
plot.fig.suptitle("bandgap_energy_ev vs id after outlier clean")

normalized = outliers_clean[['id', 'bandgap_energy_ev', 'formation_energy_ev_natom', 'spacegroup']]
normalized['bandgap_energy_ev'] = np.log1p(normalized['bandgap_energy_ev'])
normalized['formation_energy_ev_natom'] = np.log1p(normalized['formation_energy_ev_natom'])
plot = sns.lmplot(x = 'id', y = "bandgap_energy_ev", data = normalized, hue="spacegroup")
plot.fig.suptitle("bandgap_energy_ev vs id, normalized bandgap_energy_ev, outlier free")

plot = sns.lmplot(x = 'id', y = 'formation_energy_ev_natom', data = normalized, hue="spacegroup")
plot.fig.suptitle("formation_energy_ev_natom vs id, normalized formation_energy_ev_natom, outlier free")

train_ids = outliers_clean['id']
outliers_clean = outliers_clean.drop('id',axis = 1)

print('train rows with outliers:', len(train))
print('train rows w/o outliers:', len(outliers_clean))

#Correlation matrix plot
cor_mat = outliers_clean.corr()
fig , ax = plt.subplots()
sns.heatmap(cor_mat,center= 0,annot=True)
ax.set_title('Correlation matrix')


#sort by spacegroup and Natoms
test = test.sort_values(['spacegroup','Natoms'], axis=0, ascending=True)
ids = test['id']

unique_SGs = np.unique(outliers_clean['spacegroup'])
dict_SGs = {key: value for key, value in outliers_clean.groupby('spacegroup')}
dict_test_SGs = {key: value for key, value in test.groupby('spacegroup')}

sg_list = []
linearRegs = []
sgt_list = []
y_pred1 = []
y_pred2 = []

for i in range(len(unique_SGs)):
    sg_list.append(dict_SGs[unique_SGs[i]])
    sgt_list.append(dict_test_SGs[unique_SGs[i]])
    linearRegs.append(linear_model.LinearRegression())

# fit lin regression on every spacegroup seperately
for i in range(len(unique_SGs)):
    labels1 = sg_list[i]['formation_energy_ev_natom']
    labels2 = sg_list[i]['bandgap_energy_ev']
    feat = sg_list[i].drop(['formation_energy_ev_natom','bandgap_energy_ev'], axis=1)
    sgt_list[i] = sgt_list[i].drop('id',axis = 1)
    
    #check which regressor is best for each spacegroup
    bestClassifier(feat, labels1, labels2,i)
    
    linearRegs[i].fit(feat, labels1) 
    y_pred1.extend(linearRegs[i].predict(sgt_list[i]))
    linearRegs[i].fit(feat, labels2) 
    y_pred2.extend(linearRegs[i].predict(sgt_list[i]))  
    
submit = pd.DataFrame({'id':ids,'formation_energy_ev_natom':y_pred1[:]
                                        ,'bandgap_energy_ev':y_pred2[:]})
submit = submit.sort_values(['id'], axis=0, ascending=True)
submit.to_csv('30_12.csv',index=False)
    
#%%
#
#
##types of linear regression
#ridge = linear_model.Ridge(alpha=10)
#lasso = linear_model.Lasso()
#LassoLars = linear_model.LassoLars(alpha=.1)
#BayesianRidge = linear_model.BayesianRidge()
#LinearRegression = linear_model.LinearRegression()
#enet = linear_model.ElasticNet(alpha=0.1)
# 
#S_regressions = ['lasso','ridge','LassoLars','ElasticNet','LinearRegression']
#avgs = []
#
#cv_results = cross_validate(lasso, features, labels, scoring = rmsle_score
#                            , cv=5,return_train_score=False)
#print('lasso:',cv_results['test_score'])
#avgs.append(sum(cv_results['test_score']) / float(len(cv_results['test_score'])))
#
#cv_results = cross_validate(ridge, features, labels, scoring = rmsle_score
#                            , cv=5,return_train_score=False)
##print(sorted(cv_results.keys()))
#print('ridge:', cv_results['test_score'])
#avgs.append(sum(cv_results['test_score']) / float(len(cv_results['test_score'])))
#
#cv_results = cross_validate(LassoLars, features, labels, scoring = rmsle_score
#                            , cv=5,return_train_score=False)
#print('LassoLars:', cv_results['test_score'])
#avgs.append(sum(cv_results['test_score']) / float(len(cv_results['test_score'])))
#
#cv_results = cross_validate(enet, features, labels, scoring = rmsle_score
#                            , cv=5,return_train_score=False)
#print('ElasticNet:', cv_results['test_score'])
#avgs.append(sum(cv_results['test_score']) / float(len(cv_results['test_score'])))
#
#cv_results = cross_validate(LinearRegression, features, labels, scoring = rmsle_score
#                            , cv=5,return_train_score=False)
#print('LinearRegression:', cv_results['test_score'])
#avgs.append(sum(cv_results['test_score']) / float(len(cv_results['test_score'])))
#print('\navg for every result: \n', avgs, '\n')
#
#winner_reg_ind = avgs.index(max(avgs))
##check if negating the score is really the score.
#print('winner avg err:', max(avgs)*(-1), ',reg name:', S_regressions[winner_reg_ind] , 'regression ') #index from 0 to n-1
#
#
#IDs = test['id']
#test = test.drop('id',axis = 1)
#LinearRegression.fit(features, labels) 
#y_pred = LinearRegression.predict(test)
#
#
##col = ['formation_energy_ev_natom','bandgap_energy_ev']
##score = rmsle(test[col[0]],y_pred)
##print('R square for {} is {}:'.format(col[1],score))
##score = rmsle(test[col[1]],y_pred)
##print('R square for {} is {}:'.format(col[1],score))
#
#
## confusion_matrix(y_true, y_pred)
#
#submit = pd.DataFrame({'id':IDs,'formation_energy_ev_natom':y_pred[:,0],'bandgap_energy_ev':y_pred[:,1]})
#submit.to_csv('lin_reg_kaggle.csv',index=False)
