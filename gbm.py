# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 08:03:05 2021

@author: willis
"""
#########
#IMPORTS#
#########
import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
##############
#LOADING DATA#
##############
#changes to project directory
os.chdir(r'D:\BEST\FOLDER\EVER')
#loads data from csv.
df = pd.read_csv('marketing_campaign_data.csv')
#creates copy of original dataset
df_orig = df
###############
#DATA CLEANING#
###############
'''
EDUCATION:
    needs label encoding
'''
# numerical representation of education
def edu_clean(barg):
    i = -1
    if barg == 'Basic':
        i = 0
    elif barg == 'Graduation':
        i = 1
    elif barg == '2n Cycle':
        i = 2
    elif barg == 'Master':
        i = 2
    elif barg == "PhD":
        i = 3
    else:
        i = i
    return  i
# applies function to dataframe
df['edu_cat'] = df['Education'].apply(edu_clean)
'''
AGE:
    There was no variable for age. As remedy the customer's birth year was
    subtracted from the year 2021.
'''
# Uses the Year_Birth column to calculate customers age
df['age'] = 2021 - df['Year_Birth']
'''
MARITAL STATUS:
    needs label encoding
'''
# numerical representation of marital status
def mari_clean(carg):
    i = -1
    if carg == 'Single': 
        i = 0
    elif carg == 'Divorced':
        i = 1
    elif carg == 'Widow':
        i = 2
    elif carg == 'Together':
        i = 3
    elif carg == "Married":
        i = 4
    else:
        i = i
    return i
# applies function to dataframe
df['mari_status'] = df['Marital_Status'].apply(mari_clean)
'''
NUMBER OF CHILDREN:
    Kids and Teens are split in this dataset.  There are added together to 
    create a count of the number of children in a hh.
    Darg: (ARG)number of "kids"
    Farg: (ARG)number of "teens"
    RETURNS: sum of darg and farg
'''
def youth_calc(darg, farg):
    i = darg + farg
    return i
#applies function of df
df['youth_calc'] = df.apply(lambda x: youth_calc(x['Kidhome'], x['Teenhome']), axis=1)
#dropping rows with missing values for two_person_hh_di which was stored as -1
df = df.drop(df[df.mari_status < 0].index)
#renaming income variable
df['inc'] = df['Income']
#combining advert engagement
df['advert_engagement'] = df['AcceptedCmp5'] + df['AcceptedCmp4'] + df['AcceptedCmp3'] + df['AcceptedCmp2'] + df['AcceptedCmp1']
#these variables have no information about them, and were dropped due to lack of interpretability
df = df.drop(['AcceptedCmp5', 'AcceptedCmp4', 'AcceptedCmp3', 'AcceptedCmp2', 'AcceptedCmp1', 'Response', 'Complain','Z_Revenue', 'Z_CostContact', 'MntGoldProds'], axis = 1)
#separating working variables
x1 = df[df.columns[8:25]]
'''
GBMs are sensitive to outliers in the DV/Target. As such, observations 
distinct from the larger distribution are culled from the dataset.
'''
#plotting possible targets
plt.figure(figsize = (20,15))
plt.subplot(1,5,1)
sns.boxplot(data = x1['MntWines'], color='orange')
plt.subplot(1,5,2)
sns.boxplot(data = x1['MntFruits'], color='purple')
plt.subplot(1,5,3)
sns.boxplot(data = x1['MntMeatProducts'], color='brown')
plt.subplot(1,5,4)
sns.boxplot(data = x1['MntFishProducts'], color='green')
plt.subplot(1,5,5)
sns.boxplot(data = x1['MntSweetProducts'], color='red')
plt.show()
'''
From the boxplots plotted above it appears that MntMeatProducts and 
MntSweetProducts have distinct breaks in their tails. They are also all 
severely postively skewed. 

MntMeatProducts > 1500 were dropped
MntSweetProdcuts > 250 were dropped
'''
#dropping rows with extreme values for MntMeatProducts.
x1 = x1.drop(x1[x1.MntMeatProducts > 1500].index)
#dropping rows with extreme values for MntSweetProducts.
x1 = x1.drop(x1[x1.MntSweetProducts > 250].index)
#plotting age and inc to check for outliers
plt.figure(figsize = (20,15))
plt.subplot(1,2,1)
sns.boxplot(data = x1['age'], color='orange')
plt.subplot(1,2,2)
sns.boxplot(data = x1['inc'], color='purple')
plt.show()
'''
The boxplots constructed above show that there are some extreme outliers in 
age and inc. These variables were effectively limited to their IQR.
Age > 100
Income > 140,000
'''
#dropping rows with extreme values for income.
x1 = x1.drop(x1[x1.inc > 140000].index)
#dropping rows with extreme values for age.
x1 = x1.drop(x1[x1.age > 100].index)
'''
Education and marital status are measured categorically and have to be broken 
out into dummy variables
*I SHOULD DO THIS EARLIER!*
'''
#education dummies
edu_dum = pd.get_dummies(x1['edu_cat'])
edu_dum = edu_dum.rename(columns={0: 'basic_edu', 1: 'graduation_edu', 2: 'master_edu', 3: 'doc_edu'})
x1 = pd.concat([x1, edu_dum], axis=1)
x1 = x1.drop(['edu_cat'], axis=1)
#marital status dummies
mari_dum = pd.get_dummies(x1['mari_status'])
mari_dum = mari_dum.rename(columns={0: 'single', 1: 'divorced', 2: 'widow', 3: 'together', 4: 'married'})
x1 = pd.concat([x1, mari_dum], axis=1)
x1 = x1.drop(['mari_status'], axis=1)
'''
Although GBMs can handle missing data, and do so very well, for my first I 
removed all observations without complete informations
'''
x1 = x1.dropna(axis=0, how='any')
'''
To simplify life I am creating a new variable that will be the regressors' 
final target it is equal to the log of the sum of the 5 amount features.
doing it here just affixes it to the end of the df and makes future calls easier    
'''
#log amount target
x1['log_total_amount'] = np.log(x1['MntWines']+x1['MntFruits']+x1['MntMeatProducts']+x1['MntFishProducts']+x1['MntSweetProducts'])
#checking the identity of the sum
sum_total_amount = x1['MntWines']+x1['MntFruits']+x1['MntMeatProducts']+x1['MntFishProducts']+x1['MntSweetProducts']
#checking what a square root transformation does
sqrt_total_amount = np.sqrt(x1['MntWines']+x1['MntFruits']+x1['MntMeatProducts']+x1['MntFishProducts']+x1['MntSweetProducts'])
#dropping individual amount features
x1 = x1.drop(['MntWines', 'MntMeatProducts', 'MntFruits', 'MntFishProducts', 'MntSweetProducts'], axis=1)
#plotting target for safety check
plt.figure(figsize=(20,15))
sns.displot(x1['log_total_amount'])
plt.show
#correlation matrix just for fun
def correlation_heatmap(dataframe,l,w):
    correlation = dataframe.corr()
    plt.figure(figsize=(l,w))
    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
    plt.title('Correlation between different features')
    plt.show();
correlation_heatmap(x1, 30,15)
################
#SPLITTING DATA#
################
#checking target column location for feature/target split
print(x1.columns.get_loc("log_total_amount"))
#gets column values and separates them into features and target
dataset = x1.values
X = dataset[:,0:19]
Y = dataset[:,19]
#creates scaler and scales features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data into train and test sets
seed = 2448 #seed for replication
test_size = 0.30 #train on 70% of observations test on 15% and validate on 15%
#first split
X_train, X_remain, y_train, y_remain = train_test_split(X, Y, test_size=test_size, random_state=seed)
#second split
X_validate, X_test, y_validate, y_test = train_test_split(X_remain, y_remain, test_size=.5, random_state=seed)
#tranforming data into xgboost DataMatrices
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_test, label=y_test)
xval = xgb.DMatrix(X_validate, label=y_validate)
#############
#Y-BAR MODEL#
#############
mean_train = np.mean(y_train)
baseline_predictions = np.ones(y_test.shape) * mean_train
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))
##############
#BUILDING GBM#
##############
params = {
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': .1,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective': 'reg:squarederror',
    }
params['eval_metric'] = 'mae'
num_boost_rounds = 999
model = xgb.train(
    params, 
    dtrain,
    num_boost_round = num_boost_rounds,
    evals = [(dval, "Val")], #sets
    early_stopping_rounds = 10 #model must improve every ten epochs or training halts
    )

print("Best MAE: {:.5f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

###################
#EVAL UN TUNED GBM#
###################
# make predictions for test data
predictions = model.predict(dval)


# evaluate predictions
print('R2 Value:',metrics.r2_score(np.exp(y_test), np.exp(predictions)))
#print('Accuracy',100- (np.mean(np.abs((y_test - predictions) / y_test)) * 100))

plot_importance(model)
pyplot.show()

plt.figure(figsize= (20,15)).suptitle('GBM predictions & true value (N=2087)')
plt.scatter(np.exp(predictions), np.exp(y_test))
plt.show()

plt.figure(figsize= (20,15))
sns.displot(data = predictions, kde=True)
plt.show()

plt.figure(figsize= (20,15))
sns.displot(data =y_test, kde=True)
plt.show()


####################################
#TUNING TREE DEPTH AND CHILD WEIGHT#
####################################
#setting up a grid search!
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(6,12)
    for min_child_weight in range(5,8)
]

min_mae = float("Inf")
best_params = None

for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    
    # update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    
    # run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed=43,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=5
    )
    
    # update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
        
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))



params['max_depth'] = 10
params['min_child_weight'] = 5

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]

min_mae = float("Inf")
best_params = None

# start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    
    # update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    
    # run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    
    # update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
        
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

params['subsample'] = 0.9
params['colsample_bytree'] = 0.8


min_mae = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    
    # update our parameters
    params['eta'] = eta
    
    # run and time CV
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_rounds,
        seed=42,
        nfold=5,
        metrics=['mae'],
        early_stopping_rounds=10
    )
    
    # update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))

params['eta'] = .1

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_rounds,
    evals=[(dval, "Val")],
    early_stopping_rounds=10
)
########################
#TUNED MODEL EVALUATION#
########################
# make predictions for test data
predictions = model.predict(xval)
#feature importance
plot_importance(model)
pyplot.show()
#pseudo residuals
pseudo_resi = np.exp(y_validate) - np.exp(predictions)
#y validation identity
y_val_iden = np.exp(y_validate)
#prediction identity
prediction_iden = np.exp(predictions)
#pseudo r^2
print('R2 Value:',metrics.r2_score(y_val_iden, prediction_iden))
#predictions vs. true values
plt.figure(figsize= (20,15)).suptitle('GBM predictions & true value')
plt.scatter(y_val_iden,prediction_iden)
plt.show()
#distribution of predictions
plt.figure(figsize= (20,15))
sns.displot(data = prediction_iden, kde=True)
plt.show()
#distribution of true values
plt.figure(figsize= (20,15))
sns.displot(data = y_val_iden, kde=True)
plt.show()
#pseudo residual plot
plt.figure(figsize= (20,15)).suptitle('GBM predictions & true value (N=2087)')
plt.scatter(y_val_iden, pseudo_resi)
plt.show()

