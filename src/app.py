import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_validate
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Read dataset
url = 'https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv'
url = '../data/raw/dataset.csv'
df_raw = pd.read_csv(url)

# Make a copy of the raw datset
df = df_raw.copy()

# Split dataframe in features and target
# The variable chosen as target is 'Total Specialist Physicians (2019)'
X= df.drop(['CNTY_FIPS','fips','Active Physicians per 100000 Population 2018 (AAMC)','Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)','Active General Surgeons per 100000 Population 2018 (AAMC)','Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)','Total nurse practitioners (2019)','Total physician assistants (2019)','Total physician assistants (2019)','Total Hospitals (2019)','Internal Medicine Primary Care (2019)','Family Medicine/General Practice Primary Care (2019)','STATE_NAME','COUNTY_NAME','ICU Beds_x','Total Specialist Physicians (2019)'], axis=1)
y=df['Total Specialist Physicians (2019)']

# Define train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


## Models

## Model 1: alpha selected (arbitrarily) = 10

scaler = StandardScaler() # to apply Lasso we have to scale the data

train_scaler = scaler.fit(X_train) # fit the scaler to train data

data_esc = train_scaler.transform(X_train) # transform X_train

model = Lasso(alpha=10) 

model.fit(data_esc, y_train) # fit the data to the Lasso model with alpha=10

print('Score in train data - Lasso alpha=10:', model.score(data_esc, y_train)) 

# Predict in test dataset and get score

test_esc = train_scaler.transform(X_test)

print('Score in test data - Lasso alpha=10:', model.score(test_esc, y_test)) 

## Model 2: alpha selected by CV

pipe2 = make_pipeline(StandardScaler(), Lasso())

params={
    'lasso__fit_intercept':[True,False],
    'lasso__alpha':10.0**np.arange(-2, 6, 1)
}

#setting up the grid search
gs=GridSearchCV(pipe2,params,n_jobs=-1,cv=5)

#fitting gs to training data
gs.fit(X_train, y_train)

print(#checking the selected permutation of parameters
gs.best_params_)

#checking how well the model does on the test-set
print('Score in test data - Lasso alpha selected by CV:', gs.score(X_test,y_test))

## Selecting features based on this result:

# Features whose coefficiet is different from zero selected with Model 2
model2 = gs.best_estimator_

# fit model with the hiperaparameters selected by CV
model2.fit(X_train, y_train)

# get coefficients
coef_list=model2[1].coef_

# Location of Lasso coefficients
loc2=[i for i, e in enumerate(coef_list) if e != 0]

# Features whose coefficiet is different from zero
col_name=X_train.columns
print(col_name[loc2])


## Running OLS regression in train data with selected features

# OLS with StatsModels and the selected features

X_ols = X_train[col_name[loc2]]

X_ols_int = sm.add_constant(X_ols) 

modelo_ols = sm.OLS(y_train, X_ols_int)

results = modelo_ols.fit()

results.summary()


## Saving models

# Save OLS model in models folder

filename='../models/final_ols_model.sav'
pickle.dump(modelo_ols, open(filename, 'wb'))

# Save Lasso model in models folder

model_lasso = gs.best_estimator_

filename='../models/final_lasso_model.sav'
pickle.dump(model_lasso, open(filename, 'wb'))
