# %% [markdown]
# ML ZOOMCAMP
# WEEK 2: LINEAR REGRESSION
# 
# GOAL: to create a regression model for predicting housing prices (column 'median_house_value')
# 
# DATASET: The California Housing Prices from Kaggle.
# 
# Here's a wget-able link:
# 
# wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv
# 
# 

# %%
#Load the libraries

import pandas as pd
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn


# %%
#load the data

housing=pd.DataFrame(pd.read_csv("housing_data.csv"))
housing.head(5)

# %% [markdown]
# STEP 1: EDA
#     Load the data.
#     Look at the median_house_value variable. Does it have a long tail?
#     
#     Features
#     For the rest of the homework, you'll need to use only these columns:
# 
#       'latitude',
#       'longitude',
#       'housing_median_age',
#       'total_rooms',
#       'total_bedrooms',
#       'population',
#       'households',
#       'median_income',
#       'median_house_value'
#       Select only them.

# %%
sns.histplot(housing.median_house_value)
plt.show()

# From chart below we can see that the attribute has slightly skewed distribution (long tail). A lot of observation has highest values (we can see peak). 

# %%
sns.histplot(np.log1p(housing.median_house_value))
plt.show

# %%
#let's exclude features which we are not gonna use. 

house_data = housing.drop(['ocean_proximity'], axis=1)
house_data.head(5)

# %% [markdown]
# Question 1:  Find a feature with missing values. How many missing values does it have?
# 
# Question 2 : What's the median (50% percentile) for variable 'population'?

# %%
#Let's check if there are NA values in dataset

house_data.isna().sum()

# answer: we have one attribute which has 207 missing values = total_bedrooms

# %%
house_data.total_bedrooms.describe()

# %%
# calculate median for population attribute

house_data.population.median()

# answer: median(population) = 1166.0

# %% [markdown]
# STEP 2. Split the data
#     Shuffle the initial dataset, use seed 42.
#     Split your data in train/val/test sets, with 60%/20%/20% distribution.
#     Make sure that the target value ('median_house_value') is not in your dataframe.
#     Apply the log transformation to the median_house_value variable using the np.log1p() function.

# %%
np.random.seed(42)

# %%
# size of dataset
n = len(house_data)

# split dataframe into 3 parts: train, validation, test sets
# let evaluate number of observation of each split
val_n = int(0.2 * n)
test_n = int(0.2 * n)
train_n = n - (val_n + test_n)

#shuffle the dataset
idx = np.arange(n)
np.random.shuffle(idx)

house_data_shuffled = house_data.iloc[idx]

#final split
df_train = house_data_shuffled.iloc[:train_n].copy()
df_val = house_data_shuffled.iloc[train_n:train_n+val_n].copy()
df_test = house_data_shuffled.iloc[train_n+val_n:].copy()

# %%
#let's split predictors and outcome
Y_train_orig = df_train.median_house_value.values
Y_val_orig = df_val.median_house_value.values
Y_test_orig = df_test.median_house_value.values

Y_train = np.log1p(df_train.median_house_value.values)
Y_val = np.log1p(df_val.median_house_value.values)
Y_test = np.log1p(df_test.median_house_value.values)

df_train.drop(['median_house_value'], axis=1)
df_val.drop(['median_house_value'],axis=1)
df_test.drop(['median_house_value'],axis=1)

# %% [markdown]
# Question 3: We need to deal with missing values for the column from Q1.
# 
#       We have two options: fill it with 0 or with the mean of this variable.
#       Try both options. For each, train a linear regression model without regularization using the code from the lessons.
#       For computing the mean, use the training only!
#       Use the validation dataset to evaluate the models and compare the RMSE of each option.
#       Round the RMSE scores to 2 decimal digits using round(score, 2)
#       Which option gives better RMSE?
# 

# %%
# to define a function to fill the missing values

def prepare_X(df, method):

    if method=="zero":
       df= df.fillna(0)

    elif method=="mean":
       avg_value = 537.870553
       print('Mean of attribute is', avg_value)
       
       df=df.fillna(avg_value)    

    X = df.values
    return X


# %%
# define a function to build linear regression

def train_linear_regression(X,y):
    
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]


# %%
#define a function to find best model (criteria = RMSE)

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


# %%
# Scenario 1: fill NA with zeros

X_train=prepare_X(df_train, "zero")
print(X_train.shape)

w_0, w = train_linear_regression(X_train, Y_train)
y_pred = w_0 + X_train.dot(w)
rmse(Y_train,y_pred)

#answer: 0.16494619513100753

# %%
#Scenario 2: fill NA with mean

X_train=prepare_X(df_train, "mean") 
  
w_0, w = train_linear_regression(X_train, Y_train)
y_pred = w_0 + X_train.dot(w)
rmse(Y_train,y_pred)

#answer: 0.16494619513100753

# %% [markdown]
# STEP 3. Let's train a regularized linear regression.
# 
#    For this question, fill the NAs with 0.
#    Try different values of r from this list: [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10].
#    Use RMSE to evaluate the model on the validation dataset.
#    Round the RMSE scores to 2 decimal digits.

# %%
# to define a function to build regularized linear function

def train_linear_regression_reg(X, y, r):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

# %%
for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = train_linear_regression_reg(X_train, Y_train, r=r)
    #print('%5s, %.2f, %.2f, %.2f' % (r, w_0, w[1], w[5]))
    

# %% [markdown]
# Question 4: 
# 
# Which r gives the best RMSE?
# If there are multiple options, select the smallest r.
# 
# Options: [0,0.000001,0.001,0.0001]
# 

# %%
#define a function which would compare model performance and return the best value of parameter r

def best_r(x_train, y_train, x_val,y_val,r):
    best=()
    performance = {}

    for param in r:
        w_0,w=train_linear_regression_reg(x_train,y_train,r=param)
        y_pred=w_0+x_val.dot(w)
        p=round(rmse(y_val,y_pred),3)

        if not best: 
            best = (param,p)
            performance[param]=p
        elif best[1] <= p:
            performance[param]=p
        else: 
            best=(param,p)
            performance[param]=p

    print(performance)

    return best

X_train=prepare_X(df_train, "zero")
X_val=prepare_X(df_val, "zero")
r = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

best_rmse=best_r(X_train,Y_train,X_val,Y_val, r)
print(best_rmse)

# %% [markdown]
# STEP 4. 
# 
#   We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
#   Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
#   For each seed, do the train/validation/test split with 60%/20%/20% distribution.
#   Fill the missing values with 0 and train a model without regularization.
#   For each seed, evaluate the model on the validation dataset and collect the RMSE scores.

# %%
seed_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

seed_output={}
for val in seed_values:
    np.random.seed(val)
    
    #shuffle the dataset
    idx = np.arange(n)
    np.random.shuffle(idx)

    house_data_shuffled = house_data.iloc[idx]

    #final split
    df_train = house_data_shuffled.iloc[:train_n].copy()
    df_val = house_data_shuffled.iloc[train_n:train_n+val_n].copy()
    df_test = house_data_shuffled.iloc[train_n+val_n:].copy()

    #let's split predictors and outcome
   
    Y_train = np.log1p(df_train.median_house_value.values)
    Y_val = np.log1p(df_val.median_house_value.values)
    Y_test = np.log1p(df_test.median_house_value.values)

    df_train.drop(['median_house_value'], axis=1)
    df_val.drop(['median_house_value'],axis=1)
    df_test.drop(['median_house_value'],axis=1)

    X_train=prepare_X(df_train, "zero")
    X_val=prepare_X(df_val,"zero")
    

    w_0, w = train_linear_regression(X_train, Y_train)
    y_pred = w_0 + X_val.dot(w)
    p=round(rmse(Y_val,y_pred),2)

    seed_output[val]=p

print(seed_output)
print(min(seed_output.values()))


# %% [markdown]
# Question 5
# 
# What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
# Round the result to 3 decimal digits (round(std, 3))
# 
#    Note: Standard deviation shows how different the values are. If it's low, then all values are approximately the same. 
#          If it's high, the values are different. If standard deviation of scores is low, then our model is stable.
# 
# Options: [0.16,0.00005,0.005,0.15555]
# 

# %%
print(round(np.std(list(seed_output.values())),3))

# %% [markdown]
# STEP 5. 
# 
#    Split the dataset like previously, use seed 9.
#    Combine train and validation datasets.
#    Fill the missing values with 0 and train a model with r=0.001.

# %% [markdown]
# Question 6
# 
# What's the RMSE on the test dataset?
# 
# Options: [0.35,0.135,0.450,0.245]
# 

# %%

    np.random.seed(9)

    # let evaluate number of observation of each split
    
    test_n = int(0.2 * n)
    train_n = n - test_n
    


    #shuffle the dataset
    idx = np.arange(n)
    np.random.shuffle(idx)

    house_data_shuffled = house_data.iloc[idx]

    #final split
    df_train = house_data_shuffled.iloc[:train_n].copy()
    df_test = house_data_shuffled.iloc[train_n:].copy()

    #let's split predictors and outcome
   
    Y_train = np.log1p(df_train.median_house_value.values)
    Y_test = np.log1p(df_test.median_house_value.values)

    df_train.drop(['median_house_value'],axis=1)
    df_test.drop(['median_house_value'],axis=1)

    X_train=prepare_X(df_train, "zero")
    X_test=prepare_X(df_test,"zero")
    

    w_0, w = train_linear_regression_reg(X_train, Y_train, r=0.001)
       
    y_pred=w_0+X_test.dot(w)
    print('test:',rmse(Y_test, y_pred))


