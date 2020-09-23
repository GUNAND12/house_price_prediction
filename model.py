# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

df=pd.read_csv("train.csv")

# MSSubClass,
# YrSold,MoSold
# convert numeric variable  to object (categorical) variable
selected_discrete_features=["MSSubClass","YrSold","MoSold"]
for i in selected_discrete_features:
    df[i] = df[i].astype('object')

#missing value
nan_features=df.isnull().sum()[df.isnull().sum()>0].index
nan_percentage_val=df.isnull().sum()[df.isnull().sum()>0].values/len(df)
nan_percentage_val=[round(i,3)*100 for i in nan_percentage_val]
missing_data=dict(zip(nan_features,nan_percentage_val))
sorted_missing_data={k: v for k, v in sorted(missing_data.items(), key=lambda item: item[1],reverse=True)}
missing_df=pd.DataFrame(list(sorted_missing_data.items()),columns=["Features_name","Missing_percentage"])

# features such as Alley,FireplaceQu,PoolQC,Fence,MiscFeature have more than 45% missing value
df=df.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],1)
df=df.drop("Id",1) #this is unique for all the datapoints

# Replacing missing data with 0 Since No garage = no cars in such garage.
# filling null values of columns GarageType, GarageFinish, GarageQual and GarageCond 
# None indicate there is no Garage for the house 
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
df["GarageType"] = df["GarageType"].fillna("none")
df["GarageFinish"] = df["GarageFinish"].fillna("none")
df["GarageQual"] = df["GarageQual"].fillna("none")
df["GarageCond"] = df["GarageCond"].fillna("none")

# by the median LotFrontage of all the neighborhood
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# filling null values of columns MasVnrType
# for having no masonry veneer for these houses
# extra space outside the house
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
df["MasVnrType"] = df["MasVnrType"].fillna("none")


# It has one NA value. Since this feature has mostly 'SBrkr', 
# so we are filling missing values with "SBrkr" using mode
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])


# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2
# replacing missing values with none for having on basement
df["BsmtQual"] = df["BsmtQual"].fillna("none")
df["BsmtCond"] = df["BsmtCond"].fillna("none")
df["BsmtExposure"] = df["BsmtExposure"].fillna("none")
df["BsmtFinType1"] = df["BsmtFinType1"].fillna("none")
df["BsmtFinType2"] = df["BsmtFinType2"].fillna("none")


# correlation bwtween continuous features
# generate the correlation matrix
corr =  num_df.corr()
plt.figure(figsize=(20,8))
sns.heatmap(corr,annot=True)


# TotRmsAbvGrd and GrLivArea (0.83) (drop either one of them)
# GarageCars and GarageArea (0.88)
# 1stFlrSF and TtalBsmtSF (0.82)
df=df.drop(["TotRmsAbvGrd","GarageArea","1stFlrSF"],1)

# Utilities,Street,Condition2,RoofMatl,Heating are the features which has very low variance
# drop this features
df=df.drop(["Utilities","Street","Condition2","RoofMatl","Heating"],axis=1)


#outliers
# remove outliers w.r.t target variable

q1=df["SalePrice"].quantile(0.25)
q3=df["SalePrice"].quantile(0.75)
iqr=q3-q1
ub=q3+1.5*iqr
lb=q1-1.5*iqr
print("upper_bound:",ub)
print("lower_bound:",lb)

df=df[~((df["SalePrice"]<lb) | (df["SalePrice"]>ub))]
df.reset_index(drop=True,inplace=True)

# encoding
# Excellent-5,Good-4,Typical/Average-3,Fair-2,Poor-1
ordinal_encode={"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"none":0}

df["ExterQual"]=df["ExterQual"].map(ordinal_encode)
df["ExterCond"]=df["ExterCond"].map(ordinal_encode)
df["BsmtQual"]=df["BsmtQual"].map(ordinal_encode)
df["BsmtCond"]=df["BsmtCond"].map(ordinal_encode)
df["HeatingQC"]=df["HeatingQC"].map(ordinal_encode)
df["KitchenQual"]=df["KitchenQual"].map(ordinal_encode)
df["GarageQual"]=df["GarageQual"].map(ordinal_encode)
df["GarageCond"]=df["GarageCond"].map(ordinal_encode)


# Good Exposure-4, Average Exposure-3, Mimimum Exposure-2, No Exposure-1, No Basement-0 
ordinal_encode_2={"Gd":4,"Av":3,"Mn":2,"No":1,"none":0}
df["BsmtExposure"]=df["BsmtExposure"].map(ordinal_encode_2)

# Good Living Quarters-6, Average Living Quarters-5, Below Average Living Quarters-4, Average Rec Room-3
# Low Quality-2, Unfinshed-1, No Basement-0
ordinal_encode_3={"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"none":0}
df["BsmtFinType1"]=df["BsmtFinType1"].map(ordinal_encode_3)
df["BsmtFinType2"]=df["BsmtFinType2"].map(ordinal_encode_3)

cat_df=df.select_dtypes(exclude=np.number)
df.drop(cat_df.columns,1,inplace=True)
df.shape


X=df.drop("SalePrice",1)
y=df["SalePrice"]
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
s=pd.Series(index=X.columns,data=model.feature_importances_)
s.sort_values(ascending=False)
 
df_final=df[["OverallQual","GrLivArea","KitchenQual","GarageCars","Fireplaces","YearBuilt"]]
df_final.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_final, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 10, verbose=2, random_state=0, n_jobs = -1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
y_pred=rf_random.predict(X_test)
sns.distplot(y_test-y_pred)
plt.scatter(y_test,y_pred)

# open a file, where you ant to store the data
file = open('rf_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)
