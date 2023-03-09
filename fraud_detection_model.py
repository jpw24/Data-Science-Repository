
############################################################
# STEP 1
############################################################
####Import required python packages
from collections import Counter
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sb

import numpy as np
import pandas as pd

import sklearn as sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split,KFold,TimeSeriesSplit,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,RobustScaler

import shap

# Oversampling and under sampling
# from imblearn.over_sampling import RandomOverSampler, SMOTE
# from imblearn.under_sampling import RandomUnderSampler, NearMiss

import scipy.stats as stats


##read in the data set using pandas
fraud_df=pd.read_csv(r"Scratch Data Sets, Articles and Ideas/Credit Card Fraud.csv")
fraud_df.set_index('Time',inplace=True)
fraud_df.sort_index(inplace=True)

y=fraud_df['Class']
X=fraud_df.drop(labels='Class',axis=1)

############################################################
# Step 2. Target, Predictors, and Train-Test Split
############################################################

tss=TimeSeriesSplit(n_splits=2)

train_split_indices,test_split_indices=tss.split(X)
X_train,X_test=X.iloc[train_split_indices[1],:],X.iloc[test_split_indices[1],:]
y_train,y_test=y.iloc[train_split_indices[1]],y.iloc[test_split_indices[1]]

y_train.groupby('Time').mean().plot()
y_test.groupby('Time').mean().plot()

##TIME SERIES CROSS VALIDATION
cross_val_tss=TimeSeriesSplit(n_splits=5)

##these are the 5 folds we created based on the time
for i, (train_index, test_index) in enumerate(cross_val_tss.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")


############################################################
# Step 3. Exploratory Data Analysis
############################################################

fraud_df.describe()
##Checking for missing values
fraud_df.isnull().values.any()

fraud_df.shape
fraud_df.columns

fraud_df.corr()
dataplot= sb.heatmap(fraud_df.corr(),cmap="YlGnBu",cbar=False)
plt.show()


##robust scaler
robust_scaler_train=RobustScaler()
robust_scaler_test=RobustScaler()

X_train_standardized=robust_scaler_train.fit_transform(X_train)
X_train_standardized = pd.DataFrame(X_train_standardized, columns=X_train.columns)

X_test_standardized=robust_scaler_test.fit_transform(X_test)
X_test_standardized = pd.DataFrame(X_test_standardized, columns=X_test.columns)



point_bi_serial_list=X_train_standardized
point_bi_serial_threshold = .2
pointbiserialr=stats.pointbiserialr
corr_data=pd.DataFrame()
for i in point_bi_serial_list:
    pbc=pointbiserialr(y_train,X_train_standardized[i])
    corr_temp_data=[[i,pbc.correlation,"point_bi_serial"]]
    corr_temp_df=pd.DataFrame(corr_temp_data,columns=['Feature','Correlation','Correlation_Type'])
    corr_data=corr_data.append(corr_temp_df)

# Filter NA and sort based on absolute correlation
corr_data = corr_data.iloc[corr_data.Correlation.abs().argsort()]
corr_data = corr_data[corr_data['Correlation'].notna()]
corr_data = corr_data.loc[corr_data['Correlation'] != 1]

# Add thresholds

# initialize list of lists
data = [['point_bi_serial', point_bi_serial_threshold]]
threshold_df=pd.DataFrame(data,columns=["Correlation_Type","Threshold"])
corr_data=pd.merge(corr_data,threshold_df,on=["Correlation_Type"],how="left")
corr_data2 = corr_data.loc[corr_data['Correlation'].abs() > corr_data['Threshold']]
corr_top_features = corr_data2['Feature'].tolist()

corr_top_features

# corr_train=X_train_standardized
# fraud_df.columns
standardized_cols=[col for col in fraud_df.columns if "V" in col or "Amount" in col]
fraud_df_subset=fraud_df[standardized_cols]
rs=RobustScaler()
fraud_df_subset_scaled=rs.fit_transform(fraud_df_subset)
fraud_df_standardized=fraud_df

class_nf = fraud_df_standardized[fraud_df_standardized['Class'] == 0]
class_f = fraud_df_standardized[fraud_df_standardized['Class'] == 1]
for feature in corr_top_features:
    sb.boxplot(data=[class_nf[feature], class_f[feature]])
    plt.title("Fraud Class by "+feature)
    plt.xticks([0,1], ["Not fraud", "Fraud"])
    plt.ylim(-25,5)
    plt.show()
fraud_df_standardized[standardized_cols]=fraud_df_subset_scaled


##Add more code later - condensed version only!!