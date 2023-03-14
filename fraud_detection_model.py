
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

print(corr_top_features)

# corr_train=X_train_standardized
# fraud_df.columns
standardized_cols=[col for col in fraud_df.columns if "V" in col or "Amount" in col]
fraud_df_subset=fraud_df[standardized_cols]
rs=RobustScaler()
fraud_df_subset_scaled=rs.fit_transform(fraud_df_subset)
fraud_df_standardized=fraud_df
fraud_df_standardized[standardized_cols]=fraud_df_subset_scaled


class_nf = fraud_df_standardized[fraud_df_standardized['Class'] == 0]
class_f = fraud_df_standardized[fraud_df_standardized['Class'] == 1]
for feature in corr_top_features:
    sb.boxplot(data=[class_nf[feature], class_f[feature]])
    plt.title("Fraud Class by "+feature)
    plt.xticks([0,1], ["Not fraud", "Fraud"])
    plt.ylim(-25,5)
    plt.show()



############################################################
# Step 4. Baseline Model Development
############################################################

model_scores={}
####MODEL TYPE: DECISION TREE
##Now time to train a decision tree model
##Not going to do any cross validation yet until I work on tuning the hyperparameters using grid search
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_standardized[corr_top_features], y_train)

y_pred = decision_tree.predict(X_test_standardized[corr_top_features])
model_df = pd.DataFrame({'Real Values':y_test, 'Decision Tree Predicted Values':y_pred})

# The score method returns the accuracy of the model
score = decision_tree.score(X_test_standardized[corr_top_features], y_test)
model_scores['Decision Tree']={'Accuracy':score}
print(score)


####MODEL TYPE: Logistic Regression
logistic=LogisticRegression()
logistic.fit(X_train_standardized[corr_top_features], y_train)

y_pred = logistic.predict(X_test_standardized[corr_top_features])
model_df['Logistic Predicted Values']=y_pred
score=logistic.score(X_test_standardized[corr_top_features],y_test)
model_scores['Logistic Regression']={'Accuracy':score}
print(score)

##Write Function
def CV_report(n_folds,model):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    scores_accuracy=cross_val_score(model,X,y,cv=tscv)
    scores_precision=cross_val_score(model,X,y,cv=tscv,scoring='precision')
    scores_recall=cross_val_score(model,X,y,cv=tscv,scoring='recall')
    scores_f1=cross_val_score(model,X,y,cv=tscv,scoring='f1')
    performance_metrics={'Accuracy':
                         {'Mean':scores_accuracy.mean(),
                          'Standard Deviation':scores_accuracy.std()},
                        'Precision':
                         {'Mean':scores_precision.mean(),
                          'Standard Deviation':scores_precision.std()},
                        'Recall':
                         {'Mean':scores_recall.mean(),
                          'Standard Deviation':scores_recall.std()},
                        'F1':
                         {'Mean':scores_f1.mean(),
                         'Standard Deviation':scores_f1.std()}
                                  }
    return performance_metrics


##Decision Tree Cross Validation Report
cv_decision_tree=DecisionTreeClassifier()
d_tree_report=CV_report(5,cv_decision_tree)
print(d_tree_report)
##Logistic Regression Cross Validation Report
cv_logistic=LogisticRegression()
logistic_report=CV_report(5,cv_logistic)
print(logistic_report)

###setting up performance charts
accuracy_numbers=[d_tree_report['Accuracy']['Mean'],logistic_report['Accuracy']['Mean']]
precision_numbers=[d_tree_report['Precision']['Mean'],logistic_report['Precision']['Mean']]
recall_numbers=[d_tree_report['Recall']['Mean'],logistic_report['Recall']['Mean']]
F1_numbers=[d_tree_report['F1']['Mean'],logistic_report['F1']['Mean']]

model_types = ('Decision Tree','Logistic Regression')
performance_metrics = {
    'Accuracy': accuracy_numbers,
    'Precision': precision_numbers,
    'Recall': recall_numbers,
    'F1': F1_numbers
}
x = np.arange(len(model_types))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

plt.figure(1)
fig, ax = plt.subplots(layout='constrained')

for attribute, value in performance_metrics.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, value, width, label=attribute)
    ax.bar_label(rects, padding=4)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance Metrics Value')
ax.set_title('Model Performance by Model Types')
ax.set_xticks(x + width, model_types)
ax.legend(loc='upper left', ncols=4)
ax.set_ylim(0, 1.2)

plt.show()
plt.savefig('Model Performance by Model Types.png')

###Explaining feature importance
fig2,axes2=plt.subplots(nrows=1,ncols=2)
explainer=shap.TreeExplainer(decision_tree)
# shap_values=explainer.shap_values(X_test_standardized[corr_top_features])
shap.summary_plot(X_test_standardized[corr_top_features], y_train)
plt.tight_layout()
plt.show()
# axes.shap_summary(shap_values)
#
# print(shap_values)
#
# shap_summary=
# shap_summary.show()
# for i in range(len(shap_values)):
#     shap.force_plot(explainer.expected_value, shap_values[i], X_test_standardized[corr_top_features].iloc[i,:], feature_names = X_test_standardized[corr_top_features].columns)