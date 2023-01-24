import pandas as pd
import sklearn as sklearn
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Oversampling and under sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
from matplotlib import pyplot as plt
import scipy.stats as stats
pointbiserialr=stats.pointbiserialr


fraud_df=pd.read_csv(r"C:\Users\jimmyw\Documents\Data-Science-Repository\Potential Data Sets\Credit Card Fraud.csv")

fraud_df.head()
fraud_df.describe()

fraud_df.set_index('Time',inplace=True)
fraud_df.sort_index(inplace=True)

Y=fraud_df['Class']
X=fraud_df.drop(labels='Class',axis=1)

##TIME SERIES CROSS VALIDATION
tss=TimeSeriesSplit(n_splits=3)

##these are the 3 folds we created based on the time
for i, (train_index, test_index) in enumerate(tss.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

# y_train.groupby('Time').mean().plot()
# y_test.groupby('Time').mean().plot()


###BEGIN FEATURE SELECTION/ENGINEERING

#getting correlation matrix; use point_bi_serial_list because target is categorical and features are continuous
point_bi_serial_list=X_train
point_bi_serial_threshold = .2
corr_data=pd.DataFrame()
for i in point_bi_serial_list:
    pbc=pointbiserialr(y_train,X_train[i])
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

###features that are likely best are below:
print(corr_top_features)
corr_top_features_df = pd.DataFrame(corr_top_features, columns=['Feature'])
corr_top_features_df['Method'] = 'Correlation'
corr_data.tail(20)


####MODEL TYPE: DECISION TREE
##Now time to train a decision tree model
##Not going to do any cross validation yet until I work on tuning the hyperparameters using grid search
regressor = DecisionTreeClassifier()
regressor.fit(X_train[corr_top_features], y_train)

y_pred = regressor.predict(X_test[corr_top_features])
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})

# The score method returns the accuracy of the model
score = regressor.score(X_test[corr_top_features], y_test)
print(score)

# List of values to try for max_depth:
max_depth_range = list(range(1, 10))
# List to store the accuracy for each value of max_depth:
accuracy = []
precision=[]
recall=[]
f1=[]
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth=depth,
                                 random_state=0)
    clf.fit(X_train[corr_top_features], y_train)
    y_pred=clf.predict(X_test[corr_top_features])
    score = clf.score(X_test[corr_top_features], y_test)
    accuracy.append(score)
    precision.append(precision_score(y_test,y_pred))
    recall.append(recall_score(y_test,y_pred))
    f1.append(f1_score(y_test,y_pred))

max_f1=0
best_depth=0
for depth in max_depth_range:
    print("Depth")
    print(depth)
    print("Accuracy")
    print(accuracy[depth-1])
    print("Precision")
    print(precision[depth-1])
    print("Recall")
    print(recall[depth-1])
    print("F1")
    print(f1[depth-1])
    if f1[depth-1]>max_f1:
        best_depth=depth
        max_f1=f1[depth-1]
print("Best Depth")
print(best_depth)
print("Best F1")
print(max_f1)


##Plot on a graph
# plt.plot(max_depth_range,accuracy, label="Accuracy")
# plt.plot(max_depth_range,precision, label="Precision")
# plt.plot(max_depth_range,recall, label="Recall")
# plt.plot(max_depth_range,f1, label="F1")
# plt.legend()

# y_test.groupby('Time').mean().plot()

####NEED TO WORK ON SAMPLING FOR CLASS IMBALANCE:
##https://medium.com/p/8f63474ff8c7
####GOING TO USE THE SMOTE TECHNIQUE TO HANDLE THE CLASS IMBALANCE, MAY NOT BE NECESSARY FOR DECISION TREES


##create baseline random forest model

rf= RandomForestClassifier()
baseline_model = rf.fit(X_train[corr_top_features],y_train)
baseline_prediction=baseline_model.predict(X_test[corr_top_features])
print(classification_report(y_test,baseline_prediction))


##SMOTE CREATION
smote=SMOTE(random_state=34)
X_train_smote,y_train_smote=smote.fit_resample(X_train[corr_top_features],y_train)
print(sorted(Counter(y_train_smote).items()))

rf_smote=RandomForestClassifier()
smote_model=rf_smote.fit(X_train_smote,y_train_smote)
smote_prediction = smote_model.predict(X_test[corr_top_features])

print(classification_report(y_test,smote_prediction))

##GBM
gb_smote=GradientBoostingClassifier()
gb_smote_model=gb_smote.fit(X_train_smote,y_train_smote)
gb_smote_prediction=gb_smote_model.predict(X_test[corr_top_features])

##Logistic Regression
logistic_smote=LogisticRegression()
logistic_smote_model=logistic_smote.fit(X_train_smote,y_train_smote)
logistic_smote_prediction=logistic_smote_model.predict(X_test[corr_top_features])

print(classification_report(y_test,logistic_smote_prediction))
##K Nearest Neighbors

##


##


##NOTE that because we want fewer false negatives than false positives, we should prioritize RECALL

###READ NEXT:
####https://medium.com/illumination/the-ultimate-data-science-portfolio-f52a4b3a9934
####https://heliosaditya.medium.com/everything-i-needed-to-know-to-become-a-job-ready-data-scientist-3dcfe00f5c77


####MODEL TYPE: _____________________
