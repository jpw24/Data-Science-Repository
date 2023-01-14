import pandas as pd
import sklearn as sklearn
from sklearn.model_selection import train_test_split,TimeSeriesSplit
import matplotlib
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

for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

y_train.groupby('Time').mean().plot()
y_test.groupby('Time').mean().plot()


###BEGIN FEATURE SELECTION/ENGINEERING

#getting correlation matrix; use point_bi_serial_list because target is categorical and features are continuous
point_bi_serial_list=X_train
point_bi_serial_threshold = .3
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