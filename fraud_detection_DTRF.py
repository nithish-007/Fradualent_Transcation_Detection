#Basic libraries
import pandas as pd
import numpy as np

#Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Ignoring ununnecessary warnings
import warnings
warnings.filterwarnings("ignore")

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pd.set_option('display.max_columns',29)

data = pd.read_csv("/home/nithish07/Drive-D/fraud_Detection_DTRF/dataset/dataset_f.csv")

print(data.head())

# No. of Legit and Non Legit(fraud) transcations

legit = len(data[data.isFraud == 0])
fraud = len(data[data.isFraud == 1])
legit_percent = (legit / (fraud + legit)) * 100
fraud_percent = (fraud / (fraud + legit)) * 100
print("Number of Legit transactions: ", legit)
print("Number of Fraud transactions: ", fraud)
print("Percentage of Legit transactions: {:.4f} %".format(legit_percent))
print("Percentage of Fraud transactions: {:.4f} %".format(fraud_percent))

plt.figure(figsize=(5,10))
labels = ["Legit", "Fraud"]
count_classes = data.value_counts(data['isFraud'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()

#preprocessing libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# find uniqueness in all the columns

for i in data.columns:
    print('Unique Values in "{}":\n'.format(i),data[i].unique(), "\n\n")


#creating a copy of original dataset to train and test models

new_df=data.copy()
new_df.head()

# Renaming column names for convenience

new_df.columns = ['Transaction_Hours','Type','Transaction_Amt','Sender','Sender_Bal_b4','Sender_Bal_After','Receiver','Receiver_Bal_b4','Receiver_Bal_After','isFraud','isFlaggedFraud']

# Features with numerical value

features = ['Transaction_Amt','Sender_Bal_b4', 'Transaction_Hours','Sender_Bal_After','Receiver_Bal_b4','Receiver_Bal_After']

# Plotting all numerical features for distrubution check

plt.figure(figsize=(20,8))
warnings.filterwarnings('ignore')

for feature in features:
  plt.subplot(2,3,features.index(feature)+1)
  sns.distplot(new_df[feature])
plt.show()

# Plotting boxplot to find the outliers
plt.figure(figsize=(20,8))
warnings.filterwarnings('ignore')

for feature in features:
  plt.subplot(2,3,features.index(feature)+1)
  sns.boxplot(new_df[feature])
plt.show()

# Checking how many attributes are dtype: object

objList = new_df.select_dtypes(include = "object").columns
print (objList)

#Label Encoding for object to numeric conversion

le = LabelEncoder()

for feat in objList:
    new_df[feat] = le.fit_transform(new_df[feat].astype(str))

print (new_df.info())

new_df['Actual_amount_orig(Sender)'] = new_df.apply(lambda x: x['Sender_Bal_b4'] - x['Sender_Bal_After'],axis=1)
new_df['Actual_amount_dest(Receiver)'] = new_df.apply(lambda x: x['Receiver_Bal_b4'] - x['Receiver_Bal_After'],axis=1)
new_df['TransactionPath'] = new_df.apply(lambda x: x['Sender'] + x['Receiver'],axis=1)

#Dropping columns
new_df = new_df.drop(['Transaction_Hours','Sender','Sender_Bal_b4','Sender_Bal_After','Receiver','Receiver_Bal_b4','Receiver_Bal_After'],axis=1)

for i in new_df.columns:
    print('Unique Values in "{}":\n'.format(i),new_df[i].unique(), "\n\n")

features =['Transaction_Amt','Actual_amount_orig(Sender)','Actual_amount_dest(Receiver)','TransactionPath']

# Removing outliers using Inter-Quartile Range (IQR) proximity rule  

for feature in features:
  percentile25 = new_df[feature].quantile(0.25)
  percentile75 = new_df[feature].quantile(0.75)
  percentile50 = new_df[feature].quantile(0.50)
  iqr = percentile75 - percentile25
  upper_limit = percentile75 + 1.5 * iqr
  lower_limit = percentile25 - 1.5 * iqr
  new_df[feature] = np.where(new_df[feature] > upper_limit,new_df[feature].mean(),new_df[feature])
  new_df[feature] = np.where(new_df[feature] < lower_limit,new_df[feature].mean(),new_df[feature])

  # Correlation between features and target 

print(new_df.corr())

# Plotting after removing outliers

plt.figure(figsize=(20,8))
warnings.filterwarnings('ignore')
for feature in features:
  plt.subplot(2,3,features.index(feature)+1)
  sns.distplot(new_df[feature])
plt.show()

plt.figure(figsize=(20,8))
warnings.filterwarnings('ignore')
for feature in features:
  plt.subplot(2,3,features.index(feature)+1)
  sns.boxplot(new_df[feature])
plt.show()

print(new_df.shape)

print(new_df.isFraud.value_counts())

# we have to sample the data to retify the imbalance in the dataset

x = new_df.drop(["isFraud","isFlaggedFraud","Type"], axis= 1)

y = new_df["isFraud"]

# scaling

names = x.columns
names

scaler = StandardScaler()

for i in names:
	x[i] = scaler.fit_transform(x[i].values.reshape(-1, 1))
        
x = pd.DataFrame(x,columns = names)

x["isFlaggedFraud"] = new_df.isFlaggedFraud
x["Type"] = new_df.Type

print(x.shape,y.shape)

# Importing library to split the data into training part and testing part.
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

#ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC                          # Support vector machine model
from sklearn.neighbors import KNeighborsClassifier


# Importing metrics used for evaluation of our models
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Hyperparameter tuner and Cross Validation
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=46) 

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

print(len(y_train[y_train==1]), len(y_train[y_train==0]), y_train.shape)

print(len(y_test[y_test==1]), len(y_test[y_test==0]), y_test.shape)

rus = RandomUnderSampler(sampling_strategy='majority')
X_train_down,y_train_down = rus.fit_resample(X_train, y_train)
print(len(y_train_down[y_train_down==0]), len(y_train_down[y_train_down==1]))
print(len(X_train_down))

smote = SMOTETomek()
x_b,y_b = smote.fit_resample(X_train, y_train)
print(y.value_counts())
print(y_b.value_counts())
