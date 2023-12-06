# lung-cacer-detection-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing

from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv('/content/drive/MyDrive/survey lung cancer.csv')

data[data.duplicated()]

data.drop_duplicates()

data['GENDER']=data['GENDER'].map({'M':1,'F':0})

data['LUNG_CANCER']=data['LUNG_CANCER'].map({'YES':1,'NO':0})

data.shape

data.info()

data.columns

data.head()

data.tail()

data.describe()

data.isnull().sum() #checking for total null values

data["LUNG_CANCER"].value_counts()

data["SMOKING"].value_counts()

# See the min, max, mean values
print('The highest hemoglobin was of:',data['SMOKING'].max())
print('The lowest hemoglobin was of:',data['SMOKING'].min())
print('The average hemoglobin in the data:',data['SMOKING'].mean())

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['LUNG_CANCER']==1]['SMOKING'].value_counts()

ax1.hist(data_len,color='red')
ax1.set_title('Having CANCER')

data_len=data[data['LUNG_CANCER']==0]['SMOKING'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('NOT Having CANCER')

fig.suptitle('LUNG CANCER')
plt.show()

plt.plot(data['SMOKING'])
plt.xlabel("Smoking")
plt.ylabel("Levels")
plt.title("Smoking Line Plot")
plt.show()

data[1:5]

from sklearn import preprocessing
import pandas as pd

d = preprocessing.normalize(data.iloc[:,1:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY"])
scaled_df.head()

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression

train,test=train_test_split(data,test_size=0.2,random_state=42,stratify=data['LUNG_CANCER'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=data[data.columns[:-1]]
Y=data['LUNG_CANCER']
len(train_X), len(train_Y), len(test_X), len(test_Y)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LogisticRegression(C=0.1, penalty='l2')
model.fit(train_X, train_Y)
predictions = model.predict(test_X)
print('The accuracy of Logistic Regression is:',metrics.accuracy_score(test_Y,predictions))
report = classification_report(test_Y, predictions)
print("Classification Report:\n", report)

import matplotlib.pyplot as plt
import numpy as np

# Replace these values with your actual scores
precision = [0.00, 0.97]
recall = [0.00, 1.00]
f1_score = [0.00, 0.98]

labels = ['Class 0', 'Class 1']

# Plotting the bar chart
width = 0.2
x = np.arange(len(labels))

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Adding labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Logistic Regression Model Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display the plot
plt.show()

model = LogisticRegression(C=0.1, penalty='l2')
model.fit(train_X, train_Y)
predictions = model.predict(test_X)
mse = mean_squared_error(test_Y, predictions)
rmse = mean_squared_error(test_Y, predictions, squared=False)
mae = mean_absolute_error(test_Y, predictions)
r2 = r2_score(test_Y, predictions)
print('The accuracy of Logistic Regression is:',metrics.accuracy_score(test_Y,predictions))
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('Mean Absolute Error:', mae)
print('R-squared:',r2)
