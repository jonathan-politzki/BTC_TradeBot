
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score  # Example of specific metric import from scikit-learn
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore') 

# now we load the data, thank you very much copilot, shit is good
# OHLC(‘Open’, ‘High’, ‘Low’, ‘Close’) data from 17th July 2014 to 29th December 2022 which is for 8 years for the Bitcoin price.
# im going to load the data as a flat file for now and try to do an API later

df = pd.read_csv('BTC-Daily.csv')
df['date'] = pd.to_datetime(df['date'])

# 17th July 2014 to 29th December 2022

start_date = '2014-07-17'
end_date = '2022-12-29'

# filtered data frame

filtered_df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date), ['date', 'open', 'high', 'low', 'close']]

print(filtered_df)

x = filtered_df.head()
y = filtered_df.shape

print(x,y)

filtered_df.describe()
filtered_df.info()

# plotting stuff

plt.figure(figsize=(15, 5))
plt.plot(filtered_df['date'], filtered_df['close'])
plt.title('Bitcoin Close Price', fontsize=15)
plt.ylabel('Price in Dollars ($)')
plt.xlabel('Date')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility if needed
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.show()

# checking for null values if any are present in the dataset

filtered_df.isnull().sum()

# now drawing a distribution plot for the continuous features in the dataset

# frankly i dont really want to do this it just shows the distribution of the data and its obviously concentrated around 0
# ill enumerate features and do it anyway

features = ['open', 'high', 'low', 'close']

# I also want to see what happens when I plot multiple graphs. I think you have to exit out of each one at a time

#plt.subplots(figsize=(20,10))
#for i, col in enumerate(features):
#  plt.subplot(2,2,i+1)
#  sb.distplot(filtered_df[col])
#plt.show()

# feature engineering
splitted = filtered_df['date'].dt.strftime('%Y-%m-%d').str.split('-', expand=True)
filtered_df['year'] = splitted[0].astype(int)
filtered_df['month'] = splitted[1].astype(int)
filtered_df['day'] = splitted[2].astype(int)
filtered_df['is_quarter_end'] = np.where(filtered_df['month']%3==0,1,0)
filtered_df['open-close']  = filtered_df['open'] - filtered_df['close']
filtered_df['low-high']  = filtered_df['low'] - filtered_df['high']

# this seems very important. binary classification of the days the price went up. I'm sure this is where you can do a lot of fun stuff. Daily, monthly. Momentum. New features. etc.

filtered_df['target'] = np.where(filtered_df['close'].shift(-1) > filtered_df['close'], 1, 0)

print(filtered_df.head())

# correlation mapping shows us that the date and OHLC are super correlated obviously. Test this later.

#sb.heatmap(filtered_df.corr() > 0.9, annot=True, cbar=False)
#plt.show()

# now we normalize the data

model_features = filtered_df[['open-close', 'low-high', 'is_quarter_end']]
target = filtered_df['target']

scaler = StandardScaler()
model_features = scaler.fit_transform(model_features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    model_features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

# now we are testing the models and stuff

models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
 
for i in range(3):
  models[i].fit(X_train, Y_train)
 
  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()

# lastly, a confusion matrix

y_pred = models[0].predict(X_valid)

# Create ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay.from_predictions(y_true=Y_valid, y_pred=y_pred)

# Plot confusion matrix
disp.plot()
plt.title('Confusion Matrix')
plt.show()