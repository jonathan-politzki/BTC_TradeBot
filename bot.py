
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

splitted = df['date'].str.split('-', expand=True)
df['year'] = splitted[0].astype(int)
df['month'] = splitted[1].astype(int)
df['day'] = splitted[2].astype(int)

df.head()