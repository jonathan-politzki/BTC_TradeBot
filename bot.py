
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
df.head()

x = df.shape

print(x)