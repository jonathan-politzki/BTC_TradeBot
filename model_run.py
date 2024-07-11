
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score  # Example of specific metric import from scikit-learn
from xgboost import XGBClassifier

#ignore warnings
ignore_warnings()

# Load the dataset and year groups
load_data()

# Filter the data for the data sets we want to use

# earlier, larger dataset
filtered_data(df, start_date, end_date)

# more recent, smaller dataset
updated_data(df, new_start_date)

# plotting both older and newer data
plot_data(filtered_df)
plot_data(updated_df)

# checking for null values if any are present in the dataset

filtered_df.isnull().sum()
updated_df.isnull().sum()

# create features

features = ['open', 'high', 'low', 'close']

# feature engineering
splitted = filtered_df['date'].dt.strftime('%Y-%m-%d').str.split('-', expand=True)
filtered_df['year'] = splitted[0].astype(int)
filtered_df['month'] = splitted[1].astype(int)
filtered_df['day'] = splitted[2].astype(int)
filtered_df['is_quarter_end'] = np.where(filtered_df['month']%3==0,1,0)
filtered_df['open-close']  = filtered_df['open'] - filtered_df['close']
filtered_df['low-high']  = filtered_df['low'] - filtered_df['high']

splitted = updated_df['date'].dt.strftime('%Y-%m-%d').str.split('-', expand=True)
updated_df['year'] = splitted[0].astype(int)
updated_df['month'] = splitted[1].astype(int)
updated_df['day'] = splitted[2].astype(int)
updated_df['is_quarter_end'] = np.where(updated_df['month']%3==0,1,0)
updated_df['open-close']  = updated_df['open'] - updated_df['close']
updated_df['low-high']  = updated_df['low'] - updated_df['high']

# this seems very important. binary classification of the days the price went up. I'm sure this is where you can do a lot of fun stuff. Daily, monthly. Momentum. New features. etc.
# id like to try this model with an RNN too

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

# Define the models with their respective parameters
models = [
    LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=100,
        n_jobs=-1,
        random_state=42
    ),
    SVC(
        kernel='poly',
        degree=3,
        C=1.0,
        probability=True,
        gamma='scale',
        coef0=0.0,
        shrinking=True,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr',
        break_ties=False,
        random_state=42
    ),
    XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective='binary:logistic',
        booster='gbtree',
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
]

# Train the models
for i in range(3):
    models[i].fit(X_train, Y_train)
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
    print()

# Calculate predictions on the test set
y_pred = models[0].predict(X_valid)

# Create confusion matrix
cm = confusion_matrix(Y_valid, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sb.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

