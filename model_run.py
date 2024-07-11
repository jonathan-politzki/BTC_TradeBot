#import from other files
from some_data import load_data, filtered_data, updated_data
from feature_engineering import *
from plotting import *
from import_warnings import ignore_warnings

#libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score  # Example of specific metric import from scikit-learn
from xgboost import XGBClassifier

# Now you can call ignore_warnings to suppress warnings
ignore_warnings()

# Load the dataset and year groups
df, start_date, end_date, new_start_date = load_data()

# Filter the data for the data sets we want to use

# earlier, larger dataset
filtered_df = filtered_data(df, start_date, end_date)

# more recent, smaller dataset
updated_df = updated_data(df, new_start_date)

# plotting both older and newer data
plot_data(filtered_df)
plot_data(updated_df)

# checking for null values if any are present in the dataset

filtered_df.isnull().sum()
updated_df.isnull().sum()

# create features 
filtered_df_features()
updated_df_features()

# create target variable
create_target_var(df)

# correlation mapping shows us that the date and OHLC are super correlated obviously. Test this later.
correlation_mapping_filtered()

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

