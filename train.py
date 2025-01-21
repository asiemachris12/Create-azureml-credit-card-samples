# %%
import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# %%
# load the data
credit_df = pd.read_csv(
    "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
    header=1,
    index_col=0,
)

train_df, test_df = train_test_split(
    credit_df,
    test_size=0.25,
)

# %%
# Extracting the label column
y_train = train_df.pop("default payment next month")

# convert the dataframe values to array
X_train = train_df.values

# Extracting the label column
y_test = test_df.pop("default payment next month")

# convert the dataframe values to array
X_test = test_df.values

# %%
# set name for logging
mlflow.set_experiment("Develop on cloud tutorial")
# enable autologging with MLflow
mlflow.sklearn.autolog()

# %%
# Train Gradient Boosting Classifier
print(f"Training with data of shape {X_train.shape}")

mlflow.start_run()
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
# Stop logging for this model
mlflow.end_run()

# %%
# Train  AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

print(f"Training with data of shape {X_train.shape}")

mlflow.start_run()
ada = AdaBoostClassifier()

ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)

print(classification_report(y_test, y_pred))
# Stop logging for this model
mlflow.end_run()


