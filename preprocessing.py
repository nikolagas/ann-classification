# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Step 2: Load the dataset and check for missing values
df = pd.read_csv('C:/Users/Nik/PycharmProjects/ANN/venv/as1-bank.csv')
string_columns = ["default", "housing", "loan"]
df = df.drop(columns=string_columns)
print(df.isnull().sum())

# Step 3: Data preprocessing
categorical_columns = ["marital", "education", "contact", "poutcome"]
numerical_columns = ["age", "balance", "duration", "campaign", "pdays", "previous"]

# 3.1: Encode target column to numerical values
le_y = LabelEncoder()
y = le_y.fit_transform(df.iloc[:, -1].values)

# 3.2: Scale numerical columns
scaler = StandardScaler()
X_numerical = scaler.fit_transform(df[numerical_columns].values)

# 3.3: One-hot encode categorical columns to floats
ohe = OneHotEncoder()
ohe.fit(df[categorical_columns])
X_categorical = ohe.transform(df[categorical_columns]).toarray()

# 3.4: Concatenate numerical and categorical columns
X = np.concatenate((X_categorical, X_numerical), axis=1)
print("Shape of concatenated data: ", X.shape)

# 3.5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)