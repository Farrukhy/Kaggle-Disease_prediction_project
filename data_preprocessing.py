This script handles:

Loading the dataset.
Checking for missing values both column-wise and row-wise.
Handling missing data using SimpleImputer with a strategy of mean imputation.
Label encoding the target variable (prognosis).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


data = pd.read_csv('/kaggle/input/disease-prediction-using-machine-learning/Training.csv')

missing_values_per_column = data.isnull().any()
print("Missing values per column:\n", missing_values_per_column)


X = data.drop(columns=['prognosis', 'Unnamed: 133'])
y = data['prognosis']


X_train, X_test_all, y_train, y_test_all = train_test_split(X, y, test_size=0.2, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_test_all, y_test_all, test_size=0.5, random_state=42)


imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
