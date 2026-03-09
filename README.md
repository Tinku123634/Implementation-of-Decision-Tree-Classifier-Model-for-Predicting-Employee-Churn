# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and import the required libraries for data processing and machine learning

2.Preprocess the data by converting categorical values into numerical form and separating features (X) and target variable (y).

3.Split the dataset into training and testing sets and train the Decision Tree Classifier using the training data.

4.Predict and evaluate the model performance using the testing data and calculate accuracy, confusion matrix, and classification report.

## Program:
```

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = {
    'satisfaction_level':[0.38,0.80,0.11,0.72,0.37,0.41,0.10,0.92,0.89,0.42],
    'last_evaluation':[0.53,0.86,0.88,0.87,0.52,0.50,0.77,0.85,0.90,0.62],
    'number_project':[2,5,7,5,2,2,6,5,5,2],
    'average_montly_hours':[157,262,272,223,159,153,247,259,224,142],
    'time_spend_company':[3,6,4,5,3,3,4,5,5,3],
    'salary':['low','medium','medium','low','low','low','low','medium','medium','low'],
    'left':[1,0,1,0,1,0,1,0,0,1]
}

df = pd.DataFrame(data)


le = LabelEncoder()
df['salary'] = le.fit_transform(df['salary'])


X = df.drop('left', axis=1)
y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, pred))
```

## Output:


<img width="594" height="320" alt="Screenshot 2026-03-09 100551" src="https://github.com/user-attachments/assets/3a8968f2-ceda-4aba-8ccf-cbab09341987" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
