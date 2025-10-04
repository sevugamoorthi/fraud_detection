import pandas as pd

# dataset download link: https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0
data = pd.read_csv("Fraud.csv")
data = pd.get_dummies(data, columns=["type"], dtype=int)
data['isFraud'] = data['isFraud'].replace({0:"Not Fraud",1:"Fraud"})

# pd.set_option('display.max_columns', None)
# print(data.head())
# na = 0
# for i in data:
#     if pd.isna(i):
#         na += 1
# print(f"The dataset contains {na} missing values")

data_col = []
for i in data.columns:
    if pd.api.types.is_numeric_dtype(data[i]) and (i != "isFraud"):
        data_col.append(i)

# print(data_col)
x = data.loc[:,data_col]
y = data.loc[:, "isFraud"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train, y_train)

y_pred = lg.predict(x_test)

from sklearn.metrics import r2_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(lg.score(x_test,y_test))

import pickle
model = pickle.dump(lg,open("fraud_transaction_detection.pkl", "wb"))