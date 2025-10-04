import pandas as pd
import pickle

model = pickle.load(open("fraud_transaction_detection.pkl", "rb"))

data = pd.read_csv("Fraud.csv")
data = pd.get_dummies(data, columns=["type"])
data_col = []

for i in data.columns:
    if pd.api.types.is_numeric_dtype(data[i]) and i != "isFraud":
        data_col.append(i)

values = []
count = 0
while True:

    if data_col[count] == "isFlaggedFraud" or data_col[count] == "type_CASH_IN" or data_col[count] == 'type_CASH_OUT' or data_col[count] == "type_DEBIT" or data_col[count] == 'type_PAYMENT' or data_col[count] == "type_TRANSFER":
        inputs = input(f"Enter the {data_col[count]} values, Please enter '0' for 'No' and '1' for 'Yes': ")

        if inputs == '0' or inputs == '1':
            values.append(int(inputs))
            count += 1

        else:
            print("Please enter '0' for 'No' and '1' for 'Yes'!")

        if count == len(data_col):
            break

    if data_col[count] != "isFlaggedFraud" and data_col[count] != 'type_CASH_IN' and data_col[count] != 'type_CASH_OUT' and data_col[count] != "type_DEBIT" and data_col[count] != 'type_PAYMENT' and data_col[count] != "type_TRANSFER":
        inputs = input(f"Enter the {data_col[count]} values, Please enter numbers not to use Alphabets: ")
        try:
            if inputs.isnumeric():
                values.append(int(inputs))
                count += 1

        except ValueError:
            print("Please enter only the number!")

        if count == len(data_col):
            break

# values = [743,850002.52,850002.52,0.0,6510099.11,7360101.63,0,0,0,0,1,0]
values = [list(values)]
predicts = model.predict(values)
print(predicts[0])