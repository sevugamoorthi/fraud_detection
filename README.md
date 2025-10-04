# Fraud Transaction Detection using Logistic Regression

## Project Overview
This project demonstrates a machine learning-based **Fraud Transaction Detection** system using a Logistic Regression model. The system identifies fraudulent transactions in a dataset containing transaction details. The dataset is preprocessed, encoded, and used to train a predictive model capable of classifying transactions as **Fraud** or **Not Fraud**.

## Dataset
The dataset used for this project contains transaction records with various attributes such as transaction type, amount, and other numeric features.

**Download Link:**  
[Fraud Transaction Dataset](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0)

**Key Features:**
- `type`: Transaction type (categorical, one-hot encoded during preprocessing)
- `amount`, `oldbalanceOrg`, `newbalanceOrig`, etc.: Numeric features describing the transaction
- `isFraud`: Target variable indicating whether a transaction is fraudulent (1) or not (0)

## Project Structure
- `fraud_detection.py`: Main Python script containing data preprocessing, model training, and evaluation.
- `fraud_transaction_detection.pkl`: Serialized Logistic Regression model saved using `pickle`.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/sevugamoorthi/fraud_detection.git
cd fraud_detection/
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
scikit-learn
```

## Usage

1. Load the dataset and preprocess:
```python
import pandas as pd

data = pd.read_csv("Fraud.csv")
data = pd.get_dummies(data, columns=["type"], dtype=int)
data['isFraud'] = data['isFraud'].replace({0:"Not Fraud", 1:"Fraud"})
```

2. Prepare features and target:
```python
data_col = [i for i in data.columns if pd.api.types.is_numeric_dtype(data[i]) and i != "isFraud"]
x = data.loc[:, data_col]
y = data.loc[:, "isFraud"]
```

3. Split the dataset into training and test sets:
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
```

4. Train Logistic Regression model:
```python
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train, y_train)
```

5. Evaluate the model:
```python
from sklearn.metrics import confusion_matrix

y_pred = lg.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(lg.score(x_test, y_test))
```

6. Save the trained model:
```python
import pickle
pickle.dump(lg, open("fraud_transaction_detection.pkl", "wb"))
```

## Model Evaluation
- **Confusion Matrix:** Provides the count of true positives, true negatives, false positives, and false negatives.
- **Accuracy Score:** Logistic Regression accuracy on test data can be printed using `lg.score(x_test, y_test)`.

## Future Improvements
- Use more advanced models like Random Forest, XGBoost, or Neural Networks for better performance.
- Perform feature engineering to extract meaningful patterns.
- Handle class imbalance with techniques like SMOTE or class weighting.
- Deploy the model using Flask/Django for real-time transaction fraud detection.

## Author
**Your Name**  
- Email: sevugamoorthi738@gmail.com  
- GitHub: [sevugamoorthi](https://github.com/sevugamoorthi/fraud_detection.git)

