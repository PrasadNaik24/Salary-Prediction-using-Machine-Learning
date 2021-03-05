# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tkinter as tk

#GUI FOR SALARY DETECTION
top = tk.Tk()

top.mainloop()

# IMPORTING DATASET
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# SPLITTING THE DATASET INTO TRAINING AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# TRAINING THE MODEL
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTING THE TEST SET RESULTS
y_prediction = regressor.predict(X_test)

# FUNCTION FOR PREDICTING SALARY
def predict_salary(experience):
   salary = int(regressor.predict([[experience]]))
   print("Experience: {} || Predicted Salary: {}".format(experience, salary))

#TAKING EXPERIENCE AS USER INPUT
experience = float(input("Experience: "))
predict_salary(experience)


