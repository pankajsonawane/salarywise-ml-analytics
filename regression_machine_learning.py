import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


#dataset = pd.read_csv(r'F:\Full_Stack_Gen_AI_Agentic_AI\Salary_Data.csv')
BASE_DIR = Path(__file__).resolve().parent

dataset = pd.read_csv(BASE_DIR / "Salary_Data.csv")

X = dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)


plt.scatter(X_test, y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_inter = regressor.intercept_
print(c_inter)

emp_exp_12 = m_slope*12+ c_inter
print(emp_exp_12)

emp_exp_20 = m_slope*20+ c_inter
print(emp_exp_20)


bias = regressor.score(X_train, y_train)
print(bias)

variance = regressor.score(X_test, y_test)
print(variance)


# == stat to machine learning

dataset.mean()
dataset['Salary'].mean()

dataset.median()
dataset['Salary'].mode()

dataset.var()
dataset.std()
dataset.skew()

#SSR

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE

y = y[0:6] #taking 6 because y_pred has only 6 records

SSE = np.sum((y-y_pred)**2)
print(SSE)

#sst

mean_total = np.mean(dataset.values)#
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#r2

r_square = 1 - SSR/SST
print(r_square)


import pickle

MODEL_PATH = BASE_DIR / "linear_regression_model.pkl"

with MODEL_PATH.open("wb") as file:
    pickle.dump(regressor, file)

print(f"Model saved to: {MODEL_PATH}")
