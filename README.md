# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Load necessary libraries for data handling, metrics, and visualization.

2. **Load Data**: Read the dataset using `pd.read_csv()` and display basic information.

3. **Initialize Parameters**: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4. **Gradient Descent**: Perform iterations to update `m` and `c` using gradient descent.

5. **Plot Error**: Visualize the error over iterations to monitor convergence of the model.

## Program and Outputs:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sethukkarasi C
RegisterNumber:  212223230201
*/
```
<br>

```
import pandas as pd
import numpy as np
```
<br>

```
df = pd.read_csv("50_Startups.csv")
```
<br>

```
df.head()
```
<br>

![output1](/o1.png)
<br>

```
df.tail()
```
<br>

![out2](/o2.png)

```
df.info()
```
<br>

![out3](/o3.png)
<br>

```
X = (df.iloc[1:,:-2].values)
y = (df.iloc[1:,-1].values).reshape(-1,1)
```
<br>

```
print(X)
```
<br>

![out4](/o4.png)
<br>

```
print(y)
```
<br>

![out5](/o5.png)
<br>

```
from sklearn.preprocessing import StandardScaler
def multivariate_linear_regression(X1,Y):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    num_iters = 1000
    error = []
    learning_rate = 0.001
    
    for _ in range(num_iters):
        # Calculatenpredictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #Calculate errors
        errors = (predictions - Y).reshape(-1,1)
        
        #Upadte theta using gradient descent
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
        
        # Record the error for each iteration
        error.append(np.mean(errors ** 2))

    return theta, error, num_iters
```
<br>

```
scaler = StandardScaler()
```
<br>

```
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(y)
```
<br>

```
print(X_scaled)
```
<br>

![out6](/o6.png)
<br>

```
print(Y_scaled)
```
<br>

![out7](/o7.png)
<br>

```
# Train the model using scaled data
theta, error, num_iters = multivariate_linear_regression(X_scaled,Y_scaled)
```
<br>

```
# Print the results
print("Theta:", theta)
print("Errors:", error)
print("Number of iterations:", num_iters)
```
<br>

![out8](/o8.png)
<br>

```
type(error)
print(len(error))
```
<br>

![out9](/o9.png)
<br>

```
import matplotlib.pyplot as plt
plt.plot(range(0,num_iters),error)
```
<br>

![out10](/o10.png)
<br>

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
