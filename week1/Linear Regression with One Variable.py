import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#calculation of cost function
def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

#gradient descent calculation
def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp)
        theta = theta - (alpha/m) * temp
    return theta


data = pd.read_csv('ex1data1.txt', header = None) #read from dataset
X= data.iloc[:,0]
y=data.iloc[:,1]
m=len(y)

plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in 10,000 $')
plt.show()

#convert rank1 arrays to rank2
X = X[:,np.newaxis]
y = y[:,np.newaxis]

theta = np.zeros([2,1]) #setting the initial values of thetha
iterations = 1500 #setting iterations as given
alpha = 0.01 #learning rate =0.01
ones = np.ones((m,1)) #for adding the intercept term as mentioned.
X = np.hstack((ones, X)) #horizontally stack the ones and X  


#intital value of thetha
J = computeCost(X, y, theta)
print(J)
#for the new thetha 
theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

#calculate cost after updating the theta
J = computeCost(X, y, theta)
print(J)

plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()