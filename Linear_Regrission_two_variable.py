import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#read data
path='dataset.txt'
data=pd.read_csv(path,header=None ,names=['Size','Bedrooms','Price'])

#rescaling data
data=(data-data.mean())/data.std()

#add ones column
data.insert(0,'Ones',1)

#seprate X(training data ) from y (target variable )
cols=data.shape[1]
X=data.iloc[:92,0:cols-1]
y=data.iloc[:92,cols-1:cols]

#Convert from data frames to numpy matrices
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0,0])) 

#initialize variables for learning rate and iterations
alpha=0.01
iters=500

#cost function
def ComputeCost(X,y,theta):
    z=np.power(((X*theta.T)-y),2)
    return np.sum(z)/(2*len(X))

#gradientDescent function
def GradientDescent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost= np.zeros(iters)
    
    for i in range(iters):
        error=(X * theta.T)-y
        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j]-((alpha/len(X))*np.sum(term))
        theta=temp
        cost[i]=ComputeCost(X,y,theta)
    return theta,cost

#preform gradientDescent to 'fit' the model parameters 
g,cost=GradientDescent(X, y, theta, alpha, iters)

#get best fit line 
x_Size=np.linspace(data.Size.min(),data.Size.max(),100)
x_Bedrooms=np.linspace(data.Bedrooms.min(),data.Bedrooms.max(),100)
f=g[0,0]+(g[0,1] * x_Size)+(g[0,2]* x_Bedrooms)

#get the cost (error) of the model
thiscost=ComputeCost(X,y,theta)

#draw the line for Size vs Price
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(x_Size,f,'r',label='Prediction')
ax.scatter(data.Size,data.Price,label='Trianing Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Predicted Size vs. Price')

#draw the line for Bedrooms vs Price
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(x_Bedrooms,f,'r',label='Prediction')
ax.scatter(data.Bedrooms,data.Bedrooms,label='Trianing Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Predicted Bedrooms vs. Price')

#draw error graph
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Data')
