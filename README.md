# project2
This repository contains the presentation on the concepts of Locally Weighted Regression and Random Forest.

Before uploading our data, we import some libraries to help us read and manipulate our data.

```Python
#importing libraries

import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
```

Next, I downloaded the kernels to help in constructing the non-linear decision boundaries using linear classifiers. In this presentation I am only going to show the Tricubic Kernel because it is the only one I used.

```Python
# Tricubic Kernel
def tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)
```

After downloading the kernel, I imported the cars.csv data from my google drive. 

```Python
data = pd.read_csv('drive/MyDrive/Colab Notebooks/Data_410/data/cars.csv')
```

Before our Analysis, I separated the data into independent and dependent variables. The dependent variable is Mileage for the cars.

```Python
x = data[['CYL','ENG','WGT']].values
y = data['MPG'].values

x = data['WGT'].values
y= data['MPG'].values

```

After separating the data, I created a Locally Weighted Regression Function that performs a regression around a point of interest using only training data that are local to that point. This locally weighted regression function takes the independent, dependent value, the choice of kernel, and hyperparameter tau. In this function, we expect x to be sorted in an increasing order
```Python
def lowess_reg(x, y, xnew, kern, tau):
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
       
        beta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = beta[0] + beta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)
```

After that, we are going to use cross validation to compare and select a better model for our data. I split the data into 2 parts, Train data and Test data. Train data is used to train the model and the test data is used forpredictions. If the model performs well over the test data, we can use the model for predictions. 75% of the data is train data and 25% is the tests data.

```Python 

xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.25, random_state=123)
```

Next, I standardize the data using standardScaler to make sure that the data I am using is consistent.

```Python

scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain.reshape(-1,1))
xtest_scaled = scale.transform(xtest.reshape(-1,1))

```
After standardizing our data, I chacked our predicted value of the results of out test set and start guessing values for tau hyper-parameter.

```Python
yhat_test = lowess_reg(xtrain_scaled.ravel(),ytrain,xtest_scaled,tricubic,0.1)

```
After this, I chacked the mean square error for our test set

```Python
mse(yhat_test,ytest)

```
The mse = 15.961885966790932 

```Python
plt.plot(np.sort(xtest_scaled.ravel()),yhat_test)
```
![download (1)](https://user-images.githubusercontent.com/98835048/153718249-de76a865-63e2-46dc-8736-75ba76dd9b48.png)

# Random Forest. 

After applying Locally weighted regression method on this data, I decidedto try Random Forest as well to findout which one works better. The random forest establishes the outcome based on the predictions of the other decision trees using the average of the mean. Random forest does not need hyperparameters like the locally weighted regression model.

```Python
rf = RandomForestRegressor(n_estimators=100,max_depth=3)

```
Let's fit the modelon our train data and check the predictons it makes
```Python
rf.fit(xtrain_scaled,ytrain)
yhat = rf.predict(xtest_scaled)
```
```Python

```

