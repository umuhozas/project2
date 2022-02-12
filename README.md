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

Next, I standardize the data using standardScaler to make sure that the data I amusing is consistent.

```Python
# Let's standardize our data using standardScaler to make sure that our data is consistent.
scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain.reshape(-1,1))
xtest_scaled = scale.transform(xtest.reshape(-1,1))

```


In this project, I used the cars.csv dataset and I separated our data into independent and dependent variables. The output is the mileage for the cars in the dataset. Before applying Locally Weighted Regession and the Random Forest methods, we download the kernels to help in constructing non-linear decision boundaries using linear classifiers. 
