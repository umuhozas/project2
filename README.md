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

In this project, I used the cars.csv dataset and I separated our data into independent and dependent variables. The output is the mileage for the cars in the dataset. Before applying Locally Weighted Regession and the Random Forest methods, we download the kernels to help in constructing non-linear decision boundaries using linear classifiers. 
