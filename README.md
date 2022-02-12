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

In this project, I used the cars.csv dataset and I separated our data into independent and dependent variables. The output is the mileage for the cars in the dataset. Before applying Locally Weighted Regession and the Random Forest methods, we download the kernels to help in constructing non-linear decision boundaries using linear classifiers. 
