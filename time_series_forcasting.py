# -*- coding: utf-8 -*-
"""time series forcasting.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/152d_PYFDBYTQxz4mTGXuh3dpju4WvlGk

NAME - DEVANSHU KUMAR


TASK- 4

PROJECT NAME - TIME_SERIES-FORECASTING
"""

import pandas as pd
df = pd.DataFrame()

df = pd.read_csv('/content/Alcohol_Sales.csv',index_col = 'DATE',parse_dates=True)
df.index.freq = 'MS'

df.head()

df.columns = ['Sales']
df.plot(figsize=(12,8))

df['Sale_LastMonth']=df['Sales'].shift(+1)
df['Sale_2Monthsback']=df['Sales'].shift(+2)
df['Sale_3Monthsback']=df['Sales'].shift(+3)
df

df=df.dropna()
df

from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()

from sklearn.ensemble import RandomForestRegressor
 model = RandomForestRegressor(n_estimators=100,max_features=3,random_state=1)

import numpy as np

x1,x2,x3,y = df['Sale_LastMonth'],df['Sale_2Monthsback'],df['Sale_2Monthsback'],df['Sales']
x1,x2,x3,y = np.array(x1),np.array(x2),np.array(x3),np.array(y)
x1,x2,x3,y = x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),y.reshape(-1,1)
final_x=np.concatenate((x1,x2,x3),axis=1)
print(final_x)

x_train,x_test,y_train,y_test = final_x[:-30],final_x[-30:],y[:-30],y[-30:]

model.fit(x_train,y_train)
lin_model.fit(x_train,y_train)

pred = model.predict(x_test)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]=(12,8)
plt.plot(pred,label='Random_Forest_predictions')
plt.plot(y_test,label='Actual Sales')
plt.legend(loc='upper left')
plt.show()

lin_pred = lin_model.predict(x_test)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]=(12,8)
plt.plot(lin_pred,label='Linear_Regression_predictions')
plt.plot(y_test,label='Actual Sales')
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_rf = sqrt(mean_squared_error(pred,y_test))
rmse_lr = sqrt(mean_squared_error(lin_pred,y_test))

print('MSE  for random forest model:',rmse_rf)
print('MSE for linear regression model:',rmse_lr)