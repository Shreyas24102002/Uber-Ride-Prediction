#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("C:/Users/Shreyas/Desktop/Submission Material/uber.csv")


# In[4]:


df


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


# Cleaning: -


# In[9]:


df.isna().sum()


# In[10]:


df = df.drop(['Unnamed: 0','key'], axis=1)


# In[11]:


df.isna().sum()


# In[12]:


# Remove null rows
df.dropna(axis=0, inplace=True)


# In[13]:


df.isna().sum()


# In[14]:


# Fixing datatype of pickup_datetime from Object to DateTime
df.pickup_datetime = pd.to_datetime(df.pickup_datetime, errors='coerce')


# In[15]:


df.pickup_datetime


# In[16]:


df = df.assign(
    second = df.pickup_datetime.dt.second,
    minute = df.pickup_datetime.dt.minute,
    hour = df.pickup_datetime.dt.hour,
    day = df.pickup_datetime.dt.day,
    month = df.pickup_datetime.dt.month,
    year = df.pickup_datetime.dt.year,
    dayofweek = df.pickup_datetime.dt.dayofweek
    
)
df = df.drop('pickup_datetime', axis=1)

df.info()


# In[17]:


df.head()


# In[ ]:





# In[18]:


incorrect_coordinates = df.loc[
    (df.pickup_latitude>90)|(df.pickup_latitude<-90)|
    (df.dropoff_latitude>90)|(df.dropoff_latitude<-90)|
    (df.pickup_longitude>90)|(df.pickup_longitude<-90)|
    (df.dropoff_longitude>90)|(df.dropoff_longitude<-90)
]

df.drop(incorrect_coordinates, inplace=True, errors='ignore')


# In[19]:


def distance_transform(longitude1, latitude1, longitude2, latitude2):
    long1, lati1, long2, lati2 = map(np.radians, [longitude1, latitude1, longitude2, latitude2])
    dist_long = long2 - long1
    dist_lati = lati2 - lati1
    a = np.sin(dist_lati/2)**2 + np.cos(lati1) * np.cos(lati2) * np.sin(dist_long/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) * 6371
    # long1,lati1,long2,lati2 = longitude1[pos],latitude1[pos],longitude2[pos],latitude2[pos]
    # c = sqrt((long2 - long1) ** 2 + (lati2 - lati1) ** 2)asin 
       
    return c


# In[20]:


df['Distance'] = distance_transform(
    df['pickup_longitude'],
    df['pickup_latitude'],
    df['dropoff_longitude'],
    df['dropoff_latitude']
)


# In[21]:


df.columns


# In[22]:


df.info()


# In[23]:


df.head()


# In[24]:


import matplotlib.pyplot as plt
import pylab
import seaborn as sns


# In[25]:


# Outliers


# In[26]:


plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel('Fare Amount')


# In[27]:


plt.figure(figsize=(20,12))
sns.boxplot(data = df)


# In[28]:


df.drop(df[df['Distance'] >= 60].index, inplace = True)
df.drop(df[df['fare_amount'] <= 0].index, inplace = True)


# In[29]:


df.drop(df[(df['fare_amount']>100) & (df['Distance']<1)].index, inplace = True )
df.drop(df[(df['fare_amount']<100) & (df['Distance']>100)].index, inplace = True )


# In[30]:


plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel("fare_amount")


# In[31]:


corr  =df.corr()
corr.style.background_gradient(cmap='BuGn')


# In[32]:


from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing


# In[33]:


X = df['Distance'].values.reshape(-1,1)
y = df['fare_amount'].values.reshape(-1,1)


# In[34]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
y_std = std.fit_transform(y)
print(y_std)

x_std = std.fit_transform(X)
print(x_std)


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_std, y_std, test_size=0.2, random_state=0)


# In[37]:


from sklearn.linear_model import LinearRegression
l_reg = LinearRegression()
l_reg.fit(X_train,y_train)

print("Training set score: {:.2f}".format(l_reg.score(X_train,y_train)))
print("Test set score: {:.7f}".format(l_reg.score(X_test, y_test)))


# In[38]:


y_pred = l_reg.predict(X_test)
y_pred


# In[40]:


result = pd.DataFrame()
result[['Actual']]=y_test
result[['Predicted']]=y_pred

result.sample(10)


# In[41]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Absolute % Error:', metrics.mean_absolute_percentage_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared (R²):', np.sqrt(metrics.r2_score(y_test, y_pred)))


# In[42]:


plt.subplot(2,2,1)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, l_reg.predict(X_train), color="blue")
plt.title("Fare vs Distance (Training Set)")
plt.ylabel("fare_amount")
plt.xlabel("Distance")


# In[43]:


plt.subplot(2,2,2)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, l_reg.predict(X_train), color ="blue")
plt.ylabel("fare_amount")
plt.xlabel("Distance")
plt.title("Fare vs Distance (Test Set)")


# In[45]:


cols = ['Model', 'RMSE', 'R-Sqaured']

resultT = pd.DataFrame(columns = cols)

linreg_metrics = pd.DataFrame([[
  "Linear Regression Model",
   np.sqrt(metrics.mean_squared_error(y_test,y_pred)),
   np.sqrt(metrics.r2_score(y_test,y_pred)) 
]   
], columns=cols)

resultT = pd.concat([resultT, linreg_metrics], ignore_index=True)
resultT


# In[50]:


# Random Forest Regressor

rf_reg = RandomForestRegressor(n_estimators=100, random_state=10)

rf_reg.fit(X_train,y_train)


# In[ ]:


# RandomForestRegressor(random_state=10)


# In[54]:


y_pred_RF = rf_reg.predict(X_test)
result = pd.DataFrame()
result[['Actual']]=y_test
result['Predicted']=y_pred_RF

result.sample(10)


# In[55]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_RF))
print('Mean Absolute % Error:', metrics.mean_absolute_percentage_error(y_test, y_pred_RF))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_RF))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF)))
print('R Squared (R²):', np.sqrt(metrics.r2_score(y_test, y_pred_RF)))


# In[56]:


plt.scatter(X_test, y_test, c = 'b', alpha = 0.5, marker = '.', label = 'Real')
plt.scatter(X_test, y_pred_RF, c = 'r', alpha = 0.5, marker = '.', label = 'Predicted')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.grid(color = '#D3D3D3', linestyle = 'solid')
plt.legend(loc = 'lower right')


plt.tight_layout()
plt.show()


# In[58]:


rf_metrics=pd.DataFrame([[
    "Randome Forest Regressor",
     np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF)),
     np.sqrt(metrics.r2_score(y_test, y_pred_RF))
]], columns=cols)

resultT = pd.concat([resultT, rf_metrics], ignore_index=True)
resultT


# In[ ]:




