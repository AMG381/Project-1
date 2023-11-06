#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Dependencies
import matplotlib.pyplot as plt 
import pandas as pd  
import datetime as dt 
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression 
import os 


#read data 
avgtempdata = pd.read_csv('/Users/alphawave138/Documents/Proj.1data/Michavg.temp128yrs.csv')
maxtempdata = pd.read_csv('/Users/alphawave138/Documents/Proj.1data/Maxtemp128yrs.csv')
mintempdata = pd.read_csv('/Users/alphawave138/Documents/Proj.1data/Mintemp128yrs.csv')
raindata = pd.read_csv('/Users/alphawave138/Documents/Proj.1data/Rain128yrs.csv')

# Merge (Average temp and average rain)/ Merge Max/min tempature
MaxonMin = pd.merge(maxtempdata, mintempdata, how="left", on="Date")
TemponRain = pd.merge(avgtempdata, raindata, how="left", on="Date")
TemponRain






# In[33]:


MaxonMin


# In[34]:


TemponRain.plot.scatter(x='Date', 
                y='Temp (F)',
                ylim=(34,50)        )




# In[40]:


TR = TemponRain.plot.scatter(x='Date', 
                y='accum. (in)',
                ylim=(5,30)        )
x = TemponRain['Date']
y = TemponRain['accum. (in)']
linreg = LinearRegression()
linreg.fit(x.array.reshape(-1,1), y.array.reshape(-1,1))
plt.scatter(x, y)
plt.plot(np.linspace(190001,202001,128).reshape(-1,1), linreg.predict(np.linspace(189601,202301,128).reshape(-1,1)), 'r')
plt.ylim(5,30) 
plt.show()


# In[54]:


#linear regression on Max and date

MaxonMin.plot.scatter(x='Date', 
                y='Temp (F)',
                ylim=(35,60))
x = MaxonMin['Date']
y = MaxonMin['Temp (F)']
linreg = LinearRegression()
linreg.fit(x.array.reshape(-1,1), y.array.reshape(-1,1))
plt.scatter(x, y)
plt.plot(np.linspace(190001,202001,128).reshape(-1,1), linreg.predict(np.linspace(189601,202301,128).reshape(-1,1)), 'r')
plt.ylim(40,60) 
plt.show()


# In[48]:


MaxonMin.plot.scatter(x='Date', 
                y='Temp(F)',
                ylim=(20,40))
x = MaxonMin['Date']
y = MaxonMin['Temp(F)']
linreg = LinearRegression()
linreg.fit(x.array.reshape(-1,1), y.array.reshape(-1,1))
plt.scatter(x, y)
plt.plot(np.linspace(190001,202001,128).reshape(-1,1), linreg.predict(np.linspace(189601,202301,128).reshape(-1,1)), 'r')
plt.ylim(20,40) 
plt.show()


# In[57]:


#anomalies in max tempature

MaxonMin.plot.scatter(x='Date',
                      y='Anomaly_x',
                      ylim=(20,60))
x = MaxonMin['Date']
y = MaxonMin['Anomaly_x']
linreg = LinearRegression()
linreg.fit(x.array.reshape(-1,1), y.array.reshape(-1,1))
plt.scatter(x, y)
plt.plot(np.linspace(190001,202001,128).reshape(-1,1), linreg.predict(np.linspace(189601,202301,128).reshape(-1,1)), 'r')
plt.ylim(-10,10) 
plt.show()


# In[60]:


#anomalies in average tempature F

TemponRain.plot.scatter(x='Date',
                       y='Anomaly_x',
                        ylim=(0,35))
x = TemponRain['Date']
y = TemponRain['Anomaly_x']
linreg = LinearRegression()
linreg.fit(x.array.reshape(-1,1), y.array.reshape(-1,1))
plt.scatter(x, y)
plt.plot(np.linspace(190001,202001,128).reshape(-1,1), linreg.predict(np.linspace(189601,202301,128).reshape(-1,1)), 'r')
plt.ylim(-10,10) 
plt.show()


# In[61]:


#anomalies in min tempatue 

MaxonMin.plot.scatter(x='Date',
                      y='Anomaly_y',
                      ylim=(20,60))
x = MaxonMin['Date']
y = MaxonMin['Anomaly_y']
linreg = LinearRegression()
linreg.fit(x.array.reshape(-1,1), y.array.reshape(-1,1))
plt.scatter(x, y)
plt.plot(np.linspace(190001,202001,128).reshape(-1,1), linreg.predict(np.linspace(189601,202301,128).reshape(-1,1)), 'r')
plt.ylim(-10,10) 
plt.show()


# In[62]:


TemponRain.plot.scatter(x='Date',
                       y='Anomaly_y',
                        ylim=(0,35))
x = TemponRain['Date']
y = TemponRain['Anomaly_y']
linreg = LinearRegression()
linreg.fit(x.array.reshape(-1,1), y.array.reshape(-1,1))
plt.scatter(x, y)
plt.plot(np.linspace(190001,202001,128).reshape(-1,1), linreg.predict(np.linspace(189601,202301,128).reshape(-1,1)), 'r')
plt.ylim(-10,10) 
plt.show()


# In[ ]:




