# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 02:46:19 2022

@author: Akshay Ghatage
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import stats

!pip install plotly
import plotly.express as px
dir(px)
import plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode,plot,iplot

df = pd.read_csv(r'madagascar_data.csv', sep = '\t')

df.columns.tolist()

df.isnull().any(axis=0)       
df.isnull().sum() 

df1 = df[['order','decimalLatitude','decimalLongitude','locality']] 
df1.dropna(axis = 0, subset = ['order','locality'], inplace=True) 
df1.isnull().any(axis=0)
df1['order'].unique()
len(df1['order'].unique())
len(df['locality'].unique())

#create dataframe having localities and records at each locality
location = df1.groupby('locality')['order'].count().reset_index()

lat = []
lon = []
for i in location['locality'].unique():
    p = df1.loc[df1['locality'] == i, 'decimalLatitude']
    q = df1.loc[df1['locality'] == i, 'decimalLongitude']
    lat.append(p.unique())
    lon.append(q.unique())
    
lat = np.array(lat)
lat1 = []
for i in range(7383):
    lat1.append(np.mean(lat[i]))
     
lon = np.array(lon) 
lon1 = []
for i in range(7383):
    lon1.append(np.mean(lon[i]))

location['lat'] = lat1
location['lon'] = lon1
location.rename(columns = {'order' : 'count'}, inplace = True)

corr = df1.groupby('order')['locality'].count().reset_index()

list1 = []
for i in corr['order'].unique():
    r = df1.loc[df1['order'] == i, 'locality']
    list1.append(r.unique())

corr['sites'] = list1

#finding percentages
#1-hymenoptera
x = corr.iloc[8,2]
x1 = corr.iloc[2,2]    #coleptera
xx1 = np.intersect1d(x,x1)
corr_per_xx1 = ((len(xx1)/len(x))*100)

x2 = corr.iloc[4,2] #diptera
xx2 = np.intersect1d(x,x2)
corr_per_xx2 = ((len(xx2)/len(x))*100) 

x3 = corr.iloc[6,2] #ephemeroptera
xx3 = np.intersect1d(x,x3)
corr_per_xx3 = ((len(xx3)/len(x))*100)

x4 = corr.iloc[7,2] #hemiptera
xx4 = np.intersect1d(x,x4)
corr_per_xx4 = ((len(xx4)/len(x))*100)

x5 = corr.iloc[9,2] #lepidoptera
xx5 = np.intersect1d(x,x5)
corr_per_xx5 = ((len(xx5)/len(x))*100)

x6 = corr.iloc[12,2] #odonata
xx6 = np.intersect1d(x,x6)
corr_per_xx6 = ((len(xx6)/len(x))*100)

x7 = corr.iloc[13,2] #orthoptera
xx7 = np.intersect1d(x,x7)
corr_per_xx7 = ((len(xx7)/len(x))*100)

x8 = corr.iloc[20,2] #trichoptera
xx8 = np.intersect1d(x,x8)
corr_per_xx8 = ((len(xx8)/len(x))*100)

#2-coleoptera
corr_per_x1x = ((len(xx1)/len(x1))*100)  #corr_per_hym

x1x2 = np.intersect1d(x1,x2)
corr_per_x1x2 = ((len(x1x2)/len(x1))*100) #dip

x1x3 = np.intersect1d(x1,x3)
corr_per_x1x3 = ((len(x1x3)/len(x1))*100) #ephem

x1x4 = np.intersect1d(x1,x4)
corr_per_x1x4 = ((len(x1x4)/len(x1))*100) #hemi

x1x5 = np.intersect1d(x1,x5)
corr_per_x1x5 = ((len(x1x5)/len(x1))*100) #lepi

x1x6 = np.intersect1d(x1,x6)
corr_per_x1x6 = ((len(x1x6)/len(x1))*100) #odo

x1x7 = np.intersect1d(x1,x7)
corr_per_x1x7 = ((len(x1x7)/len(x1))*100) #ortho

x1x8 = np.intersect1d(x1,x8)
corr_per_x1x8 = ((len(x1x8)/len(x1))*100) #trich

#3- diptera
corr_per_x2x = ((len(xx2)/len(x2))*100)  #corr_per_hym

corr_per_x2x1 = ((len(x1x2)/len(x2))*100) #col

x2x3 = np.intersect1d(x2,x3)
corr_per_x2x3 = ((len(x2x3)/len(x2))*100) #ephem

x2x4 = np.intersect1d(x2,x4)
corr_per_x2x4 = ((len(x2x4)/len(x2))*100) #hemi

x2x5 = np.intersect1d(x2,x5)
corr_per_x2x5 = ((len(x2x5)/len(x2))*100) #lepi

x2x6 = np.intersect1d(x2,x6)
corr_per_x2x6 = ((len(x2x6)/len(x2))*100) #odo

x2x7 = np.intersect1d(x2,x7)
corr_per_x2x7 = ((len(x2x7)/len(x2))*100) #ortho

x2x8 = np.intersect1d(x2,x8)
corr_per_x2x8 = ((len(x2x8)/len(x2))*100) #trich

#4-ephemeroptera
corr_per_x3x = ((len(xx3)/len(x3))*100)  #corr_per_hym

corr_per_x3x1 = ((len(x1x3)/len(x3))*100) #col

corr_per_x3x2 = ((len(x2x3)/len(x3))*100) #dip

x3x4 = np.intersect1d(x3,x4)
corr_per_x3x4 = ((len(x3x4)/len(x3))*100) #hemi

x3x5 = np.intersect1d(x3,x5)
corr_per_x3x5 = ((len(x3x5)/len(x3))*100) #lepi

x3x6 = np.intersect1d(x3,x6)
corr_per_x3x6 = ((len(x3x6)/len(x3))*100) #odo

x3x7 = np.intersect1d(x3,x7)
corr_per_x3x7 = ((len(x3x7)/len(x3))*100) #ortho

x3x8 = np.intersect1d(x3,x8)
corr_per_x3x8 = ((len(x3x8)/len(x3))*100) #trich

#5-hemiptera
corr_per_x4x = ((len(xx4)/len(x4))*100)  #corr_per_hym

corr_per_x4x1 = ((len(x1x4)/len(x4))*100) #col

corr_per_x4x2 = ((len(x2x4)/len(x4))*100) #dip

corr_per_x4x3 = ((len(x3x4)/len(x4))*100) #ephem

x4x5 = np.intersect1d(x4,x5)
corr_per_x4x5 = ((len(x4x5)/len(x4))*100) #lepi

x4x6 = np.intersect1d(x4,x6)
corr_per_x4x6 = ((len(x4x6)/len(x4))*100) #odo

x4x7 = np.intersect1d(x4,x7)
corr_per_x4x7 = ((len(x4x7)/len(x4))*100) #ortho

x4x8 = np.intersect1d(x4,x8)
corr_per_x4x8 = ((len(x4x8)/len(x4))*100) #trich

#6-lepidoptera
corr_per_x5x = ((len(xx5)/len(x5))*100)  #corr_per_hym

corr_per_x5x1 = ((len(x1x5)/len(x5))*100) #col

corr_per_x5x2 = ((len(x2x5)/len(x5))*100) #dip

corr_per_x5x3 = ((len(x3x5)/len(x5))*100) #ephem

corr_per_x5x4 = ((len(x4x5)/len(x5))*100) #hemi

x5x6 = np.intersect1d(x5,x6)
corr_per_x5x6 = ((len(x5x6)/len(x5))*100) #odo

x5x7 = np.intersect1d(x5,x7)
corr_per_x5x7 = ((len(x5x7)/len(x5))*100) #ortho

x5x8 = np.intersect1d(x5,x8)
corr_per_x5x8 = ((len(x5x8)/len(x5))*100) #trich

#7-odonata
corr_per_x6x = ((len(xx6)/len(x6))*100)  #corr_per_hym

corr_per_x6x1 = ((len(x1x6)/len(x6))*100) #col

corr_per_x6x2 = ((len(x2x6)/len(x6))*100) #dip

corr_per_x6x3 = ((len(x3x6)/len(x6))*100) #ephem

corr_per_x6x4 = ((len(x4x6)/len(x6))*100) #hemi

corr_per_x6x5 = ((len(x5x6)/len(x6))*100) #lepi

x6x7 = np.intersect1d(x6,x7)
corr_per_x6x7 = ((len(x6x7)/len(x6))*100) #ortho

x6x8 = np.intersect1d(x6,x8)
corr_per_x6x8 = ((len(x6x8)/len(x6))*100) #trich

#8-orthoptera
corr_per_x7x = ((len(xx7)/len(x7))*100)  #corr_per_hym

corr_per_x7x1 = ((len(x1x7)/len(x7))*100) #col

corr_per_x7x2 = ((len(x2x7)/len(x7))*100) #dip

corr_per_x7x3 = ((len(x3x7)/len(x7))*100) #ephem

corr_per_x7x4 = ((len(x4x7)/len(x7))*100) #hemi

corr_per_x7x5 = ((len(x5x7)/len(x7))*100) #lepi

corr_per_x7x6 = ((len(x6x7)/len(x7))*100) #odo

x7x8 = np.intersect1d(x7,x8)
corr_per_x7x8 = ((len(x7x8)/len(x7))*100) #trich

#9-trichoptera
corr_per_x8x = ((len(xx8)/len(x8))*100)  #corr_per_hym

corr_per_x8x1 = ((len(x1x8)/len(x8))*100) #col

corr_per_x8x2 = ((len(x2x8)/len(x8))*100) #dip

corr_per_x8x3 = ((len(x3x8)/len(x8))*100) #ephem

corr_per_x8x4 = ((len(x4x8)/len(x8))*100) #hemi

corr_per_x8x5 = ((len(x5x8)/len(x8))*100) #lepi

corr_per_x8x6 = ((len(x6x8)/len(x8))*100) #odo

corr_per_x8x7 = ((len(x7x8)/len(x8))*100) #ortho



    









