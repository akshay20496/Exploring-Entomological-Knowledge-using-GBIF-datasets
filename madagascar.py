# -*- coding: utf-8 -*-


# !pip install pygbif

# from pygbif import registry

# from pygbif import species

# from pygbif import occurrences as occ

# occ.search
# registry.datasets(type="OCCURRENCE")

# occ.search(country='MG',has_coordinate = True,has_geospatial_issue=False,taxonKey = 216)

# aag = occ.download('taxonKey = 1937692')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

!pip install plotly
import plotly.express as px
dir(px)
import plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode,plot,iplot

df = pd.read_csv(r'madagascar_data.csv', sep = '\t')

df.columns.tolist()
df.shape
df['order'].unique()          #total unique orders of insects
df['order'].value_counts()    #count of each order present in data                                                  
len(df['locality'].unique())  #total unique localities

df.isnull().any(axis=0)       #null values in columns in boolean format
df.isnull().sum()             #number of null values in each column

#creating dataframe having orders of insect with their latitude and longitude
df1 = df[['order','decimalLatitude','decimalLongitude']]   
#dropping nan values
df1.dropna(axis = 0, subset = ['order'], inplace=True) 
df1.isnull().any(axis=0)
df1.isnull().sum()
df1['order'].unique()
len(df1['order'].unique())    #total number of unique orders

#ord = ['lep','col','odo','hem','hym','orth','dip','neur','cnem','psoc','blat','thys','derm','mant','phasm','trich','ephem','plec','strep','siph','embio','archae','meco','proto']

#generating the basemap of Madagascar
!pip install folium
def generatebasemap(default_location=[-18.9276,48.4142],default_zoom_start=6):
    basemap = folium.Map(location = default_location, zoom_start = default_zoom_start)
    return basemap

import folium
basemap = generatebasemap()
basemap.save('MG_map.html')

#generating heatmap for the all orders of insects
from folium.plugins import HeatMap
x = HeatMap(df1[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x.save('MG1_map.html')

#generating marker cluster for all orders of insects
from folium.plugins import FastMarkerCluster
y = FastMarkerCluster(df1[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y.save('MG1-marker_cluster.html')

#generating scatter map for all orders of insects
fig = px.scatter_mapbox(df1,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
fig.update_layout(mapbox_style="open-street-map")
fig.write_html("plotly_mapbox_scatter.html")

#generating heatmaps, marker clusters, scatter mapbox for each and every order of insect 
lep = df1[df1['order'] == 'Lepidoptera']
x1 = HeatMap(lep[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x1.save('MG_lep_map.html')

basemap = generatebasemap()
y1 = FastMarkerCluster(lep[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y1.save('lep_marker_cluster.html')

z1 = px.scatter_mapbox(lep,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z1.update_layout(mapbox_style="open-street-map")
z1.write_html("lep_mapbox_scatter.html")

col = df1[df1['order'] == 'Coleoptera']
x2 = HeatMap(col[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x2.save('MG_col_map.html')

basemap = generatebasemap()
y2 = FastMarkerCluster(col[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y2.save('col_marker_cluster.html')

z2 = px.scatter_mapbox(col,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z2.update_layout(mapbox_style="open-street-map")
z2.write_html("col_mapbox_scatter.html")

odo = df1[df1['order'] == 'Odonata']
x3 = HeatMap(odo[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x3.save('MG_odo_map.html')

basemap = generatebasemap()
y3 = FastMarkerCluster(odo[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y3.save('odo_marker_cluster.html')

z3 = px.scatter_mapbox(odo,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z3.update_layout(mapbox_style="open-street-map")
z3.write_html("odo_mapbox_scatter.html")

hem = df1[df1['order'] == 'Hemiptera']
x4 = HeatMap(hem[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x4.save('MG_hem_map.html')

basemap = generatebasemap()
y4 = FastMarkerCluster(hem[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y4.save('hem_marker_cluster.html')

z4 = px.scatter_mapbox(hem,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z4.update_layout(mapbox_style="open-street-map")
z4.write_html("hem_mapbox_scatter.html")

neur = df1[df1['order'] == 'Neuroptera']
x5 = HeatMap(neur[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x5.save('MG_neur_map.html')

basemap = generatebasemap()
y5 = FastMarkerCluster(neur[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y5.save('neur_marker_cluster.html')

z5 = px.scatter_mapbox(neur,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z5.update_layout(mapbox_style="open-street-map")
z5.write_html("neur_mapbox_scatter.html")

hym = df1[df1['order'] == 'Hymenoptera']
x6 = HeatMap(hym[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x6.save('MG_hym_map.html')

basemap = generatebasemap()
y6 = FastMarkerCluster(hym[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y6.save('hym_marker_cluster.html')

z6 = px.scatter_mapbox(hym,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z6.update_layout(mapbox_style="open-street-map")
z6.write_html("hym_mapbox_scatter.html")

dip = df1[df1['order'] == 'Diptera']
x7 = HeatMap(dip[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x7.save('MG_dip_map.html')

basemap = generatebasemap()
y7 = FastMarkerCluster(dip[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y7.save('dip_marker_cluster.html')

z7 = px.scatter_mapbox(dip,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z7.update_layout(mapbox_style="open-street-map")
z7.write_html("dip_mapbox_scatter.html")

orth = df1[df1['order'] == 'Orthoptera']
x8 = HeatMap(orth[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x8.save('MG_orth_map.html')

basemap = generatebasemap()
y8 = FastMarkerCluster(orth[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y8.save('orth_marker_cluster.html')

z8 = px.scatter_mapbox(orth,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z8.update_layout(mapbox_style="open-street-map")
z8.write_html("orth_mapbox_scatter.html")

trich = df1[df1['order'] == 'Trichoptera']
x9 = HeatMap(trich[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x9.save('MG_trich_map.html')

basemap = generatebasemap()
y9 = FastMarkerCluster(trich[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y9.save('trich_marker_cluster.html')

z9 = px.scatter_mapbox(trich,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z9.update_layout(mapbox_style="open-street-map")
z9.write_html("trich_mapbox_scatter.html")

ephem = df1[df1['order'] == 'Ephemeroptera']
x10 = HeatMap(ephem[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x10.save('MG_ephem_map.html')

basemap = generatebasemap()
y10 = FastMarkerCluster(ephem[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y10.save('ephem_marker_cluster.html')

z10 = px.scatter_mapbox(ephem,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z10.update_layout(mapbox_style="open-street-map")
z10.write_html("ephem_mapbox_scatter.html")

mant = df1[df1['order'] == 'Mantodea']
x11 = HeatMap(mant[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x11.save('MG_mant_map.html')

basemap = generatebasemap()
y11 = FastMarkerCluster(mant[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y11.save('mant_marker_cluster.html')

z11 = px.scatter_mapbox(mant,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z11.update_layout(mapbox_style="open-street-map")
z11.write_html("mant_mapbox_scatter.html")

psoc = df1[df1['order'] == 'Psocodea']
x12 = HeatMap(psoc[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x12.save('MG_psoc_map.html')

basemap = generatebasemap()
y12 = FastMarkerCluster(psoc[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y12.save('psoc_marker_cluster.html')

z12 = px.scatter_mapbox(psoc,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z12.update_layout(mapbox_style="open-street-map")
z12.write_html("psoc_mapbox_scatter.html")

blat = df1[df1['order'] == 'Blattodea']
x13 = HeatMap(blat[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x13.save('MG_blat_map.html')

basemap = generatebasemap()
y13 = FastMarkerCluster(blat[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y13.save('blat_marker_cluster.html')

z13 = px.scatter_mapbox(blat,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z13.update_layout(mapbox_style="open-street-map")
z13.write_html("blat_mapbox_scatter.html")

plec = df1[df1['order'] == 'Plecoptera']
x14 = HeatMap(plec[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x14.save('MG_plec_map.html')

basemap = generatebasemap()
y14 = FastMarkerCluster(plec[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y14.save('plec_marker_cluster.html')

z14 = px.scatter_mapbox(plec,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z14.update_layout(mapbox_style="open-street-map")
z14.write_html("plec_mapbox_scatter.html")


siph = df1[df1['order'] == 'Siphonaptera']
x15 = HeatMap(siph[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x15.save('MG_siph_map.html')

basemap = generatebasemap()
y15 = FastMarkerCluster(siph[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y15.save('siph_marker_cluster.html')

z15 = px.scatter_mapbox(siph,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z15.update_layout(mapbox_style="open-street-map")
z15.write_html("siph_mapbox_scatter.html")

thys = df1[df1['order'] == 'Thysanoptera']
x16 = HeatMap(thys[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x16.save('MG_thys_map.html')

basemap = generatebasemap()
y16 = FastMarkerCluster(thys[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y16.save('thys_marker_cluster.html')

z16 = px.scatter_mapbox(thys,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z16.update_layout(mapbox_style="open-street-map")
z16.write_html("thys_mapbox_scatter.html")

phasm = df1[df1['order'] == 'Phasmida']
x17 = HeatMap(phasm[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x17.save('MG_phasm_map.html')

basemap = generatebasemap()
y17 = FastMarkerCluster(phasm[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y17.save('phasm_marker_cluster.html')

z17 = px.scatter_mapbox(phasm,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z17.update_layout(mapbox_style="open-street-map")
z17.write_html("phasm_mapbox_scatter.html")

derm = df1[df1['order'] == 'Dermaptera']
x18 = HeatMap(derm[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x18.save('MG_derm_map.html')

basemap = generatebasemap()
y18 = FastMarkerCluster(derm[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y18.save('derm_marker_cluster.html')

z18 = px.scatter_mapbox(derm,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z18.update_layout(mapbox_style="open-street-map")
z18.write_html("derm_mapbox_scatter.html")

list = ['Embioptera','Archaeognatha','Strepsiptera','Cnemidolestodea','Mecoptera', 'Protorthoptera']
other = df1[df['order'].isin(list)]
x19 = HeatMap(other[['decimalLatitude','decimalLongitude']], zoom = 7).add_to(basemap)
x19.save('MG_other_map.html')

basemap = generatebasemap()
y19 = FastMarkerCluster(other[['decimalLatitude','decimalLongitude','order']], zoom = 7).add_to(basemap)
y19.save('other_marker_cluster.html')

z19 = px.scatter_mapbox(other,lat='decimalLatitude',lon='decimalLongitude',hover_name='order',color_discrete_sequence=["fuchsia"],zoom=4)
z19.update_layout(mapbox_style="open-street-map")
z19.write_html("other_mapbox_scatter.html")


# df2 = df[['day','month','year']]
# df2['date'] = pd.to_datetime(df2[['day','month','year']]).dt.date
# df1['date'] = df2['date']
# df1.dropna(axis = 0, subset = ['date'], inplace = True)
# df1 = df1.sort_values(by='date')
# df1['location'] = df['locality']

# df2['date'] = df2['date'].str.split().str[0]
# # df2['date'].dt.tz_convert(None)
# # df['eventDate'] = df['eventDate'].dt.tz_convert(None)
# df2['dateInt']=df2['day'].astype(str) + df2['month'].astype(str).str.zfill(2)+ df2['year'].astype(str).str.zfill(2)
# df2['date'] = pd.to_datetime(df2['dateInt'], format='%d%m%Y')

#create new dataframe having locality, order with latitude and longitude
df3 = df[['order','decimalLatitude','decimalLongitude','locality']] 
df3.dropna(axis = 0, subset = ['order','locality'], inplace = True)
#finding unique localities
len(df3['locality'].unique())

#create dataframe having localities and records at each locality
location = df3.groupby('locality')['order'].count().reset_index()

lat = []
lon = []
for i in location['locality'].unique():
    p = df3.loc[df3['locality'] == i, 'decimalLatitude']
    q = df3.loc[df3['locality'] == i, 'decimalLongitude']
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

a = HeatMap(location[['lat','lon','count']], zoom = 7).add_to(basemap)
a.save('MG_count.html')

b = FastMarkerCluster(location[['lat','lon']], zoom = 7).add_to(basemap)
b.save('MG_count_marker_cluster.html')

fig1 = px.density_mapbox(location, lat='lat', lon='lon', z='count', zoom=20, mapbox_style="stamen-terrain")
fig.write_html('density_mapbox.html')

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world[world.name == 'Madagascar'].plot(color='white', edgecolor='black')

world.plot()
world.to_crs(epsg=4326).plot()

from shapely.geometry import Point, Polygon
crs = {'init':'EPSG:4326'}
geometry = [Point(xy) for xy in zip(location['lon'], location['lat'])]
geo_df = gpd.GeoDataFrame(location, crs = crs, geometry = geometry)

fig2, ax = plt.subplots(figsize = (10,10))
world[world.name == 'Madagascar'].to_crs(epsg=4326).plot(ax=ax, color='lightgrey')
geo_df.plot(ax=ax)
ax.set_title('Locations in Madagascar')
plt.savefig('Madagascar_locations.jpg')

geo_df['count_log'] = np.log(geo_df['count'])

fig3, ax = plt.subplots(figsize = (10,10))
world[world.name == 'Madagascar'].to_crs(epsg=4326).plot(ax=ax, color='lightgrey')
geo_df.plot(column = 'count_log', ax=ax, cmap = 'rainbow', legend = True, legend_kwds={'shrink': 0.3},markersize = 25)
ax.set_title('Insect Order count in Location Heatmap')
plt.savefig('orders_count_HeatMap.jpg')

df1['order'].value_counts()
df1['order'].value_counts().plot(kind = 'bar')

fig4, ax = plt.subplots(figsize = (10,10))
geometry = [Point(xy) for xy in zip(df1['decimalLongitude'], df1['decimalLatitude'])]
geo_df1 = gpd.GeoDataFrame(df1, crs = crs, geometry = geometry)

fig4, ax = plt.subplots(figsize = (10,10))
world[world.name == 'Madagascar'].to_crs(epsg=4326).plot(ax=ax, color='lightgrey')
geo_df1.plot(ax=ax)
ax.set_title('Total Orders in Madagascar')
plt.savefig('Madagascar_orders.jpg')


