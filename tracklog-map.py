#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:08:16 2017

@author: areed145
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
from owslib.wms import WebMapService

trackfiles = ['tracklog-1','tracklog-2','tracklog-3','tracklog-4','tracklog-5','tracklog-6','tracklog-7','tracklog-8']
resample_rate = 2
trackdata = pd.DataFrame()
for trackfile in trackfiles:
    data = pd.read_csv(trackfile+'.csv', skiprows=2)
    data['File'] = trackfile
    data.sort_values('Timestamp', axis=0, ascending=True, inplace=True)
    data['Datetime'] = pd.to_datetime(data['Timestamp'], unit='ms')
    data['Altitude_deltarm'] = data['Altitude'].diff().rolling(30).mean()
    data = data.set_index('Datetime')
    data = data.resample(str(resample_rate)+'S').mean()
    data['Timestamp_cum'] = (data['Timestamp'] - data['Timestamp'][0]) / 1000
    data['Timestamp_delta'] = data['Timestamp'].diff()
    data['Latitude_delta'] = data['Latitude'].diff()
    data['Longitude_delta'] = data['Longitude'].diff()
    data['Altitude_delta'] = data['Altitude'].diff()
    data['Course_delta'] = data['Course'].diff()
    data['Speed_delta'] = data['Speed'].diff()
    data['Bank_delta'] = data['Bank'].diff()
    data['Pitch_delta'] = data['Pitch'].diff()
    data['Altitude_rate'] = data['Altitude_delta'] / resample_rate * 60
    data['Speed_rate'] = data['Speed_delta'] / resample_rate
    data['Course_rate'] = data['Course_delta'] / resample_rate
    data['Bank_rate'] = data['Bank_delta'] / resample_rate
    data['Pitch_rate'] = data['Pitch_delta'] / resample_rate
    data = data[pd.notnull(data['Timestamp'])] 
    trackdata = trackdata.append(data)
    trackdata['Datapoint'] = trackdata.reset_index().index

# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.

lats = trackdata['Latitude'].values
lons = trackdata['Longitude'].values
color = trackdata['Altitude'].values
             
lllat = lats.min()-0.1
urlat = lats.max()+0.1
lllon = lons.min()-0.1
urlon = lons.max()+0.1

fig = plt.figure()
m = Basemap(llcrnrlon=lllon,llcrnrlat=lllat,urcrnrlon=urlon,urcrnrlat=urlat, epsg=4269)
#http://server.arcgisonline.com/arcgis/rest/services, EPSG Number of America is 4269
x, y = m(lons,lats)  
m.scatter(x,y,marker=',',c=color,cmap='jet',s=0.5)
#wms_server = 'http://motherlode.ucar.edu:8080/thredds/wms/fmrc/NCEP/NAM/CONUS_12km/NCEP-NAM-CONUS_12km-noaaport_best.ncd?'
#m.wmsimage(wms_server, layers=['RAS_GOES_I4'], verbose=True)
#m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=4000, dpi=400, verbose=True)
#m.arcgisimage(service='World_Topo_Map', xpixels=8000, verbose=True)
#m.arcgisimage(service='NatGeo_World_Map', xpixels=12000, verbose=True)
m.arcgisimage(service='World_Imagery', xpixels=12000, verbose=True)
fig.tight_layout()
fig.savefig('foo.png', dpi=1200)