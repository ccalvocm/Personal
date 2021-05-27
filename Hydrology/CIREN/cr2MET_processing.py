#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:07:09 2021

@author: faarrosp
"""

# import libraries
from netCDF4 import Dataset
import rasterio
import os
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterstats
import numpy as np

# change to working directory
os.chdir('/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
         'AOHIA_ZC/Etapa 1 y 2/GIS/cr2MET')

# load cr2 netCDF file    
ds = xr.open_dataset('cr2MET_coords.nc')


# load station data (define paths)
paths = ['/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Maipo/RIO MAIPO_P_diario.xlsx',
      '/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Rapel/RIO RAPEL_P_diario.xlsx',
      '/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Mataquito/RIO MATAQUITO_P_diario.xlsx',
      '/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Maule/RIO MAULE_P_diario.xlsx']

# define array of geometries (empty first)    
stations_gdf_array = []    
    
# read each excel and get the coordinates in decimal degrees (DD) EPSG:4326    
for path in paths:
    stations = pd.read_excel(path, sheet_name = 'info estacion')
    stations[['latdd', 'londd']] = float()
    for index in stations.index:
        latstr = stations.loc[index, 'Latitud'].split(" ")
        d,m,s = latstr[0][:2], latstr[1][:2], latstr[2][:2]
        stations.loc[index, 'latdd'] = -(float(d) + float(m)/60 + float(s)/60/60)
        lonstr = stations.loc[index, 'Longitud'].split(" ")
        d,m,s = lonstr[0][:2], lonstr[1][:2], lonstr[2][:2]
        stations.loc[index, 'londd'] = -(float(d) + float(m)/60 + float(s)/60/60)
    gdf = gpd.GeoDataFrame(stations,
                                    geometry = gpd.points_from_xy(stations['londd'],
                                                                  stations['latdd']),
                                    crs = 'EPSG:4326')   
    stations_gdf_array.append(gdf)

# concatenate all dataframes into 1 big dataframe    
stations_gdf = pd.concat(stations_gdf_array)

# reset index to avoid having duplicate indices
stations_gdf = stations_gdf.reset_index(drop = True)

# free up some memory
del stations_gdf_array


# define the affine transformation for rasterstats module    
west = -72.7000030649119395
south = -36.6000007733906827
east = -69.7500015127247792
north = -32.8999992266093244
width, height = 59, 74
affccs = rasterio.transform.from_bounds(west, south,
                                        east, north,
                                        width, height)




# define the list of points (DGA stations)
station_coords = list(stations_gdf.geometry)

# define the array of precipitation values for each station
precip_values = []

# use rasterstats module to get the point values
for band in list(ds.data_vars)[:-1]:
    print(band)
    a = rasterstats.point_query(vectors = station_coords,
                                                         raster = ds[band].values,
                                                         affine = affccs,
                                                         nodata = np.nan)
    precip_values.append(a)
    

# reorder the array to have the station name first, then the time array
precip_ts = []
for i in range(len(precip_values[0])):
    precip_ts.append([x[i] for x in precip_values])

master = {}    
for ts, index in zip(precip_ts, stations_gdf.index):
    code = stations_gdf.loc[index, 'Codigo Estacion']
    stts = {}
    stts['Nombre estacion'] = stations_gdf.loc[index, 'Nombre estacion']
    stts['Codigo estacion'] = stations_gdf.loc[index, 'Codigo Estacion']
    a = np.array(ts)
    a[a<0] = np.nan
    stts['Timeseries'] = a
    master[code] = stts
    


#%%
date = pd.to_datetime("1979-01-01")

date = date + pd.to_timedelta(np.arange(15096), 'D')

df = pd.DataFrame(index = date, columns = master.keys())
for key in master.keys():
    df[key] = master[key]['Timeseries']


outpath = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/scripts/outputs/productos_grillados/'
name = 'cr2MET'

with pd.ExcelWriter(outpath + name + '.xlsx') as writer:  
    df.to_excel(writer, sheet_name='serietiempo')
    stations_gdf.to_excel(writer, sheet_name='info estacion')

# ds.close()