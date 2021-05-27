#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:07:57 2021

@author: faarrosp
"""

import os
import rasterstats
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio

# change directory
os.chdir('/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
         'AOHIA_ZC/Etapa 1 y 2/GIS/GPM/concat')
    
# load GPM netCDF file    
ds = xr.open_dataset('GPM_concat.nc')

west = ds.lon.min()
south = ds.lat.min()
east = ds.lon.max()
north = ds.lat.max()
width, height = ds.dims['lon'], ds.dims['lat']
affccs = rasterio.transform.from_bounds(west, south,
                                        east, north,
                                        width, height)


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

# define the list of points (DGA stations)
station_coords = list(stations_gdf.geometry)

# define the array of precipitation values for each station
precip_values = []

# use rasterstats module to get the point values
for time in range(ds['precipitationCal'].shape[0]):
    print(time/ds['precipitationCal'].shape[0] * 100)
    # data = np.array(ds['precipitationCal'][time,:,:].values)
    a = rasterstats.point_query(vectors = station_coords,
                                                          raster = ds['precipitationCal'],
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