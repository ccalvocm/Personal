# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:21:15 2021

@author: farrospide
"""
import xarray as xr
import os
import rasterstats
import rasterio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine

#%% importar puntos de estaciones

paths = ['/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Maipo/RIO MAIPO_P_diario.xlsx',
      '/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Rapel/RIO RAPEL_P_diario.xlsx',
      '/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Mataquito/RIO MATAQUITO_P_diario.xlsx',
      '/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Maule/RIO MAULE_P_diario.xlsx']

path_Windows = os.path.join('C:', os.sep, 'Users', 'farrospide', 'GitHub',
                            'Analisis-Oferta-Hidrica', 'DGA', 'datosDGA', 'Pp')
path_maipop = os.path.join(path_Windows, 'Maipo',
                           'RIO MAIPO_P_diario.xlsx')    
path_rapelp = os.path.join(path_Windows, 'Rapel',
                           'RIO RAPEL_P_diario.xlsx')
path_mataquitop = os.path.join(path_Windows, 'Mataquito',
                           'RIO MATAQUITO_P_diario.xlsx') 
path_maulep = os.path.join(path_Windows, 'Maule',
                           'RIO MAULE_P_diario.xlsx')    
   
paths = [path_maipop, path_rapelp, path_mataquitop, path_maulep]    


stations_gdf_array = []

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
    # if 'MAIPO' in path or 'RAPEL' in path:
    #     crsstring = 'EPSG:32719'
    # elif 'MAULE' in path or 'MATAQUITO' in path:
    #     crsstring = 'EPSG:32718'
    gdf = gpd.GeoDataFrame(stations,
                                    geometry = gpd.points_from_xy(stations['londd'],
                                                                  stations['latdd']),
                                    crs = 'EPSG:4326')
    # if ('MATAQUITO' in path) or ('MAULE' in path):
    #     gdf = gdf.to_crs('EPSG:32719')
    # else:
    #     pass
    stations_gdf_array.append(gdf)
    
stations_gdf = pd.concat(stations_gdf_array)
# stations_gdf.plot()
# stations_gdf = stations_gdf.to_crs('EPSG:32719')
stations_gdf = stations_gdf.reset_index(drop = True)
del stations_gdf_array
#%% try to use rasterio
path = os.path.join('netcdf:..',
                    'Etapa 1 y 2',
                    'GIS',
                    'GPM',
                    'concat',
                    'GPM_concat.nc:precipitationCal')


with rasterio.open(path) as src:
# src = rasterio.open(path)
    affine_rio = src.transform
# array = src.read(1)
    lr = src.transform * (src.width, src.height)
    print('\n'.join(['bands: ' + str(src.count),
                     'width: ' + str(src.width),
                     'height: ' + str(src.height),
                     'affine:\n' + str(src.transform),
                     'bounds: ' + str(src.bounds),
                     'lower right: ' + str(lr),
                     'crs: ' + str(src.crs)]))
    band = src.read(1)
    # plt.imshow(src.read(1), cmap='pink')
    # plt.show()

#%% Get the points
var = 'precipitationCal'
station_coords = list(stations_gdf.geometry)
precip_values = []

path = os.path.join('..',
                    'Etapa 1 y 2',
                    'GIS',
                    'GPM',
                    'concat',
                    'GPM_concat.nc')

# with xr.open_rasterio(path) as da:
#     affine_newxr = Affine.from_gdal(*da.attrs['transform'])

with xr.open_dataset(path) as ds:
    west = ds.lon.min()
    south = ds.lat.min()
    east = ds.lon.max()
    north = ds.lat.max()
    width, height = ds.dims['lon'], ds.dims['lat']
    affine_xr = rasterio.transform.from_bounds(west, south,
                                            east, north,
                                            width, height)
    affine_xr_new = Affine(0.23, 0.0, -72.75, 0, -0.23, -33.12)
    print(ds)
    print(ds['time'])
    print(ds.coords['lon'])
    print(ds.coords['lat'])
    print(ds[var][0,:,:])
    arr = ds[var][0,:,:]
    timelength = ds[var].shape[0]
    print(affine_rio)
    print(affine_xr)
    
    for i in range(timelength):
        print(i/timelength * 100)
        arr = np.array(ds[var][i,:,:])
        a = rasterstats.point_query(vectors = station_coords,
                                                          raster = arr,
                                                          affine = affine_xr_new,
                                                          nodata = np.nan)
        precip_values.append(a)
        
#%% reorder the array to have the station name first, then the time array
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
    
with xr.open_dataset(path) as ds:
    time = ds.time.values
    
df = pd.DataFrame(index = time, columns = master.keys())
for key in master.keys():
    df[key] = master[key]['Timeseries']


outpath = os.path.join('.',
                       'outputs',
                       'productos_grillados',
                       'GPM.xlsx')

with pd.ExcelWriter(outpath) as writer:  
    df.to_excel(writer, sheet_name='serietiempo')
    stations_gdf.to_excel(writer, sheet_name='info estacion')