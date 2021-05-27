# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:34:03 2021

@author: farrospide
"""
# importar librerias
import geopandas as gpd
import os
import rasterstats
import xarray as xr
from matplotlib import pyplot as plt
import pandas as pd
import rasterio
import numpy as np
import plotly.graph_objects as go
import re
from hydroeval import evaluator, nse

def notna_rows(obj):
    notna_indices = []
    for index in obj.index:
        if obj.loc[index,:].notna().all():
            notna_indices.append(index)
        else:
            pass
    return obj.loc[notna_indices,:]

#%% Cargar la capa de cuencas
# path_catchment = '/home/faarrosp/Insync/farrospide@ciren.cl/' + \
#     'OneDrive Biz - Shared/AOHIA_ZC/Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/' + \
#         'Cuencas_DARH_2015.shp'
path_catchment = os.path.join('..',
                              'Etapa 1 y 2',
                              'GIS',
                              'Cuencas_DARH',
                              'Cuencas',
                              'Cuencas_DARH_2015.shp')
catchment_gdf = gpd.read_file(path_catchment)

# filtrar por cuenca de interes (0600 Rapel 1300 Maipo 0703 Maule 0701 Mata...)
filtro = catchment_gdf['COD_CUENCA'].isin(['0600', '1300', '0703', '0701'])
catchment_gdf = catchment_gdf[filtro]
catchment_gdf = catchment_gdf.to_crs('EPSG:4326')

# definir la bbox a cortar
xmin = catchment_gdf.geometry.bounds['minx'].values.min()
xmax = catchment_gdf.geometry.bounds['maxx'].values.max()
ymin = catchment_gdf.geometry.bounds['miny'].values.min()
ymax = catchment_gdf.geometry.bounds['maxy'].values.max()

#%% Reproyectar con gdalwarp
# definir la ruta a los tiffs a reproyectar y cortar (reproject and crop)
path_tiffs_src = ['../Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CDR/raw',
                  '../Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CCS/raw']


# definir la ruta de destino
path_tiffs_dst = \
    ['../Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CDR/reprojected_epsg32719',
     '../Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CCS/reprojected_epsg32719']

for srcfolder, dstfolder in zip(path_tiffs_src, path_tiffs_dst):

    # definir arreglo con los archivos tiff a reproyectar
    files = [x for x in os.listdir(srcfolder) if x.endswith('.tif')]
    files.sort()
    
    
    # reproyectar y cortar usando gdal_warp (reproject and crop)
    for file in files:
        src = '/'.join([srcfolder, file])
        dst = '/'.join([dstfolder, file])
        src = src.replace(' ', '\\ ')
        dst = dst.replace(' ', '\\ ')
        os.system(' '.join(['gdalwarp',
                            '-s_srs EPSG:4326',
                            '-t_srs EPSG:32719',
                            '-te',
                            str(xmin),
                            str(ymin),
                            str(xmax),
                            str(ymax),
                            # '-overwrite',
                            src,
                            dst]))
#%% Transformar a netcdf
path = os.path.join('..',
                    'Etapa 1 y 2',
                    'GIS',
                    'PERSIANN')

# os.chdir('/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/AOHIA_ZC/Etapa 1 y 2/GIS/PERSIANN/')
os.chdir(path)

wd = ['./PERSIANN-CDR/reprojected_epsg32719',
     './PERSIANN-CCS/reprojected_epsg32719']

dd = ['./PERSIANN-CDR/netcdf_epsg32719',
     './PERSIANN-CCS/netcdf_epsg32719']

for srcfolder, dstfolder in zip(wd, dd):
    files = [x for x in os.listdir(srcfolder) if x.endswith('.tif')]
    files.sort()
    
    for file in files:
        src = '/'.join([srcfolder, file])
        dst = '/'.join([dstfolder, file[:-3] + 'nc'])
        src = src.replace(' ', '\\ ')
        dst = dst.replace(' ', '\\ ')
        os.system(' '.join(['gdal_translate',
                            '-of NetCDF',
                            src,
                            dst]))
#%% Assign time variable

wd = ['/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CCS/netcdf_epsg32719',
      '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CDR/netcdf_epsg32719']

for srcfolder in wd:
    os.chdir(srcfolder)
    files = [x for x in os.listdir() if x.endswith('.nc')]
    files.sort()
    for file in files:
        if 'CDR' in file:
            yr, mon, day = file[-12:-8], file[-8:-6], file[-6:-4]
        else:
            yr, mon, day = file[-11:-7], file[-7:-5], file[-5:-3]
        src = file
        dst = './timefiles/' + src
        os.system(' '.join(['cdo',
                        # '-O',
                        '-setdate,' + yr + '-' + mon + '-' + day,
                        '-setcalendar,proleptic_gregorian',
                        '-settunits,days',
                        src,
                        dst,
                        ]))

#%% Concatenar en tiempo

wd = ['/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CCS/netcdf_epsg32719/' + \
      'timefiles',
      '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CDR/netcdf_epsg32719/' + \
      'timefiles']


for srcfolder in wd[1:2]:
    os.chdir(srcfolder)
    files = [x for x in os.listdir()]
    files.sort()
    filelist = ' '.join(files)
    dst = '../../pp.nc'
    print(' '.join(['cdo cat',
                        filelist,
                        dst]))
