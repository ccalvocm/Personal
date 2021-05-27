# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:32:45 2021

@author: farrospide
"""
import os
import pandas as pd
from unidecode import unidecode
import math
import numpy as np
from pyproj import Transformer
import geopandas as gpd

def latlon2dd(num):
    invalid = ['nan', np.nan, 0, '', ' ', '0', 'S/N']
    if num not in invalid:
        num=str(num)
        if num not in "nan":
            num=str(num)
            grados=float(num[0:2])
            minutos=float(num[2:4])/60
            segundos=float(num[4:6])/3600
            dd=-(grados+minutos+segundos)
            return dd
        else:
            return num
    else:
        return np.nan

#%%
DAA = 'MAIPO'
path = os.path.join('..', 'Etapa 1 y 2', 'DAA', 'DAA_CFA', 'Tablas_DAA', DAA +\
                    '.xlsx')

df = pd.read_excel(path, dtype = {'Huso': str, 'Datum UTM': str, 'Datum': str,
                                  'Latitud Captación': str, 'Longitud Captación': str})

columns = df.columns

new_columns = []

for col in columns:
    new_columns.append(unidecode(col))
    
df.columns = new_columns
    
#%% 

CRS = {'1984': {'18': 'EPSG:32718', '19': 'EPSG:32719'},
       '1969': {'18': 'EPSG:29188', '19': 'EPSG:29189'},
       '1956': {'18': 'EPSG:24878', '19': 'EPSG:24879'},
       '84': {'18': 'EPSG:32718', '19': 'EPSG:32719'}}

CRSg = {'1984': 'EPSG:4326',
        '1956': 'EPSG:4248',
        '1969': 'EPSG:4724'}

nan_rows = df['UTM Norte Captacion WGS84 (m)'].isna() | \
    df['UTM Este Captacion WGS84 (m)'].isna()
    
df_capt = df.copy()

df_capt['x_32719'] = df[~nan_rows]['UTM Este Captacion WGS84 (m)']
df_capt['y_32719'] = df[~nan_rows]['UTM Norte Captacion WGS84 (m)']
df_capt['x'] = 0.0
df_capt['y'] = 0.0
df_capt['CRS'] = 'NONE'

nan_coords_indices = []
for index in df_capt.index:
    d = df_capt.loc[index, 'Datum UTM']
    h = df_capt.loc[index, 'Huso']
    nanx = math.isnan(df_capt.loc[index, 'x_32719'])
    nany = math.isnan(df_capt.loc[index, 'y_32719'])
    if nanx or nany:
        nan_coords_indices.append(index)
    else:
        pass
    try:
        df_capt.loc[index, 'CRS'] = CRS[d][h]
    except:
        pass

#%% Usar o UTM o latlon

for index in nan_coords_indices:
    crs = df_capt.loc[index, 'CRS']
    xutm = df_capt.loc[index, 'UTM Este Captacion (m)']
    yutm = df_capt.loc[index, 'UTM Norte Captacion (m)']
    if math.isnan(xutm) or math.isnan(yutm):
        lat = df_capt.loc[index, 'Latitud Captacion']
        lon = df_capt.loc[index, 'Longitud Captacion']
        lat = latlon2dd(lat)
        lon = latlon2dd(lon)
        dat = str(df_capt.loc[index, 'Datum'])
        if math.isnan(lat) or math.isnan(lon) or (dat in ['nan', 'S/N']):
            pass
        else:
            # print(index, dat)
            df_capt.loc[index, 'x'] = lon
            df_capt.loc[index, 'y'] = lat
            df_capt.loc[index, 'CRS'] = CRSg[dat]
    else:
        print(index)
        df_capt.loc[index, 'x'] = xutm
        df_capt.loc[index, 'y'] = yutm

transformable = (df_capt['x']) != 0 & (df_capt['y'] != 0)

#%% Reproyectar

projectionsR = [x for x in df_capt[transformable]['CRS'].unique() if x not in ['NONE', 'EPSG:32719']]
transformers = {}
for proj in projectionsR:
    transformers[proj] = Transformer.from_crs(proj, 'EPSG:32719', always_xy=True)
    
for index, idx in enumerate(nan_coords_indices):
        print(index)
        CRSog = df_capt.loc[idx, 'CRS']
        if CRSog not in ['EPSG:32719', 'NONE']:
            x = df_capt.loc[idx, 'x']
            y = df_capt.loc[idx, 'y']
            newx, newy = transformers[CRSog].transform(x,y)
            df_capt.loc[idx,'x_32719'] = newx
            df_capt.loc[idx,'y_32719'] = newy
        elif CRSog == 'EPSG:32719':
            x = df_capt.loc[idx, 'x']
            y = df_capt.loc[idx, 'y']
            df_capt.loc[idx,'x_32719'] = x
            df_capt.loc[idx,'y_32719'] = y
        else:
            pass

#%% Crear el geodataframe
georreferenciable = ~df_capt['x_32719'].isna() & \
    ~df_capt['y_32719'].isna() & (df_capt['x_32719'] != 0) & \
        (df_capt['y_32719'] != 0)


df_capt_georref = df_capt[georreferenciable].copy()

# utilizar unidecode en todos los campos del dataframe
for i in range(df_capt_georref.shape[0]):
    for j in range(df_capt_georref.shape[1]):
        if type(df_capt_georref.iloc[i,j]) == str:
            df_capt_georref.iloc[i,j] = unidecode(df_capt_georref.iloc[i,j])
        else:
            pass

columnas = [unidecode(x) for x in df_capt_georref.columns]
df_capt_georref.columns = columnas

gdf_capt = gpd.GeoDataFrame(df_capt_georref, crs = 'EPSG:32719',
                                     geometry = gpd.points_from_xy(df_capt_georref['x_32719'],
                                                                   df_capt_georref['y_32719']))

filename2 = os.path.join('shapes_output',
                        'captaciones_post_' + DAA + '_CFA.shp')
path = os.path.join('..',
                    'Etapa 1 y 2',
                    'DAA')

outpath2 = os.path.join(path, filename2)

if filename2[-3:] == 'shp':
    datefields = ['Fecha Ingreso Gobernacion',
                  'Fecha Ingreso DGA',
                  'Fecha de Resolucion/Envio al Juez',
                  'Fecha Toma Razon']
else:
    pass

for field in datefields:
    gdf_capt[field] = gdf_capt[field].astype(str)

gdf_capt.to_file(outpath2, driver = 'ESRI Shapefile')