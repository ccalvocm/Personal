#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:18:04 2020

@author: felipe
"""
# importar librerias
import pandas as pd
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
import contextily as ctx
from pyproj import Transformer
from unidecode import unidecode
import sys
import math
import os

# definir funciones
def NestedDictValues(d):
  for v in d.values():
    if isinstance(v, dict):
      yield from NestedDictValues(v)
    else:
      yield v

def clean_character(dataframe, columns, char1, char2):
    
    for column in columns:
        dataframe[column] = dataframe[column].str.replace(char1, char2)
        
    return dataframe

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
        return num
    
    
def checkUTMmagnitudesN(num):
    invalid = ['nan', np.nan, 0, '', ' ', '0', 'S/N']
    if num not in invalid:
        if type(num) == 'str':
            num = float(num)
        else:
            pass
        if num < 1e6 or num > 1e7:
            exponent = math.floor(math.log10(num))
            exponent = 6 - exponent
            num = num * (10**exponent)
        else:
            pass
        return num
    else:
        return num
    

def checkUTMmagnitudesE(num):
    invalid = ['nan', np.nan, 0, '', ' ', '0', 'S/N']
    if num not in invalid:
        if type(num) == 'str':
            num = float(num)
        else:
            pass
        if num < 1e5 or num > 1e6:
            exponent = math.floor(math.log10(num))
            exponent = 5 - exponent
            num = num * (10**exponent)
        else:
            pass
        return num
    else:
        return num

# lee la planilla excel que contiene las entradas con coordenadas
path = os.path.join('..',
                    'Etapa 1 y 2',
                    'DAA')

filename = 'DAA_filtro_fecha.xlsx'
filepath = os.path.join(path,
                        filename)



df = pd.read_excel(filepath,
                   sheet_name="c-coord-post")


# limpiar espacios en las columnas
df = clean_character(df, ['Datum',
                          'Datum.1',
                          'Huso'], ' ', '')

#%%

# transformar lat lon a DD (decimal degrees)
for index in df.index:
    num=df.loc[index,"Longitud Captación"]
    df.loc[index,"Longitud Captación"]=latlon2dd(num)
    num=df.loc[index,"Latitud Captación"]
    df.loc[index,"Latitud Captación"]=latlon2dd(num)
    num=df.loc[index,"Latitud Restitución"]
    df.loc[index,"Latitud Restitución"]=latlon2dd(num)
    num=df.loc[index,"Longitud Restitución"]
    df.loc[index,"Longitud Restitución"]=latlon2dd(num)
    
# checkear magnitud de las UTM (miles o miles de millones)
for index in df.index:
    num = df.loc[index, 'UTM Norte Captación (m)']
    df.loc[index, 'UTM Norte Captación (m)'] = checkUTMmagnitudesN(num)
    num = df.loc[index, 'UTM Este Captación (m)']
    df.loc[index, 'UTM Este Captación (m)'] = checkUTMmagnitudesE(num)
    num = df.loc[index, 'UTM Norte Restitución (m)']
    df.loc[index, 'UTM Norte Restitución (m)'] = checkUTMmagnitudesN(num)
    num = df.loc[index, 'UTM Este Restitución (m)']
    df.loc[index, 'UTM Este Restitución (m)'] = checkUTMmagnitudesE(num)

# transformar datetime en str para exportar a shp
df['Fecha de Resolución/ Envío al Juez/ Inscripción C.B.R.'] = \
    df['Fecha de Resolución/ Envío al Juez/ Inscripción C.B.R.'].astype(str)


#%% Separar captaciones de restituciones (OK)


# wUTMRestitucion (with UTM Restitucion): selecciona todos los campos donde
#   el valor UTM Este Restitucion o UTM Norte Restitucion es igual a 0

# define posibles caracteres invalidos para el analisis
invalid = ['nan', np.nan, 0, '', ' ', '0', 'S/N']

valid_rows = []
for index in df.index:
    utme = df.loc[index, 'UTM Este Restitución (m)']
    utmn = df.loc[index, 'UTM Norte Restitución (m)']
    lat = df.loc[index, 'Latitud Restitución']
    lon = df.loc[index, 'Longitud Restitución']
    c1, c2, c3, c4 = [x in invalid for x in [utme, utmn, lat, lon]]
    if not((c1 or c2) and (c3 or c4)):
        valid_rows.append(index)
    else:
        pass
        
df_restituciones = df.loc[valid_rows,:] 

valid_rows = []
for index in df.index:
    utme = df.loc[index, 'UTM Este Captación (m)']
    utmn = df.loc[index, 'UTM Norte Captación (m)']
    lat = df.loc[index, 'Latitud Captación']
    lon = df.loc[index, 'Longitud Captación']
    c1, c2, c3, c4 = [x in invalid for x in [utme, utmn, lat, lon]]
    if not((c1 or c2) and (c3 or c4)):
        valid_rows.append(index)
    else:
        pass
df_captaciones = df.loc[valid_rows,:]

# Filtrar por registros georreferenciables (OK)
valid_rows = []
for index in df_restituciones.index:
    d1, d2 = df.loc[index, 'Datum'], df.loc[index, 'Datum.1']
    c1, c2 = [x in invalid for x in [d1,d2]]
    if not (c1 and c2):
        valid_rows.append(index)
    else:
        pass
df_restituciones_georref = df_restituciones.loc[valid_rows,:]

valid_rows = []
for index in df_captaciones.index:
    d1, d2 = df.loc[index, 'Datum'], df.loc[index, 'Datum.1']
    c1, c2 = [x in invalid for x in [d1,d2]]
    if not (c1 and c2):
        valid_rows.append(index)
    else:
        pass
df_captaciones_georref = df_captaciones.loc[valid_rows,:]

#%% Identificar sistema coordenado y asignar codigo y columnas

CRS = {'1984': {'18': 'EPSG:32718', '19': 'EPSG:32719'},
       '1969': {'18': 'EPSG:29188', '19': 'EPSG:29189'},
       '1956': {'18': 'EPSG:24878', '19': 'EPSG:24879'} }

CRSg = {'1984': 'EPSG:4326', '1969': 'EPSG:4291', '1956': 'EPSG:4248'}

df_restituciones_georref['CRS'] = 'NO'
df_restituciones_georref['x'] = 0.0
df_restituciones_georref['y'] = 0.0

df_captaciones_georref['CRS'] = 'NO'
df_captaciones_georref['x'] = 0.0
df_captaciones_georref['y'] = 0.0

# Datum
datum = [x for x in df['Datum'].unique() if x not in invalid]
datum1 = [x for x in df['Datum.1'].unique() if x not in invalid]

# Husos
huso = [x for x in df['Huso'].unique() if x not in invalid]

for index in df_restituciones_georref.index:
    h = df_restituciones_georref.loc[index, 'Huso']
    d = df_restituciones_georref.loc[index, 'Datum']
    if d in datum and h in huso:
        df_restituciones_georref.loc[index, 'CRS'] = CRS[d][h]
        df_restituciones_georref.loc[index, 'x'] = abs(float(df_restituciones_georref.loc[index, 'UTM Este Restitución (m)']))
        df_restituciones_georref.loc[index, 'y'] = abs(float(df_restituciones_georref.loc[index, 'UTM Norte Restitución (m)']))
    else:
        pass
    if df_restituciones_georref.loc[index, 'x'] <= 0 or \
        df_restituciones_georref.loc[index, 'y'] <= 0:
            d = df_restituciones_georref.loc[index, 'Datum.1']
            if d not in invalid:
                df_restituciones_georref.loc[index, 'CRS'] = CRSg[d]
                df_restituciones_georref.loc[index, 'x'] = df_restituciones_georref.loc[index, 'Longitud Restitución']
                df_restituciones_georref.loc[index, 'y'] = df_restituciones_georref.loc[index, 'Latitud Restitución']
            else:
                pass
    else:
        pass

for index in df_captaciones_georref.index:
    h = df_captaciones_georref.loc[index, 'Huso']
    d = df_captaciones_georref.loc[index, 'Datum']
    if d in datum and h in huso:
        df_captaciones_georref.loc[index, 'CRS'] = CRS[d][h]
        df_captaciones_georref.loc[index, 'x'] = abs(float(df_captaciones_georref.loc[index, 'UTM Este Captación (m)']))
        df_captaciones_georref.loc[index, 'y'] = abs(float(df_captaciones_georref.loc[index, 'UTM Norte Captación (m)']))
    else:
        pass
    if df_captaciones_georref.loc[index, 'x'] <= 0 or \
        df_captaciones_georref.loc[index, 'y'] <= 0:
            d = df_captaciones_georref.loc[index, 'Datum.1']
            if d not in invalid:
                df_captaciones_georref.loc[index, 'CRS'] = CRSg[d]
                df_captaciones_georref.loc[index, 'x'] = df_captaciones_georref.loc[index, 'Longitud Captación']
                df_captaciones_georref.loc[index, 'y'] = df_captaciones_georref.loc[index, 'Latitud Captación']
            else:
                pass
    else:
        pass

# Final check before reprojecting
filt = df_captaciones_georref['CRS'] != 'NO'
df_captaciones_georref = df_captaciones_georref[filt]

filt = df_restituciones_georref['CRS'] != 'NO'
df_restituciones_georref = df_restituciones_georref[filt]

    

#%% Reproyectar todo lo que no sea WGS 1984

# nuevas columnas de coordenadas definitivas
df_restituciones_georref['x_32719'] = 0
df_restituciones_georref['y_32719'] = 0

df_captaciones_georref['x_32719'] = 0
df_captaciones_georref['y_32719'] = 0


projectionsR = df_restituciones_georref['CRS'].unique()
transformers = {}
for proj in projectionsR:
    transformers[proj] = Transformer.from_crs(proj, 'EPSG:32719', always_xy=True)

for index, idx in enumerate(df_restituciones_georref.index):
    print(index)
    CRSog = df_restituciones_georref.loc[idx, 'CRS']
    if CRSog not in ['EPSG:32719']:
        x = df_restituciones_georref.loc[idx, 'x']
        y = df_restituciones_georref.loc[idx, 'y']
        newx, newy = transformers[CRSog].transform(x,y)
        df_restituciones_georref.loc[idx,'x_32719'] = newx
        df_restituciones_georref.loc[idx,'y_32719'] = newy
    elif CRSog == 'EPSG:32719':
        x = df_restituciones_georref.loc[idx, 'x']
        y = df_restituciones_georref.loc[idx, 'y']
        df_restituciones_georref.loc[idx,'x_32719'] = x
        df_restituciones_georref.loc[idx,'y_32719'] = y
    else:
        pass
    
projectionsC = df_captaciones_georref['CRS'].unique()
transformers = {}
for proj in projectionsC:
    transformers[proj] = Transformer.from_crs(proj, 'EPSG:32719', always_xy=True)
    
for idx in df_captaciones_georref.index:
    print(idx)
    CRSog = df_captaciones_georref.loc[idx, 'CRS']
    if CRSog not in ['EPSG:32719']:
        x = df_captaciones_georref.loc[idx, 'x']
        y = df_captaciones_georref.loc[idx, 'y']
        newx, newy = transformers[CRSog].transform(x,y)
        df_captaciones_georref.loc[idx,'x_32719'] = newx
        df_captaciones_georref.loc[idx,'y_32719'] = newy
    elif CRSog == 'EPSG:32719':
        x = df_captaciones_georref.loc[idx, 'x']
        y = df_captaciones_georref.loc[idx, 'y']
        df_captaciones_georref.loc[idx,'x_32719'] = x
        df_captaciones_georref.loc[idx,'y_32719'] = y
    else:
        pass

#%% exportar a shp

    
# utilizar unidecode en todos los campos del dataframe
for i in range(df_restituciones_georref.shape[0]):
    for j in range(df_restituciones_georref.shape[1]):
        if type(df_restituciones_georref.iloc[i,j]) == str:
            df_restituciones_georref.iloc[i,j] = unidecode(df_restituciones_georref.iloc[i,j])
        else:
            pass

# utilizar unidecode en todos los campos del dataframe
for i in range(df_captaciones_georref.shape[0]):
    for j in range(df_captaciones_georref.shape[1]):
        if type(df_captaciones_georref.iloc[i,j]) == str:
            df_captaciones_georref.iloc[i,j] = unidecode(df_captaciones_georref.iloc[i,j])
        else:
            pass


columnas = [unidecode(x) for x in df_restituciones_georref.columns]
df_restituciones_georref.columns = columnas
columnas = [unidecode(x) for x in df_captaciones_georref.columns]
df_captaciones_georref.columns = columnas


restituciones_gdf = gpd.GeoDataFrame(df_restituciones_georref, crs = 'EPSG:32719',
                                     geometry = gpd.points_from_xy(df_restituciones_georref['x_32719'],
                                                                   df_restituciones_georref['y_32719']))

captaciones_gdf = gpd.GeoDataFrame(df_captaciones_georref, crs = 'EPSG:32719',
                                     geometry = gpd.points_from_xy(df_captaciones_georref['x_32719'],
                                                                   df_captaciones_georref['y_32719']))

filename1 = os.path.join('shapes_output',
                        'restituciones_post_all.geojson')

filename2 = os.path.join('shapes_output',
                        'captaciones_post_all.geojson')

outpath1 = os.path.join(path, filename1)
outpath2 = os.path.join(path, filename2)

restituciones_gdf.to_file(outpath1, driver = 'GeoJSON')
captaciones_gdf.to_file(outpath2, driver = 'GeoJSON')

#%% Plotear
# fig = plt.figure()
# ax = fig.add_subplot(111)
# restituciones_gdf.plot(ax = ax, color = 'blue')
# captaciones_gdf.plot(ax = ax, color = 'red')


