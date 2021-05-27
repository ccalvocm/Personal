# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:11:23 2021

@author: farrospide
"""
import pandas as pd
import os
import geopandas as gpd
from unidecode import unidecode
import re
import numpy as np
from matplotlib import pyplot as plt
import contextily as ctx
from pyproj import Transformer
import os

path_file = os.path.join('..', 'Etapa 1 y 2', 'DAA', 'ADM5395PlanillaControlSeguimiento.xlsx')

df = pd.read_excel(path_file, header = 0, dtype = str)

# funciones >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def clean_headers(dataframe):
        
    header = []
    
    for column in dataframe.columns:
        new_col = str(column).replace('\n', ' ')
        new_col = re.sub(' +', ' ', new_col)
        new_col = new_col.lstrip()
        new_col = new_col.rstrip()
        header.append(new_col)
        
    dataframe.columns = header
    dataframe.reset_index(inplace = True, drop = True)
    
    return dataframe

def clean_spaces(dataframe, columns):
    
    for column in columns:
        dataframe[column] = dataframe[column].str.lstrip()
        dataframe[column] = dataframe[column].str.rstrip()
        
    return dataframe


def clean_character(dataframe, columns, char1, char2):
    
    for column in columns:
        dataframe[column] = dataframe[column].str.replace(char1, char2)
        
    return dataframe

def column_to_numeric(dataframe, columns):
    
    for column in columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors = 'ignore')
        
    return dataframe

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#%% Pretratamiento de planilla

df = clean_headers(df)


#%% Filtrado y separacion de coordenadas validas en orden de magnitud

columns = ['Coordenada Norte Captación', 'Coordenada Este Captación',
           'Coordenada Norte Restitución', 'Coordenada Este Restitución']

df = clean_spaces(df, columns)
df = clean_character(df, columns, ',', '.')
df = clean_character(df, columns, chr(8208), '')
# recordar que el hyphen del teclado no es el mismo que el de la planilla, que
# es ASCII 8208
# df = column_to_numeric(df, columns)

invalid_char = ['nan', '-', '0', '0.0', '', np.nan]

valid_rows = []
invalid_rows = []
no_dat_huso = []
for index in df.index:
    utme = df.loc[index, 'Coordenada Este Restitución']
    utmn = df.loc[index, 'Coordenada Norte Restitución']
    c1, c2 = [x in invalid_char for x in [utme, utmn]]
    # ninf, nsup = [utmn < 1e6, utmn>1e7]
    if not((c1 or c2)):
        valid_rows.append(index)
    else:
        invalid_rows.append(index)
    
    
df_restituciones = df.loc[valid_rows,:] 
df_restituciones = column_to_numeric(df_restituciones, ['Coordenada Este Restitución',
                                                        'Coordenada Norte Restitución'])

df_res_noCoords = df.loc[invalid_rows,:]




fil_E = (df_restituciones['Coordenada Este Restitución'] > 1e5) & \
    (df_restituciones['Coordenada Este Restitución'] < 1e6)

fil_N = (df_restituciones['Coordenada Norte Restitución'] > 1e6) & \
    (df_restituciones['Coordenada Norte Restitución'] < 1e7)

df_restituciones = df_restituciones[fil_E & fil_N]

valid_rows = []
invalid_rows = []
for index in df.index:
    utme = df.loc[index, 'Coordenada Este Captación']
    utmn = df.loc[index, 'Coordenada Norte Captación']
    
    c1, c2 = [x in invalid_char for x in [utme, utmn]]
    if not((c1 or c2)):
        valid_rows.append(index)
    else:
        invalid_rows.append(index)

df_cap_noCoords = df.loc[invalid_rows,:]



df_captaciones = df.loc[valid_rows,:] 
df_captaciones = column_to_numeric(df_captaciones, ['Coordenada Este Captación',
                                                        'Coordenada Norte Captación'])

fil_E = (df_captaciones['Coordenada Este Captación'] > 1e5) & \
    (df_captaciones['Coordenada Este Captación'] < 1e6)

fil_N = (df_captaciones['Coordenada Norte Captación'] > 1e6) & \
    (df_captaciones['Coordenada Norte Captación'] < 1e7)

df_captaciones = df_captaciones[fil_E & fil_N]
# filter_utm = df['Tipo de Coordenada'] == 'UTM'
# df_utm = df[filter_utm].copy()

#%% Asignar coordenadas UTM 32719

CRS = {'1984': {'18': 'EPSG:32718', '19': 'EPSG:32719'},
       '1969': {'18': 'EPSG:29188', '19': 'EPSG:29189'},
       '1956': {'18': 'EPSG:24878', '19': 'EPSG:24879'},
       '84': {'18': 'EPSG:32718', '19': 'EPSG:32719'}}

dfs_no_dat_no_hus = []
for dataframe, tipo in zip([df_captaciones, df_restituciones],
                                ['Captación', 'Restitución']):
    
    dataframe['x'] = 0.0
    dataframe['y'] = 0.0
    dataframe['CRS'] = 'NO'
    
    no_dat_no_hus_idx = []
    
    for index in dataframe.index:
        if dataframe.loc[index, 'Cuenca'] in ['18', '19']:
            h = dataframe.loc[index, 'Cuenca']
        else:
            h = dataframe.loc[index, 'Huso']
        
        d = dataframe.loc[index, 'Datum']
        try:
            dataframe.loc[index, 'CRS'] = CRS[d][h]
            dataframe.loc[index, 'x'] = abs(float(dataframe.loc[index, 'Coordenada Este ' + tipo]))
            dataframe.loc[index, 'y'] = abs(float(dataframe.loc[index, 'Coordenada Norte ' + tipo]))
    
        except:
            dataframe.loc[index, 'CRS'] = 'ERROR'
        if d == '0' or h == '0':
            no_dat_no_hus_idx.append(index)
        else:
            pass
    dfs_no_dat_no_hus.append(dataframe.loc[no_dat_no_hus_idx,:])
            
    # transformacion a EPSG 32719    
    dataframe['x_32719'] = 0.0
    dataframe['y_32719'] = 0.0
    
    projectionsR = [x for x in dataframe['CRS'].unique() if x not in ['ERROR']]
    transformers = {}
    for proj in projectionsR:
        transformers[proj] = Transformer.from_crs(proj, 'EPSG:32719', always_xy=True)
#%%    
    for index, idx in enumerate(dataframe.index):
        print(index)
        CRSog = dataframe.loc[idx, 'CRS']
        if CRSog not in ['EPSG:32719', 'ERROR']:
            x = dataframe.loc[idx, 'x']
            y = dataframe.loc[idx, 'y']
            newx, newy = transformers[CRSog].transform(x,y)
            dataframe.loc[idx,'x_32719'] = newx
            dataframe.loc[idx,'y_32719'] = newy
        elif CRSog == 'EPSG:32719':
            x = dataframe.loc[idx, 'x']
            y = dataframe.loc[idx, 'y']
            dataframe.loc[idx,'x_32719'] = x
            dataframe.loc[idx,'y_32719'] = y
        else:
            pass
        
    # exportar a .shp
    for i in range(dataframe.shape[0]):
        for j in range(dataframe.shape[1]):
            if type(dataframe.iloc[i,j]) == str:
                dataframe.iloc[i,j] = unidecode(dataframe.iloc[i,j])
            else:
                pass
    columnas = [unidecode(x) for x in dataframe.columns]
    dataframe.columns = columnas


filtro_cap = df_captaciones[df_captaciones['x_32719'] > 0]
filtro_res = df_restituciones[df_restituciones['x_32719'] > 0]

gdf_restituciones = gpd.GeoDataFrame(filtro_res, crs = 'EPSG:32719',
                                     geometry = gpd.points_from_xy(filtro_res['x_32719'],
                                                                   filtro_res['y_32719']))

gdf_captaciones = gpd.GeoDataFrame(filtro_cap, crs = 'EPSG:32719',
                                     geometry = gpd.points_from_xy(filtro_cap['x_32719'],
                                                                   filtro_cap['y_32719']))

#%% save into GIS format and spreadsheets

ruta = os.path.join("..", "Etapa 1 y 2", "DAA")
nombre_archivo = "DAA_NEW_SOURCE_INVALID_ROWS.xlsx"
ruta_final = os.path.join(ruta, nombre_archivo)

with pd.ExcelWriter(ruta_final) as writer:
    df_cap_noCoords.to_excel(writer, sheet_name = 'captaciones', index = False)
    df_res_noCoords.to_excel(writer, sheet_name = 'restituciones', index = False)
    dfs_no_dat_no_hus[0].to_excel(writer, sheet_name = 'cap-sin-dat-o-hus', index = False)
    dfs_no_dat_no_hus[1].to_excel(writer, sheet_name = 'res-sin-dat-o-hus', index = False)


path = os.path.join('..',
                    'Etapa 1 y 2',
                    'DAA')

filename1 = os.path.join('shapes_output',
                        'restituciones_post_all_NEW_SOURCE.geojson')

filename2 = os.path.join('shapes_output',
                        'captaciones_post_all_NEW_SOURCE.geojson')

outpath1 = os.path.join(path, filename1)
outpath2 = os.path.join(path, filename2)

gdf_restituciones.to_file(outpath1, driver = 'GeoJSON')
gdf_captaciones.to_file(outpath2, driver = 'GeoJSON')
