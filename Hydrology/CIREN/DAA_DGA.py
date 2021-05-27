#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:49:06 2021

@author: faarrosp
"""

import pandas as pd
import os
from unidecode import unidecode
import numpy as np
from pyproj import Transformer
import geopandas as gpd

CRS = {'1984': {'18': 'EPSG:32718', '19': 'EPSG:32719'},
       '1969': {'18': 'EPSG:29188', '19': 'EPSG:29189'},
       '1956': {'18': 'EPSG:24878', '19': 'EPSG:24879'},
       '84': {'18': 'EPSG:32718', '19': 'EPSG:32719'}}

CRSg = {'1984': 'EPSG:4326',
        '1956': 'EPSG:4248',
        '1969': 'EPSG:4724'}

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

def import_sheet(path):
    df = pd.read_excel(path,
                       skiprows=6)
    df.columns = [unidecode(x.replace('\n', '')) for x in df.columns]
    newcols = []
    for col in df.columns:
        for s in ['/', '.', '|', '?']:
            col = col.replace(s,'')
        newcols.append(col)
    
    df.columns = newcols
    df['x'] = 0.0
    df['y'] = 0.0
    df['CRS_og'] = 'none'
    
    return df

folder = os.path.join('..', 'Etapa 1 y 2', 'DAA', 'DAA_CFA', 'Tablas_DAA')
name = 'Derechos_Concedidos_XVI_Region.xls'
fp = os.path.join(folder,name)

df = import_sheet(fp)
cols = df.columns

utm_tuples = [(41,40,43,42)] # (x,y,datum,huso)
geo_tuples = [(45,44,46)] # (xlon,ylat,datum)

def get_georreferenciables_UTM(df,utm_tuples):
    for tup in utm_tuples:
        wdatum = df.iloc[:,tup[2]].isin(['1984', '1956', '1969'])
        whuso = df.iloc[:,tup[3]].isin(['18', '19'])
        wx = (df.iloc[:,tup[0]] >= 0) & df.iloc[:,tup[0]].notna()
        wy = (df.iloc[:,tup[1]] >= 0) & df.iloc[:,tup[1]].notna() 
        filtro = wdatum & whuso & wx & wy
    return filtro

def replace_empty(s):
    if len(s) < 1:
        s = '0'
    else:
        pass
    return s

def get_georreferenciables_latlon(df,geo_tuples):
    for tup in geo_tuples:
        for element in tup:
            if df.iloc[:,element].dtypes == 'O':
                df.iloc[:,element] = df.iloc[:,element].str.replace(' ','')
                df.iloc[:,element] = df.iloc[:,element].str.replace('S/N','')
                df.iloc[:,element] = df.iloc[:,element].apply(replace_empty)
            else:
                pass
        wdatum = df.iloc[:,tup[2]].isin(['1984', '1956', '1969'])
        df.iloc[:,tup[0]] = df.iloc[:,tup[0]].astype(int)
        df.iloc[:,tup[1]] = df.iloc[:,tup[1]].astype(int)
        wx = (df.iloc[:,tup[0]] > 0) & df.iloc[:,tup[0]].notna()
        wy = (df.iloc[:,tup[1]] > 0) & df.iloc[:,tup[1]].notna() 
        filtro = wdatum & wx & wy
    return filtro

filtroUTM = get_georreferenciables_UTM(df,utm_tuples)
filtrogeo = get_georreferenciables_latlon(df,geo_tuples)

df_new = df[filtroUTM | filtrogeo].copy()

filtroUTM = get_georreferenciables_UTM(df_new,utm_tuples)
filtrogeo = get_georreferenciables_latlon(df_new,geo_tuples)

xcol = df_new.columns[41]
ycol = df_new.columns[40]
datcol = df_new.columns[43]
huscol = df_new.columns[42]

latcol = df_new.columns[44]
loncol = df_new.columns[45]
datgcol = df_new.columns[46]

for idx in df_new[filtroUTM].index:
    d = df_new.loc[idx, datcol]
    h = df_new.loc[idx, huscol]
    x = df_new.loc[idx, xcol]
    y = df_new.loc[idx, ycol]
    df_new.loc[idx, 'x'] = x
    df_new.loc[idx, 'y'] = y
    df_new.loc[idx, 'CRS_og'] = CRS[d][h]
    
for idx in df_new[~filtroUTM & filtrogeo].index:
    lat = df_new.loc[idx, latcol]
    lon = df_new.loc[idx, loncol]
    lat = latlon2dd(lat)
    lon = latlon2dd(lon)
    dat = str(df_new.loc[idx, datgcol])
    df_new.loc[idx, 'x'] = lon
    df_new.loc[idx, 'y'] = lat
    df_new.loc[idx, 'CRS_og'] = CRSg[dat]
    
projectionsR = [x for x in df_new['CRS_og'].unique() if x not in ['none', 'EPSG:32719']]
transformers = {}
for proj in projectionsR:
    transformers[proj] = Transformer.from_crs(proj, 'EPSG:32719', always_xy=True)


to_transform = ~df_new['CRS_og'].isin(['EPSG:32719'])
df_new['x_32719'] = df_new[xcol].copy()
df_new['y_32719'] = df_new[ycol].copy()
    
for idx in df_new[to_transform].index:
    CRSog = df_new.loc[idx, 'CRS_og']
    if CRSog not in ['EPSG:32719', 'NONE']:
        x = df_new.loc[idx, 'x']
        y = df_new.loc[idx, 'y']
        newx, newy = transformers[CRSog].transform(x,y)
        df_new.loc[idx,'x_32719'] = newx
        df_new.loc[idx,'y_32719'] = newy
    elif CRSog == 'EPSG:32719':
        x = df_new.loc[idx, 'x']
        y = df_new.loc[idx, 'y']
        df_new.loc[idx,'x_32719'] = x
        df_new.loc[idx,'y_32719'] = y
    else:
        pass

#%% write geodataframe

def drop_infinity(df, fields):
    # df[fields].replace([np.inf,-np.inf], np.nan, inplace=True)
    df[fields] = df[fields].replace([np.inf, -np.inf], np.nan)
    newdf = df.dropna(subset=fields, how="all")
    return newdf

df_export = drop_infinity(df_new, ['x_32719', 'y_32719'])

gdf = gpd.GeoDataFrame(df_export,
                       crs = 'EPSG:32719',
                       geometry = gpd.points_from_xy(df_export['x_32719'],
                                                     df_export['y_32719']))
gdf.to_file('DAA_nuble.geojson', driver = 'GeoJSON')

