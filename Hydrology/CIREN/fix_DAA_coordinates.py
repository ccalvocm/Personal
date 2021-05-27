#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:26:26 2021

@author: faarrosp
"""

import os
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
import contextily as ctx
from matplotlib import pyplot as plt
import numpy as np
import ee
import folium
import geemap
from IPython.display import clear_output
from IPython import get_ipython

#%%

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

# src = ctx.providers.Esri.WorldTerrain
src = ctx.providers.Esri.WorldImagery

'''dict_keys([
'WorldStreetMap', 'DeLorme', 'WorldTopoMap', 'WorldImagery', 'WorldTerrain',
'WorldShadedRelief', 'WorldPhysical', 'OceanBasemap', 'NatGeoWorldMap',
'WorldGrayCanvas'])'''

CRS = {'1984': {'18': 'EPSG:32718', '19': 'EPSG:32719'},
       '1969': {'18': 'EPSG:29188', '19': 'EPSG:29189'},
       '1956': {'18': 'EPSG:24878', '19': 'EPSG:24879'},
       '84': {'18': 'EPSG:32718', '19': 'EPSG:32719'}}

CRSg = {'1984': 'EPSG:4326',
        '1956': 'EPSG:4248',
        '1969': 'EPSG:4724'}

not_georref = os.path.join('..',
                           'Etapa 1 y 2',
                           'DAA',
                           'DAA_CFA',
                           'Tablas_DAA',
                           'DAA_CFA_no_georreferenciables.xlsx')

df = pd.read_excel(not_georref)

cols = [x for x in df.columns]
#%%

def prompt_main_menu(msg):
    choose_action = ''
    get_ipython().magic('clear')
    while choose_action not in ['X', 'x', 'P', 'p', 'N', 'n']:
        get_ipython().magic('clear')
        choose_action = input('\n'.join([msg,
                                       '| Elija una acción:',
                                       '| [X] salir del programa',
                                       '| [P] plotear registro',
                                       '| [N] próximo registro',
                                       '>']))        
    return choose_action
#%%
c1x = cols[62]
c1y = cols[63]
h1 = cols[64]
d1 = cols[65]

c2x = cols[66]
c2y = cols[67]

c3x = cols[69] # longitud
c3y = cols[68] # latitud

d2 = cols[70]

ref = cols[81]

c4x = cols[98]
c4y = cols[99]

for index in df.index[:4]:
    x1 = str(df.loc[index,c1x])
    y1 = str(df.loc[index,c1y])
    x2 = str(df.loc[index,c2x])
    y2 = str(df.loc[index,c2y])
    x3 = str(df.loc[index,c3x])
    y3 = str(df.loc[index,c3y])
    x4 = str(df.loc[index,c4x])
    y4 = str(df.loc[index,c4y])
    msg = '\n'.join(['------------------------',
                     'Registro #' + str(index),
                     '------------------------',
                     'x1: ' + x1,
                     'y1: ' + y1,
                     ' ',
                     'x2: ' + x2,
                     'y2: ' + y2,
                     ' ',
                     'lon: ' + x3,
                     'lat: ' + y3,
                     ' ',
                     'x4: ' + x4,
                     'y4: ' + y4])
    # print(msg)
    choose_mmaction = prompt_main_menu(msg)
    
    if choose_mmaction in ['P', 'p']:
        choose_pair = input('\n'.join(['Seleccione par coordenado para graficar',
                                   '[1]: ' + x1 + ' ' + y1,
                                   '[2]: ' + x2 + ' ' + y2,
                                   '[3]: ' + x3 + ' ' + y3,
                                   '[4]: ' + x4 + ' ' + y4,
                                   '[X]: salir del programa',
                                   '[N]: próximo registro',
                                   '>']))
        if choose_pair in ['1', '2', '3', '4']:
            choose_dat = ''
            choose_hus = ''
            if choose_pair != '3':
                plotx = float(x1)
                ploty = float(y1)
                
                while choose_dat not in ['1956', '1969', '1984']:
                    choose_dat = input('\n'.join(['Seleccione Datum:',
                                                  '[1956] / [1969] /[1984]',
                                                  '>']))
                
                
                while choose_hus not in ['18', '19']:
                    choose_hus = input('\n'.join(['Seleccione Huso:',
                                                  '[18] / [19]',
                                                  '>']))
                plot_CRS = CRS[choose_dat][choose_hus]
            else:
                
                while choose_dat not in ['1956', '1969', '1984']:
                    choose_dat = input('\n'.join(['Seleccione Datum:',
                                                  '[1956] / [1969] /[1984]',
                                                  '>']))
                plot_CRS = CRSg[choose_dat]
                plotx = latlon2dd(x3)
                ploty = latlon2dd(y3)
    elif choose_mmaction in ['N', 'n']:
        # os.system('clear')
        get_ipython().magic('clear')
    
    elif choose_mmaction in ['X', 'x']:
        break
    else:
        pass
            
            
    
    