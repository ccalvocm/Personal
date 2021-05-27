#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:02:28 2021

@author: faarrosp
"""

import os
import geopandas as gpd
import numpy as np
import random
import contextily as ctx
from matplotlib import pyplot as plt


folder_cuencas = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
path_cuencas = os.path.join(folder_cuencas, 'Cuencas',
                            'Cuencas_DARH_2015_AOHIA_ZC.geojson')
folder_canales = os.path.join('..', 'SIG',
                              'COBERTURAS INFRAESTRUCTURA Y ÁREAS DE RIEGO',
                              'Infraestructura_Riego_Maule_2020',
                              'canales_maule_2020.shp')


gdf_cuencas = gpd.read_file(path_cuencas)
gdf_canales = gpd.read_file(folder_canales)

#%%
mataquito = gdf_cuencas[gdf_cuencas['COD_CUENCA'] == '0701']
canales_mtq = gpd.sjoin(gdf_canales, mataquito, how = 'inner')

#canales_mtq.plot()

# diccionario de coeficientes Moritz

coef = {'Grava': 0.1,
        'Arcilloso': 0.13,
        'Franco': 0.2,
        'Ceniza': 0.21,
        'Arena-ceniza-arcilla': 0.37,
        'Arena-roca': 0.51,
        'Arenoso-grava': 0.67,
        'Estandar': 0.37}

canales_mtq['COEF_C'] = ' '

for i in canales_mtq.index:
    canales_mtq.loc[i,'COEF_C'] = random.choice(['Grava', 'Arcilloso', 'Franco',
                                       'Ceniza', 'Estandar', 'Arena-roca',
                                       'Arena-ceniza-arcilla', 'Arenoso-grava'])
canales_mtq['Q'] = 1.0
canales_mtq['V'] = 1.0

def Moritz(row):
    coef_P = 0.0375 * coef[row['COEF_C']] * np.sqrt(row['Q']/row['V'])
    return coef_P

canales_mtq['Moritz_P'] = canales_mtq.apply(lambda row: Moritz(row), axis =1)

fig, ax = plt.subplots()
canales_mtq.plot(ax = ax, column = 'Moritz_P', legend = True, scheme = 'quantiles')


