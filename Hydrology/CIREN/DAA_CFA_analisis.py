#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:20:52 2021

@author: faarrosp
"""

import os
import pandas as pd
import geopandas as gpd
import fuzzywuzzy
import contextily as ctx
from matplotlib import pyplot as plt

path_DAA = os.path.join('..', 'Etapa 1 y 2', 'DAA', 'shapes_output',
                    'captaciones_post_ALL_CFA.geojson')

path_macrocuencas = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH',
                                 'Cuencas', 'Cuencas_DARH_2015_AOHIA_ZC.geojson')

path_hidrografia = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Hidrografia', 'RedHidrograficaUTM19S.shp')

gdf_DAA = gpd.read_file(path_DAA)
gdf_macrocuencas = gpd.read_file(path_macrocuencas)

# 1300 MAIPO 0701 MATAQUITO  0703 MAULE 0600 RAPEL

#gdf_macrocuencas = gdf_macrocuencas[gdf_macrocuencas['COD_CUENCA'].isin(['1300', '0701', '0600', '0703'])]
#gdf_macrocuencas = gdf_macrocuencas[gdf_macrocuencas['COD_CUENCA'].isin(['1300'])]
gdf_DAA = gpd.sjoin(gdf_DAA, gdf_macrocuencas, how = 'inner')
gdf_hidrografia = gpd.read_file(path_hidrografia) #epsg 32719
gdf_hidrografia = gpd.sjoin(gdf_hidrografia, gdf_macrocuencas, how = 'inner')

#gdf_hidrografia.plot()
path_hidrografia = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Hidrografia', 'RedHidrograficaUTM19S_AOHIA_ZC.geojson')
gdf_hidrografia.to_file(path_hidrografia, driver = 'GeoJSON')

#%%

cols = gdf_DAA.columns

solicitantes = gdf_DAA[['Rut_Solicitante', 'Nombre_Solicitante']].drop_duplicates().sort_values(by = 'Nombre_Solicitante')

fuentes = gdf_DAA[['Fuente']].drop_duplicates().sort_values(by = 'Fuente')
usos = gdf_DAA[['Uso_del_Agua']].drop_duplicates().sort_values(by = 'Uso_del_Agua')

solicitantes_buscados =  ['ANGLO AMERICAN SUR S.A.']

sanitarias = ['AGUAS ANDINAS S.A', 'AGUAS CORDILLERA S.A.',
              'AGUAS MANQUEHUE S.A.', 'AGUAS SUBSTRATUM S.A.',
              'AGUAS DE CHILE CONSULTORES LTDA.', 'EMOS   Y OTROS.',
              'EMOS S.A.', 'EMP. DE AGUA POTABLE MANQUEHUE LTDA.   Y OTROS.',
              'EMPRESA AGUA POTABLE MANQUEHUE S.A.', 
              'EMPRESA DE AGUA POTABLE  SMP S.A.',
              'EMPRESA DE AGUA POTABLE LO AGUIRRE S.A.',
              'EMPRESA DE AGUA POTABLE LO CASTILLO LTDA',
              'EMPRESA DE AGUA POTABLE LO CASTILLO S.A.   Y OTROS.',
              'EMPRESA DE AGUA POTABLE MANQUEHUE LTDA.',
              'EMPRESA DE AGUA POTABLE MANQUEHUE LTDA.   Y OTROS.',
              'EMPRESA DE AGUA POTABLE MANQUEHUE S. A',
              'EMPRESA DE AGUA POTABLE MANQUEHUE S. A.',
              'EMPRESA DE AGUA POTABLE MANQUEHUE S.A',
              'EMPRESA DE AGUA POTABLE MANQUEHUE S.A.',
              'EMPRESA DE AGUA POTABLE MANQUEHUE S.A.   Y OTROS.',
              'EMPRESA DE AGUA POTABLE MANQUEHUE.A.',
              'EMPRESA DE AGUA POTABLE MELIPILLA NORTE S.A.',
              'EMPRESA DE AGUA POTABLE PIRQUE S.A.',
              'EMPRESA DE AGUA POTABLE QUEBRADA DE MACUL LIMITADA',
              'EMPRESA DE AGUA POTABLE SMP S.A.',
              'EMPRESA DE AGUA POTABLE VALLE NEVADO S.A.',
              'EMPRESA DE AGUA POTABLE VILLA LOS DOMINICOS S.A',
              'EMPRESA DE OBRAS SANITARIAS DE VALPARAISO S.A.',
              'EMPRESA DE SERVICIOS SANITARIOS AGUAS DE COLINA S.A.',
              'EMPRESA DE SERVICIOS SANITARIOS DEL LIBERTADOR S.A.',
              'EMPRESA DE SERVICIOS SANITARIOS LO PRADO S.A.',
              'EMPRESA DE SERVICIOS SANITARIOS LO PRADO S.A.',
              'EMPRESA METROPOLITANA DE OBRAS SANITARIA',
              'EMPRESA METROPOLITANA DE OBRAS SANITARIAS',
              'EMPRESA METROPOLITANA DE OBRAS SANITARIAS (EMOS)',
              'EMPRESA METROPOLITANA DE OBRAS SANITARIAS (EMOS)   Y OTROS.',
              'EMPRESA METROPOLITANA DE OBRAS SANITARIAS S.A.',
              ]

# Search for Agua Potable strings or Sanitarias strings
solicitantes['Nombre_Solicitante'] = solicitantes['Nombre_Solicitante'].str.lower()

aguas = solicitantes['Nombre_Solicitante'].str.find('agua')
sanit = solicitantes['Nombre_Solicitante'].str.find('sanit')
sendos = solicitantes['Nombre_Solicitante'].str.find('sendos')
solicitantes_aguas = solicitantes[aguas > -1]['Nombre_Solicitante']
solicitantes_sanit = solicitantes[sanit > -1]['Nombre_Solicitante']
solicitantes_sendos = solicitantes[sendos > -1]['Nombre_Solicitante']
solicitantes_AP = solicitantes[(aguas > -1) | (sanit > -1) | (sendos > -1)]['Nombre_Solicitante']

savepath = os.path.join('..', 'Etapa 1 y 2', 'DAA', 'shapes_output',
                    'captaciones_post_ALL_CFA_AP.geojson')
A = gdf_DAA.loc[solicitantes_AP.index,:]
A.to_file(savepath, driver = 'GeoJSON')




fuentes_buscadas = ['RIO MAPOCHO',
                    'Rio Mapocho Primera Seccion',
                    'Rio Mapocho-1 Seccion',
                    'Rio Molina',
                    'Rio San Francisco',
                    'Rio San Francisco-Angostura']

#result_fuentes = gdf_DAA[gdf_DAA['Fuente'].isin(fuentes_buscadas)]
#result_sanitarias = gdf_DAA[gdf_DAA['Nombre_Solicitante'].isin(sanitarias)]

#%%
src = ctx.providers.Esri.WorldTerrain

fig = plt.figure()
ax = fig.add_subplot(111)
result_fuentes.plot(ax = ax, color = 'red')
gdf_hidrografia.plot(ax = ax)
ctx.add_basemap(ax = ax, source = src, crs = 'EPSG:32719')

