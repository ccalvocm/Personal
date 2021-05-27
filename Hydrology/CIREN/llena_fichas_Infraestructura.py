# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:20:18 2021

@author: farrospide
"""
import geopandas
import os

capa = 'AR'

capas = {'AR': 'AR_Maule_2020.shp',
         'bocatomas': 'bocatomas_maule_2019.shp',
         'canales': 'canales_maule_2020.shp',
         'embalses': 'embalses_maule_2019.shp',
         'singularidades': 'singularidades_maule_2019.shp'}

path = os.path.join('..',
                    'SIG',
                    'COBERTURAS INFRAESTRUCTURA Y ÁREAS DE RIEGO',
                    'Infraestructura_Riego_Maule_2020',
                    capas[capa])

gdf = geopandas.read_file(path)

fields = gdf.columns

# nom_canales = gdf['NOMCAN'].unique()
# nom_fuen_hidrica = gdf['NOMFUENHID'].unique() 

print(fields)
gdf.plot()