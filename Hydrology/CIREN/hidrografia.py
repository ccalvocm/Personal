#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:36:21 2020

@author: felipe
"""

import geopandas

path = '../Etapa 1 y 2/GIS/Hidrografia/Red_Hidrografica.shp'
gdf_hidro = geopandas.read_file(path)
gdf_hidro = gdf_hidro.to_crs('EPSG:32719')

path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
gdf_cuencas = geopandas.read_file(path)
gdf_cuencas = gdf_cuencas[gdf_cuencas['COD_CUENCA'] == '0600']

gdf_hidro_wbasin = geopandas.sjoin(gdf_hidro, gdf_cuencas, how = 'inner')

gdf_hidro_wbasin.to_file('../Etapa 1 y 2/GIS/Hidrografia/Red_Hidrografica_0600_Rapel.shp')