#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:29:35 2021

@author: faarrosp
"""

import contextily as ctx
import os
import geopandas as gpd

def add_basemap_ax(ax, crs = 'EPSG:32719',
                   source = ctx.providers.Esri.WorldTerrain):
    ctx.add_basemap(ax = ax, zoom = 9, crs = crs, source = source)
    ax.set_xlabel('Coordenada UTM Este (m)')
    ax.set_ylabel('Coordenada UTM Norte (m)')
    
def import_catchments():
    folder = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
    f1 = os.path.join(folder, 'Cuencas', 'Cuencas_DARH_2015_AOHIA_ZC.geojson')
    f2 = os.path.join(folder, 'Subcuencas',
                      'SubCuencas_DARH_2015_AOHIA_ZC.geojson')
    cuencas = gpd.read_file(f1)
    scuencas = gpd.read_file(f2)
    
    return cuencas, scuencas