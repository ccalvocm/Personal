# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:11:43 2021

@author: farrospide
"""
import geopandas as gpd
import pandas as pd
import os
import contextily as ctx
from matplotlib import pyplot as plt

path_canales = os.path.join('..', 'SIG', 'Canales_Revestidos',
                    'Puntos_CNR_Inversion.shp' )

gdf_canales = gpd.read_file(path_canales)

fig = plt.figure()
ax = fig.add_subplot(111)
src = ctx.providers.CartoDB.Voyager
gdf_canales.plot(ax =ax)
ctx.add_basemap(ax, source = src, crs = 'EPSG:32719')