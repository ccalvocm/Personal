#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:27:04 2020

@author: felipe
"""

import modules_FAA
import geopandas
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.cbook as cbook
import dataframe_image as dfi
import cirenutils




path_TO = '../Etapa 1 y 2/GIS/SISS/TerritorioOperacional/TERRITORIO OPERACIONAL.shp'
path_basin = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
path_plantas = '../Etapa 1 y 2/GIS/SISS/Plantas/PLANTAS.shp'
path_APR = '../Etapa 1 y 2/GIS/DOH/Agua_Potable_Rural/APR_julio_2017.shp'


gdf_TO = geopandas.read_file(path_TO)
gdf_cuencas = geopandas.read_file(path_basin)
gdf_plantas = geopandas.read_file(path_plantas)
gdf_APR = geopandas.read_file(path_APR)

filtros = (gdf_cuencas['COD_CUENCA'] == '1300') | \
    (gdf_cuencas['COD_CUENCA'] == '0600') | \
        (gdf_cuencas['COD_CUENCA'] == '0703') | \
            (gdf_cuencas['COD_CUENCA'] == '0701')

gdf_cuencas = gdf_cuencas[filtros]
gdf_TO = gdf_TO.to_crs('EPSG:32719')
gdf_APR = gdf_APR.to_crs('EPSG:32719')


basin_TO = geopandas.sjoin(gdf_TO, gdf_cuencas, how = 'inner')
basin_APR = geopandas.sjoin(gdf_APR, gdf_cuencas, how = 'inner')

#%% Plot TO, APR y cuencas

fig = plt.figure(figsize = (19.2,9.83), tight_layout = True)

# ax = fig.add_subplot(111)
ax1, ax2, ax4, ax5 = cirenutils.vinetaCIREN(fig)

minx = gdf_cuencas.geometry.bounds['minx'].min() - 10000
maxx = gdf_cuencas.geometry.bounds['maxx'].max() + 10000
miny = gdf_cuencas.geometry.bounds['miny'].min() - 50000
maxy = gdf_cuencas.geometry.bounds['maxy'].max() + 10000

ax1.set_xlim(minx, maxx)
ax1.set_ylim(miny, maxy)

# ax3.set_xlim(minx, maxx)
# ax3.set_ylim(miny, maxy)

gdf_cuencas.plot(ax = ax1, facecolor = 'none', edgecolor = 'black', linewidth = 1, linestyle = '-.')
basin_TO.plot(ax = ax1, facecolor = 'blue', edgecolor = 'blue')
# basin_APR.plot(ax = ax, color = 'red', markersize = 2)
ctx.add_basemap(ax = ax1, crs='epsg:32719', source = ctx.providers.Esri.WorldTerrain)
scalebar = ScaleBar(1.0, location = 'upper left') # 1 pixel = 0.2 meter
ax1.add_artist(scalebar)
x, y, arrow_length = 0.95, 0.99, 0.07
ax1.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax1.transAxes)
LegendElement = [mpatches.Patch(facecolor='none', edgecolor='k', linestyle = '-.', label='Cuencas en estudio'),
                 mpatches.Patch(facecolor='blue', label='Territorio Operacional Empresas Sanitarias')]
                 #Line2D([],[], color='green', lw = 4, label='Cauces Principales'),
                 #mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor = 'red', edgecolor = 'none', label = 'Sistemas APR')]
ax2.legend(handles = LegendElement,
           #handler_map={mpatches.Circle: HandlerEllipse()},
            loc = 'center left', frameon = False)#.get_frame().set_facecolor('#00FFCC')


plt.draw()
plt.show()
plt.savefig('../Etapa 1 y 2/Figuras/mapa_demanda_AP.jpg', dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

# basin_TO['superficiekm2'] = basin_TO['geometry'].area/ 10**6
# table = basin_TO.groupby(['NOM_CUENCA']).sum()
# dfi.export(table, 'test.png')

# exportar los geodataframe
# gdf_cuencas.to_file('dda_AP_cuencas.shp')
# basin_TO.to_file('dda_AP_TO_sanitarias.shp')


#%% Plot APR y cuencas

fig = plt.figure(figsize = (11,8.5))

ax = fig.add_subplot(111)

minx = gdf_cuencas.geometry.bounds['minx'].min() - 10000
maxx = gdf_cuencas.geometry.bounds['maxx'].max() + 10000
miny = gdf_cuencas.geometry.bounds['miny'].min() - 50000
maxy = gdf_cuencas.geometry.bounds['maxy'].max() + 10000

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

gdf_cuencas.plot(ax = ax, facecolor = 'none', edgecolor = 'black', linewidth = 1, linestyle = '-.')
# basin_TO.plot(ax = ax, facecolor = 'blue', edgecolor = 'blue')
basin_APR.plot(ax = ax, color = 'red', markersize = 2)
ctx.add_basemap(ax = ax, crs='epsg:32719', source = ctx.providers.Esri.WorldTerrain)
x, y, arrow_length = 0.95, 0.99, 0.07
ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax.transAxes)
LegendElement = [mpatches.Patch(facecolor='none', edgecolor='k', linestyle = '-.', label='Cuencas en estudio'),
                 # mpatches.Patch(facecolor='blue', label='Territorio Operacional Empresas Sanitarias')]
                 #Line2D([],[], color='green', lw = 4, label='Cauces Principales'),
                 mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor = 'red', edgecolor = 'none', label = 'Sistemas APR')]
plt.legend(handles = LegendElement,
           #handler_map={mpatches.Circle: HandlerEllipse()},
           loc = 'lower right')#.get_frame().set_facecolor('#00FFCC')
# scalebar = ScaleBar(1.0, location = 'upper left') # 1 pixel = 0.2 meter
# plt.gca().add_artist(scalebar)
plt.show()
plt.savefig('../Etapa 1 y 2/Figuras/mapa_APR.jpg', dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

tableAPR = basin_APR.groupby(['NOM_CUENCA']).sum()

# export los geodataframe a shapefile
basin_APR.to_file('dda_AP_APR.shp')