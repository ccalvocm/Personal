# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:17:23 2021

@author: Carlos
"""

# librerias
# Importar librerias
from IPython.core.display import display, HTML
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import modules_FAA
import modules_CCC
import geopandas
import contextily as ctx
import matplotlib.image as mpimg
import matplotlib
import locale
from matplotlib.patches import Patch
from itertools import cycle

# funciones
# Función rut
def digito_verificador(rut):
    reversed_digits = map(int, reversed(str(rut)))
    factors = cycle(range(2, 8))
    s = sum(d * f for d, f in zip(reversed_digits, factors))
    if (-s) % 11 > 9:
        return 'K'
    else:
        return (-s) % 11

def main():
    
    # Set to Spanish locale to get comma decimal separater
    locale.setlocale(locale.LC_NUMERIC, "es_ES")
    locale.setlocale(locale.LC_TIME, "es_ES") # swedish
    plt.rcdefaults()
    
    # Tell matplotlib to use the locale we set above
    plt.rcParams['axes.formatter.use_locale'] = True
    
    # Paths y carga de datos genericos
    path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
    basin = geopandas.read_file(path)
    path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Subcuencas/SubCuencas_DARH_2015.shp'
    subbasin = geopandas.read_file(path)
    ruta_Git = r'C:\Users\ccalvo\Documents\GitHub'
    ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos'
    
    # subcuencas de cabecera
    ruta_cabecera = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Cuencas_CAMELS\Cuencas_cabecera_MaipoRapelMataquitoMaule.shp'
    
        
    # grafico de centrales
    plt.close("all")
    fig = plt.figure(figsize=(10, 8))
    axes = fig.add_subplot(111)
    scc = geopandas.read_file(ruta_cabecera)
    scc.set_crs(epsg = 4326, inplace=True)
    scc.to_crs(epsg = 32719, inplace=True)
    
    cuencas = geopandas.read_file('..//Etapa 1 y 2//GIS//Cuencas_DARH//Cuencas//Cuencas_DARH_2015.shp')
    cuencas = cuencas.loc[(cuencas['COD_CUENCA'] == '0703') | (cuencas['COD_CUENCA'] == '0701') | (cuencas['COD_CUENCA'] == '0600') | (cuencas['COD_CUENCA'] == '1300')]
    cuencas.plot(ax = axes, alpha=0.4)
    scc.plot(ax = axes, color = 'c', edgecolor = 'k', linewidth = 2)
    
    # sacar las coordenadas
    scc['coords'] = scc['geometry'].apply(lambda x: x.representative_point().coords[:])
    scc['coords'] = [coords[0] for coords in scc['coords']]
    
    # reordenar por cuenca
    scc.sort_values('gauge_id', inplace = True)
    scc.reset_index(drop = True, inplace = True)
    
    for idx, row in scc.iterrows():
        plt.annotate(s=idx+1, xy=row['coords'],
                     horizontalalignment='right', color='white', size=10)
    
    legend = [     Patch(facecolor='c', edgecolor=None, label='Subcuencas de cabecera'),
                   Patch(facecolor='steelblue', edgecolor=None, label='Cuencas en estudio')]
    axes.legend(legend, ['Subcuencas de cabecera','Cuencas en estudio'], loc = 'upper left')
        
    ctx.add_basemap(ax = axes, crs= scc.crs.to_string(),
                            source = ctx.providers.Esri.WorldTerrain, zoom = 8)
    
    # axes.legend(['Centros acuícolas en tierra'], loc = 'upper left')
    x, y, arrow_length = 0.95, 0.95, 0.07
    axes.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=axes.transAxes)
    x, y, scale_len = cuencas.bounds['minx'], cuencas.bounds['miny'].min(), 20000 #arrowstyle='-'
    scale_rect = matplotlib.patches.Rectangle((x.iloc[0],y),scale_len,200,linewidth=1,
                                            edgecolor='k',facecolor='k')
    axes.add_patch(scale_rect)
    plt.text(x.iloc[0]+scale_len/2, y+5000, s='20 KM', fontsize=10,
                 horizontalalignment='center')
    axes.set_xlabel('Coordenada Este WGS84 19S (m)')
    axes.set_ylabel('Coordenada Norte WGS84 19S (m)')

    # calcular digito verificador
    df = pd.DataFrame(scc.drop(columns='geometry'))
    for ind, row in df.iterrows():
        df.loc[ind,'gauge_id'] = str(df['gauge_id'].values[ind])+'-'+str(digito_verificador(df['gauge_id'].values[ind]))
    

    
    