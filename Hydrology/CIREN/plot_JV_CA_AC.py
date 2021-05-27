#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:12:49 2021

@author: faarrosp
"""

import os
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from plot_tools import *

# -----------------
# FUNCIONES
# -----------------

def import_OUA_shape(folder, kind):
    files = [x for x in os.listdir(folder) \
             if ((x.endswith('.shp')) and (x.endswith(kind + '.shp')))]
        
    gdf_list = []
    
    for file in files:
        fp = os.path.join(folder, file)
        gdf = gpd.read_file(fp)
        gdf_list.append(gdf)
        
    gdf_final = pd.concat(gdf_list, ignore_index = True)
    return gdf_final

def plot_OUA_shape(gdf, column, catchment = None, title = None):
    
    filtrona = gdf[column].isna() # filtro q detecta los campos NULL
    
    
    fig,ax = plt.subplots(figsize = (8.5,11))
    # plotear lo NA
    gdf[filtrona].plot(ax=ax, color = 'blue', lw = 0.5,
                       label = 'Sin registro DGA')
    # plotear lo no NA
    gdf[~filtrona].plot(ax=ax, color = 'green', lw = 1.5,
                        label = 'Aprobada en DGA')
    
    # plotear cuenca (contorno)
    if catchment is not None:
        catchment.plot(ax=ax, fc='none', ec = 'red')
    else:
        pass
    if title is not None:
        ax.set_title(title)
    else:
        pass
    
    add_basemap_ax(ax)
    ax.legend(title = 'Leyenda')


#%% Ejecutar

# folder para llegar a los shapes
folder = os.path.join('..', 'SIG',
                      'COBERTURAS INFRAESTRUCTURA Y ÁREAS DE RIEGO',
                      'IR_etapa2',
                      'OUA')


# importamos shapes de CA
gdf_CA = import_OUA_shape(folder, 'CA')

# importamos shapes de AC
gdf_AC = import_OUA_shape(folder, 'AC')

# ahora importamos las macrocuencas
cuencas, subcuencas = import_catchments()

# plotear CA x macrocuenca
cod, name = ['1300', '0600', '0701', '0703'] , ['Rio Maipo',
                                                'Rio Rapel',
                                                'Rio Mataquito',
                                                'Rio Maule']

for codigo, nombre in zip(cod,name):
    cca = cuencas[cuencas['COD_CUENCA'].isin([codigo])]
    nombre_ext = nombre.replace(' ', '')
    
    ca_sub = gpd.sjoin(gdf_CA,cca, how='inner')
    ac_sub = gpd.sjoin(gdf_AC,cca, how='inner')
    
    title = 'Situación Comunidades de Aguas\nCuenca ' + nombre
    plot_OUA_shape(ca_sub, 'SIT_LEG', catchment = cca, title = title)    
    fp = os.path.join(folder, 'CA_' + nombre_ext + '.jpg')
    plt.savefig(fp, format='jpg', bbox_inches = 'tight',
                pad_inches = 0.1)
    plt.close()
    
    
    title = 'Situación Asociaciones de Canalistas\nCuenca ' + nombre
    plot_OUA_shape(ac_sub, 'SIT_LEG', catchment = cca, title = title)
    fp = os.path.join(folder, 'AC_' + nombre_ext + '.jpg')
    plt.savefig(fp, format='jpg', bbox_inches = 'tight',
                pad_inches = 0.1)
    plt.close()
    
    
    




    

        