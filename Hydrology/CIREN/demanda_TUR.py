# -*- coding: utf-8 -*-
"""
Created on Mon May 10 06:42:22 2021

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
from matplotlib.lines import Line2D
import matplotlib
from matplotlib.patches import Patch
import locale
locale.setlocale(locale.LC_NUMERIC, "es_ES")
# Set to Spanish locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "es_ES")
locale.setlocale(locale.LC_TIME, "es_ES") # swedish

plt.rcdefaults()

# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True


def main():
    
    cuenca = 'Maipo'
    # rutas
    ruta_OD = r'C:\Users\ccalvo\OneDrive - ciren.cl'
    ruta_OD = r'E:\CIREN\OneDrive - ciren.cl'
    ruta_tur = ruta_OD+r'\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\TUR'
    # Paths y carga de datos genericos
    path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
    basin = geopandas.read_file(path)
    path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Subcuencas/SubCuencas_DARH_2015.shp'
    subbasin = geopandas.read_file(path)
    
    # Maipo
    
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    ZOIT_RM1 = geopandas.read_file(r'..\SIG\02_CUENCA_Maipo\01_CAPAS_VECTORES\12_MedioAmbiente\ZOIT_Maipo.shp')
    ZOIT_RM1.plot(ax = axes, color = 'green',zorder = 3)
    ZOIT_RM2 = geopandas.read_file(r'..\SIG\02_CUENCA_Maipo\01_CAPAS_VECTORES\12_MedioAmbiente\ZOIT_La_Chimba_Maipo.shp')
    ZOIT_RM2.plot(ax = axes, color = 'blue',zorder = 2)
    
    modules_FAA.plot_catchment_map(basin, bsn_N = '1300', ax = axes, basemap = True, linewidth = 5, edgecolor = 'black', label = 'Limite cuenca')
    plt.show()
    subbasin.style.set_properties(**{'text-align': 'left'})
    
    # leyenda
    legend = [
                Patch(facecolor='green', edgecolor=None, label='ZOIT del Río Maipo'),
                Patch(facecolor='blue', edgecolor=None, label='Zoit La Chimba'),
                Patch(facecolor='gray', edgecolor=None, label='Cuenca del Río '+cuenca)]
    axes.legend(legend, ['ZOIT San José de Maipo','Zoit La Chimba','Cuenca del Río '+cuenca], loc = 'upper left')
    
    ruta_actual = ruta_tur+r'\XIII\TURISTICO XIII REGIÓN.xlsx'
    dda_RM = pd.read_excel(ruta_actual, sheet_name = 'SINTESIS', index_col = 0) 
    dda_RM = dda_RM[['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
           'Nov', 'Dic','Ene', 'Feb', 'Mar']]
    
    # presente
    plt.figure()
    df_tur = dda_RM.iloc[0:2,:-1].transpose()
    df_tur.plot.bar()
    plt.ylabel('caudal $m^3/s$')
    # plt.figure()
    # dda_RM.iloc[-1,:-1].transpose().plot.bar(legend = True)
    plt.ylabel('caudal $m^3/s$')
    plt.title('Caudal de reserva de uso turístico cuenca del Río '+cuenca)
    plt.grid()

    # futuro
    ruta_fut = ruta_tur+r'\XIII\TURISTICO_FUT XIII REGIÓN.xlsx'
    dda_RM_fut = pd.read_excel(ruta_fut, sheet_name = 'SINTESIS', index_col = 0) 
    dda_RM_fut = dda_RM_fut[['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
           'Nov', 'Dic','Ene', 'Feb', 'Mar']]
    plt.figure()
    df_tur = dda_RM_fut.iloc[0:3,:-1].transpose()
    df_tur.plot.bar(legend = True)
    plt.ylabel('caudal $m^3/s$')
    plt.ylabel('caudal $m^3/s$')
    plt.title('Caudal de reserva de uso turístico proyectado cuenca del Río '+cuenca)
    plt.grid()


    # Rapel
    cuenca = 'Rapel'
    
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    ZOIT_Rapel = geopandas.read_file(r'..\SIG\03_CUENCA_Rapel\01_CAPAS_VECTORES\12_MedioAmbiente\ZOIT_Rapel.shp')
    ZOIT_Rapel.plot(ax = axes, color = 'green', zorder = 2)
    
    modules_FAA.plot_catchment_map(basin, bsn_N = '0600', ax = axes, basemap = True, linewidth = 5, edgecolor = 'black', label = 'Limite cuenca')
    plt.show()
    fil = subbasin['COD_CUENCA'] == '0600'
    subbasin.style.set_properties(**{'text-align': 'left'})
  # leyenda
    legend = [
                Patch(facecolor='green', edgecolor=None, label='ZOIT del Río Maipo'),
                Patch(facecolor='gray', edgecolor=None, label='Cuenca del Río '+cuenca)]
    axes.legend(legend, ['ZOIT Lago Rapel','Cuenca del Río '+cuenca], loc = 'lower right')
    axes.set_xlabel('Coordenada Este WGS84 19S (m)')
    axes.set_ylabel('Coordenada Norte WGS84 19S (m)')
    
    #presente
    ruta_actual = ruta_tur+r'\VI\TURISTICO VI REGIÓN.xlsx'
    dda_RR = pd.read_excel(ruta_actual, sheet_name = 'SINTESIS', index_col = 0) 
    dda_RR = dda_RR[['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
           'Nov', 'Dic','Ene', 'Feb', 'Mar']]
    plt.figure()
    df_tur = dda_RR.iloc[0:3,:-1].transpose()
    df_tur.plot.bar()
    plt.ylabel('caudal $m^3/s$')
    plt.title('Caudal de reserva de uso turístico cuenca del Río '+cuenca)
    plt.grid()
    
    # Maule
    cuenca = 'Maule'
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    ZOIT_Maule = geopandas.read_file(r'..\SIG\05_CUENCA_Maule\01_CAPAS_VECTORES\12_MedioAmbiente\ZOIT_Maule.shp')
    ZOIT_Maule.plot(ax = axes, color = 'green', zorder = 2)
    
    modules_FAA.plot_catchment_map(basin, bsn_N = '0703', ax = axes, basemap = True, linewidth = 5, edgecolor = 'black', label = 'Limite cuenca')
    plt.show()
    subbasin.style.set_properties(**{'text-align': 'left'})
    # leyenda
    legend = [
                Patch(facecolor='green', edgecolor=None, label='ZOIT del Río Maipo'),
                Patch(facecolor='gray', edgecolor=None, label='Cuenca del Río '+cuenca)]
    axes.legend(legend, ['ZOIT Colbún-Rari','Cuenca del Río '+cuenca], loc = 'lower right')
    axes.set_xlabel('Coordenada Este WGS84 19S (m)')
    axes.set_ylabel('Coordenada Norte WGS84 19S (m)')
    
    # presente
    ruta_actual = ruta_tur+r'\VII\TURISTICO VII REGIÓN.xlsx'
    dda_RMaule = pd.read_excel(ruta_actual, sheet_name = 'SINTESIS', index_col = 0) 
    dda_RMaule = dda_RMaule[['Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct',
           'Nov', 'Dic','Ene', 'Feb', 'Mar']]
    plt.figure()
    df_tur = dda_RMaule.iloc[0:1,:-1].transpose()
    df_tur.plot.bar()
    plt.ylabel('caudal $m^3/s$')
    plt.title('Caudal de reserva de uso turístico cuenca del Río '+cuenca)
    plt.grid()