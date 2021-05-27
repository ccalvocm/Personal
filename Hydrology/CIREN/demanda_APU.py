#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:41:24 2021

@author: faarrosp
"""

# Script que se preocupa de obtener las tablas y plots para el informe.
# La mayoria del procesamiento previo se puede obtener en el Jupyter Notebook
# de nombre DemandaOtrosUsosFAA.ipynb

import pandas as pd
import os
from matplotlib import pyplot as plt
import geopandas as gpd
import calendar
from calendar import monthrange

# ---------------------
# definir procesos
# ---------------------
def read_sifac_1():
    folder_SISS = os.path.join('..', 'Etapa 1 y 2', 
                               'Solicitudes de información', 'Recibidos',
                               'SISS', '4063-4064')
    path_sifacI = os.path.join(folder_SISS,
                               'SIFAC I - Consumos 1998 - 2011.xlsx')
    df = pd.read_excel(path_sifacI, dtype = {'Ano Info': str, 'Mes Info':str})
    
    return df

def import_2012_2020_flows():
    path_folder = os.path.join('..', 'Etapa 1 y 2',
                               'Solicitudes de información', 'Recibidos',
                               'SISS', 'resultados_outputs')
    path_file = 'SISS_AP_lps_Mensual_2012_2020.xlsx'
    fp = os.path.join(path_folder, path_file)
    
    df = pd.read_excel(fp, dtype = {'COD_DGA': str})
    df.drop('NOM_DGA', axis=1, inplace=True)
    # df.set_index('COD_DGA', inplace=True)
    # print(df.dtypes)
    return df

def import_1999_2011_flows():
    path_folder = os.path.join('..', 'Etapa 1 y 2',
                               'Solicitudes de información', 'Recibidos',
                               'SISS', 'resultados_outputs')
    path_file = 'SISS_AP_lps_Mensual_1999_2011.xlsx'
    fp = os.path.join(path_folder, path_file)
    
    df = pd.read_excel(fp, dtype = {'COD_DGA': str})
    # df.set_index('COD_DGA', inplace=True)
    return df

def import_subcatchments():
    path_folder= os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
    path_macrocuencas =  os.path.join(path_folder, 'Cuencas',
                                      'Cuencas_DARH_2015_AOHIA_ZC.geojson')
    path_subcuencas = os.path.join(path_folder, 'Subcuencas',
                                   'SubCuencas_DARH_2015_AOHIA_ZC.geojson')
    gdfcuencas = gpd.read_file(path_macrocuencas)
    gdfsubcuencas = gpd.read_file(path_subcuencas)
    
    return gdfcuencas, gdfsubcuencas

def compute_macrocuencas_TS(df_consumos, gdf_cuencas):
    # df_consumos.drop('NOM_DGA', axis=1, inplace = True)
    fields = ['COD_DGA', 'NOM_CUENCA']
    df = df_consumos.join(gdf_cuencas[fields].set_index('COD_DGA'),
                          on = 'COD_DGA')
    try:
        df.drop(['COD_DGA', 'COD_CUENCA'], axis=1, inplace=True)
    except:
        df.drop(['COD_DGA'], axis=1, inplace=True)
    df = df.groupby('NOM_CUENCA').sum()
    
    df = df.T
    return df

def plot_macrocuencas_TS(df_TS, extension, name):
    fig = plt.figure(figsize = (11,8.5))
    # ax = fig.add_subplot(111)
    
    df_TS.plot(subplots=True, layout=(2,2), legend=False,
               xlabel='Fecha', ylabel='Demanda (L/s)',
               title=['Rio Maipo', 'Rio Mataquito', 'Rio Maule', 'Rio Rapel'],
               figsize = (11,8.5))
    
    path_folder = os.path.join('..', 'Etapa 1 y 2',
                               'Solicitudes de información', 'Recibidos',
                               'SISS', 'resultados_outputs')
    path_file = name + '.' + extension
    fp = os.path.join(path_folder, path_file)
    
    plt.show()
    plt.savefig(fp, format = extension, bbox_inches = 'tight',
                pad_inches = 0.1)
    
def join_1999_2012(df1,df2):
    df = df1.join(df2.set_index('COD_DGA'), on='COD_DGA', how='outer')
    return df
    

# sifac1 = read_sifac_1()
# importar consumos mensuales por subsubcuenca DARH desde 2012 a 2020
df2012 = import_2012_2020_flows()
df1998 = import_1999_2011_flows()
df = join_1999_2012(df1998, df2012)

# importar geodataframes de las cuencas y subcuencas
gdf_cuencas, gdf_subcuencas = import_subcatchments()

# calcular el dataframe que contiene macrocuencas en columnas y fecha en filas
# determinar serie de tiempo de 2012 a 2020

df_TS_1998 = compute_macrocuencas_TS(df1998,gdf_subcuencas)
df_TS_2012 = compute_macrocuencas_TS(df2012,gdf_subcuencas)
df_TS = compute_macrocuencas_TS(df,gdf_subcuencas)

folder = os.path.join('..', 'Etapa 1 y 2', 'Demanda', 'APU')
fp = os.path.join(folder, 'Demanda_APU_TS_macrocuencas.xlsx')

df_TS_vol = df_TS.T.copy()
df_TS_vol.columns = pd.to_datetime(df_TS_vol.columns)

for col in df_TS_vol.columns:
    num_days = monthrange(col.year, col.month)[1]
    df_TS_vol.loc[:,col] = df_TS_vol.loc[:,col] * 60 * 60 * 24 * num_days
    
df_TS_vol = df_TS_vol.resample('Y', axis = 1).sum()
df_TS_vol.columns = pd.to_datetime(df_TS_vol.columns)

df_TS_LPS = df_TS_vol.copy()

    
for col in df_TS_LPS.columns:
    if calendar.isleap(col.year):
        df_TS_LPS.loc[:,col] = df_TS_LPS.loc[:,col] / 366 / 24 / 60 / 60
    else:
        df_TS_LPS.loc[:,col] = df_TS_LPS.loc[:,col] / 365 / 24 / 60 / 60
    
df_TS_LPS.to_excel(fp)
    
fp = os.path.join(folder, 'demanda_APU_SIFAC_I_LPS_subcuencas.xlsx')
df1998.to_excel(fp)

fp = os.path.join(folder, 'demanda_APU_SIFAC_II_LPS_subcuencas.xlsx')
df2012.to_excel(fp)


# plot_macrocuencas_TS(df_TS, 'jpg', 'Demanda_APU_lps_1999_2020_Macrocuencas')
# plot_macrocuencas_TS(df_TS_1998, 'jpg', 'Demanda_APU_lps_1999_Macrocuencas')
# plot_macrocuencas_TS(df_TS_2012, 'jpg', 'Demanda_APU_lps_2012_Macrocuencas')
