#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 01:21:46 2021

@author: faarrosp
"""

import os
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
import contextily as ctx
import calendar

def import_subcatchments():
    path_folder= os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
    path_macrocuencas =  os.path.join(path_folder, 'Cuencas',
                                      'Cuencas_DARH_2015_AOHIA_ZC.geojson')
    path_subcuencas = os.path.join(path_folder, 'Subcuencas',
                                   'SubCuencas_DARH_2015_AOHIA_ZC.geojson')
    gdfcuencas = gpd.read_file(path_macrocuencas)
    gdfsubcuencas = gpd.read_file(path_subcuencas)
    gdfsubcuencas['COD_DGA'] = gdfsubcuencas['COD_DGA'].astype(int)
    
    return gdfcuencas, gdfsubcuencas

def import_dataframes():
    path_folder = os.path.join('..', 'Etapa 1 y 2', 'Demanda', 'IND')
    fp_sifacII = os.path.join(path_folder,
                              'SISS_Industrial_Instant_2012_2020.xlsx')
    fp_retc = os.path.join(path_folder,
                           'dda_subcuenca_L_s_RETC.xlsx')
    sifacII = pd.read_excel(fp_sifacII, dtype={'COD_DGA':str}, index_col=0)
    retc = pd.read_excel(fp_retc, dtype={'COD_DGA':str}, index_col=0)
    
    sifacII.drop('NOM_DGA', axis=1, inplace=True)
    
    return sifacII, retc

def compute_macrocuencas_TS(df_consumos, gdf_cuencas):
    # df_consumos.drop('NOM_DGA', axis=1, inplace = True)
    fields = ['COD_DGA', 'NOM_CUENCA']
    df = gdf_cuencas[fields].join(df_consumos, on = 'COD_DGA')
                          
    try:
        df.drop(['COD_DGA', 'COD_CUENCA'], axis=1, inplace=True)
    except:
        # pass
        df.drop(['COD_DGA'], axis=1, inplace=True)
    df = df.groupby('NOM_CUENCA').sum()
    
    # df = df.T
    return df

def savefig(folderpath, filename, ext_img):
    fig = plt.gcf()
    filename = os.path.join(folderpath, filename + '.' + ext_img)
    plt.savefig(filename, format = ext_img, bbox_inches = 'tight',
                pad_inches = 0.1)
    plt.close(fig)

def plot_TS(df, title):
    df.index = ['1300', '0701', '0703', '0600']
    df.reindex(['1300', '0600', '0701', '0703']).T.plot(subplots=True, title=['Río Maipo',
                                    'Río Rapel',
                                    'Río Mataquito',
                                    'Río Maule'],
              figsize=(8.5,11), layout=(2,2), legend=False,
              xlabel='Fecha', ylabel='Demanda (L/s)')
    plt.suptitle('Demanda Industrial\n' + title)
    

gdf_cuencas, gdf_subcuencas = import_subcatchments()

df_sifac, df_retc = import_dataframes()


folder = os.path.join('..', 'Etapa 1 y 2', 'Demanda', 'IND')

TS_sifac = compute_macrocuencas_TS(df_sifac, gdf_subcuencas)
fp = os.path.join(folder, 'Demanda_IND_TS_SIFAC_macrocuencas.xlsx')
TS_sifac.to_excel(fp)


TS_retc = compute_macrocuencas_TS(df_retc, gdf_subcuencas)
TS_retc.columns = pd.to_datetime(TS_retc.columns, format="%Y")
TS_retc = TS_retc.resample('Y', axis = 1).sum()

fp = os.path.join(folder, 'Demanda_IND_TS_RETC_macrocuencas.xlsx')
TS_retc.to_excel(fp)

TS_total = pd.concat([TS_retc, TS_sifac])
TS_total = TS_total.groupby(level = 0).sum()
fp = os.path.join(folder, 'Demanda_IND_TS_TOTAL_macrocuencas.xlsx')
TS_total.to_excel(fp)

plot_TS(TS_sifac, 'SIFAC II')
savefig(folder, 'Demanda_IND_SIFAC_II_TS_macrocuencas', 'jpg')
plot_TS(TS_retc, 'RETC')
savefig(folder, 'Demanda_IND_RETC_TS_macrocuencas', 'jpg')

