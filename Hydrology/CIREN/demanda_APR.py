#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 13:47:44 2021

@author: faarrosp
"""

import os
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt

def import_database():
    folder = os.path.join('..', 'Etapa 1 y 2', 'Solicitudes de información',
                          'Recibidos', 'DOH')
    file = 'Base_SSR_Junio_2020_(1962_Sistemas).shp'
    fp = os.path.join(folder,file)
    gdf = gpd.read_file(fp)
    return gdf

def import_subcatchments():
    path_folder= os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH')
    path_macrocuencas =  os.path.join(path_folder, 'Cuencas',
                                      'Cuencas_DARH_2015_AOHIA_ZC.geojson')
    path_subcuencas = os.path.join(path_folder, 'Subcuencas',
                                   'SubCuencas_DARH_2015_AOHIA_ZC.geojson')
    
    gdfcuencas = gpd.read_file(path_macrocuencas)
    gdfsubcuencas = gpd.read_file(path_subcuencas)
    
    path_folderBNA= os.path.join('..', 'Etapa 1 y 2', 'GIS', 'CuencasBNA')
    path_subcuencasBNA =  os.path.join(path_folderBNA, 'Subcuencas_BNA',
                                      'Subcuencas_BNA.shp')
    gdfsubcuencasBNA = gpd.read_file(path_subcuencasBNA)
    # gdfsubcuencasBNA = gpd.sjoin(gdfsubcuencasBNA, gdfcuencas, how = 'inner')
    # gdfsubcuencasBNA.drop('index_right', axis=1, inplace=True)
    
    return gdfcuencas, gdfsubcuencas, gdfsubcuencasBNA

def savefig(folderpath, filename, ext_img):
    fig = plt.gcf()
    filename = os.path.join(folderpath, filename + '.' + ext_img)
    plt.savefig(filename, format = ext_img, bbox_inches = 'tight',
                pad_inches = 0)
    plt.close(fig)

gdf = import_database()

gdf_cuencas, gdf_subcuencas, gdf_subcuencasBNA = import_subcatchments()

APR_cuencas = gpd.sjoin(gdf,gdf_subcuencas, how='inner')

# dotacion 140 L/hab/dia = 140 L / hab / (24 hr * 60 min * 60 seg) 
APR_cuencas['dda_LPS'] = APR_cuencas['Beneficiar'] * 140 / (24*60*60)

APR_TS_lps_cuencas = APR_cuencas.pivot_table(index=['COD_CUENCA'],
                                             columns = 'año_puest',
                                             values = 'dda_LPS',
                                             aggfunc='sum')

APR_TS_lps_cuencas.fillna(0, inplace=True)
APR_TS_lps_cuencas = APR_TS_lps_cuencas.cumsum(axis=1)

#%%
APR_TS_lps_cuencas = APR_TS_lps_cuencas.reindex(labels=['1300',
                                                        '0600', '0701',
                                                        '0703'])


APR_TS_lps_cuencas.T.plot(figsize = (8.5,11), 
                          subplots=True, layout=(2,2), legend=False,
                          title = ['Río Maipo', 'Río Rapel', 'Río Mataquito',
                                   'Río Maule'],
                          xlabel = 'Fecha',
                          ylabel = 'Demanda (L/s)')

folder = os.path.join('..', 'Etapa 1 y 2', 'Demanda', 'APR')
file = 'Demanda_APR_LPS_Macrocuencas'
fp = os.path.join(folder,file)
savefig(folder, file, 'jpg')

fp = os.path.join(folder, 'Demanda_APR_TS_macrocuencas.xlsx')
APR_TS_lps_cuencas.to_excel(fp)