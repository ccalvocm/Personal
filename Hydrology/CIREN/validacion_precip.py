# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# importar librerias
import geopandas as gpd
import os
import rasterstats
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import rasterio
import numpy as np
import plotly.graph_objects as go
import re
from hydroeval import evaluator, nse
import math

def notna_rows(obj):
    notna_indices = []
    for index in obj.index:
        if obj.loc[index,:].notna().all():
            notna_indices.append(index)
        else:
            pass
    return obj.loc[notna_indices,:]

#%% Cargar la capa de cuencas
# path_catchment = '/home/faarrosp/Insync/farrospide@ciren.cl/' + \
#     'OneDrive Biz - Shared/AOHIA_ZC/Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/' + \
#         'Cuencas_DARH_2015.shp'
path_catchment = os.path.join('..',
                              'Etapa 1 y 2',
                              'GIS',
                              'Cuencas_DARH',
                              'Cuencas',
                              'Cuencas_DARH_2015.shp')
catchment_gdf = gpd.read_file(path_catchment)

# filtrar por cuenca de interes (0600 Rapel 1300 Maipo 0703 Maule 0701 Mata...)
filtro = catchment_gdf['COD_CUENCA'].isin(['0600', '1300', '0703', '0701'])
catchment_gdf = catchment_gdf[filtro]
catchment_gdf = catchment_gdf.to_crs('EPSG:4326')

# definir la bbox a cortar
xmin = catchment_gdf.geometry.bounds['minx'].values.min()
xmax = catchment_gdf.geometry.bounds['maxx'].values.max()
ymin = catchment_gdf.geometry.bounds['miny'].values.min()
ymax = catchment_gdf.geometry.bounds['maxy'].values.max()


    
#%% Analizar los rasters

# path_CCS = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
#       'AOHIA_ZC/Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CCS/pp.nc'
# path_CDR = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
#       'AOHIA_ZC/Etapa 1 y 2/GIS/PERSIANN/PERSIANN-CDR/pp.nc'
# path_CR2 = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
#       'AOHIA_ZC/Etapa 1 y 2/GIS/cr2MET/cr2MET_coords.nc'
# path_GPM = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
#       'AOHIA_ZC/Etapa 1 y 2/GIS/GPM/concat/GPM_concat.nc'   

path_CCS = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'PERSIANN', 'PERSIANN-CCS',
                        'pp.nc')
path_CDR = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'PERSIANN', 'PERSIANN-CDR',
                        'pp.nc')
path_CR2 = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'cr2MET',
                        'cr2MET_coords.nc')
path_GPM = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'GPM', 'concat',
                        'GPM_concat.nc')

pp_CCS = xr.open_dataset(path_CCS)
pp_CDR = xr.open_dataset(path_CDR)
pp_CR2 = xr.open_dataset(path_CR2)

# importar puntos de estaciones

paths = ['/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Maipo/RIO MAIPO_P_diario.xlsx',
      '/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Rapel/RIO RAPEL_P_diario.xlsx',
      '/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Mataquito/RIO MATAQUITO_P_diario.xlsx',
      '/home/faarrosp/Documents/GitHub/Analisis-Oferta-Hidrica/' + \
      'DGA/datosDGA/Pp/Maule/RIO MAULE_P_diario.xlsx']

path_Windows = os.path.join('..', 'Etapa 1 y 2', 'datos', 'datosDGA', 'Pp')
path_maipop = os.path.join(path_Windows, 'Maipo',
                           'RIO MAIPO_P_diario.xlsx')    
path_rapelp = os.path.join(path_Windows, 'Rapel',
                           'RIO RAPEL_P_diario.xlsx')
path_mataquitop = os.path.join(path_Windows, 'Mataquito',
                           'RIO MATAQUITO_P_diario.xlsx') 
path_maulep = os.path.join(path_Windows, 'Maule',
                           'RIO MAULE_P_diario.xlsx')    
   
paths = [path_maipop, path_rapelp, path_mataquitop, path_maulep]    


stations_gdf_array = []

for path in paths:
    stations = pd.read_excel(path, sheet_name = 'info estacion')
    stations[['latdd', 'londd']] = float()
    for index in stations.index:
        latstr = stations.loc[index, 'Latitud'].split(" ")
        d,m,s = latstr[0][:2], latstr[1][:2], latstr[2][:2]
        stations.loc[index, 'latdd'] = -(float(d) + float(m)/60 + float(s)/60/60)
        lonstr = stations.loc[index, 'Longitud'].split(" ")
        d,m,s = lonstr[0][:2], lonstr[1][:2], lonstr[2][:2]
        stations.loc[index, 'londd'] = -(float(d) + float(m)/60 + float(s)/60/60)
    # if 'MAIPO' in path or 'RAPEL' in path:
    #     crsstring = 'EPSG:32719'
    # elif 'MAULE' in path or 'MATAQUITO' in path:
    #     crsstring = 'EPSG:32718'
    gdf = gpd.GeoDataFrame(stations,
                                    geometry = gpd.points_from_xy(stations['londd'],
                                                                  stations['latdd']),
                                    crs = 'EPSG:4326')
    # if ('MATAQUITO' in path) or ('MAULE' in path):
    #     gdf = gdf.to_crs('EPSG:32719')
    # else:
    #     pass
    stations_gdf_array.append(gdf)
    
stations_gdf = pd.concat(stations_gdf_array)
# stations_gdf.plot()
# stations_gdf = stations_gdf.to_crs('EPSG:32719')
stations_gdf = stations_gdf.reset_index(drop = True)
del stations_gdf_array
#%% Comparacion series de tiempo
path_ccs_df = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/scripts/outputs/productos_grillados/PERSIANN-CCS.xlsx'
path_cdr_df = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/scripts/outputs/productos_grillados/PERSIANN-CDR.xlsx'
path_cr2_df = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/scripts/outputs/productos_grillados/cr2MET.xlsx'

path_folder = os.path.join('.', 'outputs', 'productos_grillados')      
path_ccs_df = os.path.join(path_folder, 'PERSIANN-CCS.xlsx')
path_cdr_df = os.path.join(path_folder, 'PERSIANN-CDR.xlsx')
path_cr2_df = os.path.join(path_folder, 'cr2MET.xlsx')
path_gpm_df = os.path.join(path_folder, 'GPM.xlsx')


path_ctol_Maipo = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/Etapa 1 y 2/Clima/Pp/Maipo/Relleno/MaipoP_1979-2020_series.csv'
path_ctol_Rapel = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/Etapa 1 y 2/Clima/Pp/Rapel/Relleno/RapelP_1979-2020_series.csv'
path_ctol_Mataquito = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/Etapa 1 y 2/Clima/Pp/Mataquito/Relleno/Pp_CLIMATOL_Mataquito_1979-2017_series.csv'
path_ctol_Maule = '/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
      'AOHIA_ZC/Etapa 1 y 2/Clima/Pp/Maule/Relleno/Pp_CLIMATOL_Maule_1979-2019_series.csv'

path_folder = os.path.join('..', 'Etapa 1 y 2', 'Clima', 'Pp')
path_ctol_Maipo = os.path.join(path_folder, 'Maipo', 'Relleno',
                               'MaipoP_1979-2020_series.csv')
path_ctol_Rapel = os.path.join(path_folder, 'Rapel', 'Relleno',
                               'RapelP_1979-2020_series.csv')
path_ctol_Mataquito = os.path.join(path_folder, 'Mataquito', 'Relleno',
                               'Pp_CLIMATOL_Mataquito_1979-2017_series.csv')
path_ctol_Maule = os.path.join(path_folder, 'Maule', 'Relleno',
                               'Pp_CLIMATOL_Maule_1979-2019_series.csv')



paths_ctol = [path_ctol_Maipo, path_ctol_Rapel,
              path_ctol_Mataquito, path_ctol_Maule]

df_CDR = pd.read_excel(path_cdr_df, index_col = 0)
df_CCS = pd.read_excel(path_ccs_df, index_col = 0)
df_cr2 = pd.read_excel(path_cr2_df, index_col = 0)
df_gpm = pd.read_excel(path_gpm_df, index_col = 0)


stations_data = []
stations_info = []
for path_ctol, path in zip(paths_ctol, paths):
    stations = pd.read_csv(path_ctol, index_col = 0, parse_dates = True)
    stations_data.append(stations)
    stations2 = pd.read_excel(path, sheet_name = 'info estacion', index_col = 1)
    stations_info.append(stations2)

stations_df = pd.concat(stations_data, axis = 1)
stations_info = pd.concat(stations_info)
# stations_df = stations_df.reset_index(drop = True)
del stations_data, stations
#%%

NSE_tot_CDR = []
NSE_tot_CCS = []
NSE_tot_CR2 = []
NSE_tot_GPM = []
code = []
for column in stations_df.columns:
    # fig = plt.figure(figsize = (11, 8.5))
    # ax1 = fig.add_subplot(141)
    # ax2 = fig.add_subplot(142)
    # ax3 = fig.add_subplot(143)
    # ax4 = fig.add_subplot(144)
    try:
        # plot timeseries
        # stations_df[column].resample('Y').sum().plot(ax = ax1, color = 'red', label = 'DGA')
        # stations_df[column].resample('Y').sum().plot(ax = ax2, color = 'red', label = 'DGA')
        # stations_df[column].resample('Y').sum().plot(ax = ax3, color = 'red', label = 'DGA')
        # df_CDR[column].resample('Y').sum().plot(ax = ax1, color = 'green', label = 'PERSIANN-CDR')
        # df_CCS[column].resample('Y').sum().plot(ax = ax2, color = 'blue', label = 'PERSIANN-CCS')
        # df_cr2[column].resample('Y').sum().plot(ax = ax3, color = 'black', label = 'CR2-MET')
        # df_gpm[column].resample('Y').sum().plot(ax = ax4, color = 'black', label = 'GPM')
        # title = stations_info.loc[column, 'Nombre estacion']
        # fig.suptitle('\n'.join([column, title]))
        # ax1.legend()
        # ax2.legend()
        # ax3.legend()
        # ax4.legend()
        
        # compute NSE --->
        
        sim_CDR = df_CDR[column]#.resample('Y').sum()
        sim_CCS = df_CCS[column]#.resample('Y').sum()
        sim_CR2 = df_cr2[column]
        sim_GPM = df_gpm[column]
        eva_DGA = stations_df[column]#.resample('Y').sum()
        
        comp_CDR = pd.concat([sim_CDR, eva_DGA], axis = 1)
        comp_CCS = pd.concat([sim_CCS, eva_DGA], axis = 1)
        comp_CR2 = pd.concat([sim_CR2, eva_DGA], axis = 1)
        comp_GPM = pd.concat([sim_GPM, eva_DGA], axis = 1)
            
        cdr = comp_CDR[comp_CDR.notna().all(axis=1)]
        ccs = comp_CCS[comp_CCS.notna().all(axis=1)]
        cr2 = comp_CR2[comp_CR2.notna().all(axis=1)]
        gpm = comp_GPM[comp_GPM.notna().all(axis=1)]
        
        CDR_pred = np.array(cdr.iloc[:,0].values)
        CDR_obs = np.array(cdr.iloc[:,1].values)
        
        CCS_pred = np.array(ccs.iloc[:,0].values)
        CCS_obs = np.array(ccs.iloc[:,1].values)
        
        CR2_pred = np.array(cr2.iloc[:,0].values)
        CR2_obs = np.array(cr2.iloc[:,1].values)
        
        GPM_pred = np.array(gpm.iloc[:,0].values)
        GPM_obs = np.array(gpm.iloc[:,1].values)
        
        my_nse_CDR = evaluator(nse, CDR_pred, CDR_obs)[0]
        my_nse_CCS = evaluator(nse, CCS_pred, CCS_obs)[0]
        my_nse_CR2 = evaluator(nse, CR2_pred, CR2_obs)[0]
        my_nse_GPM = evaluator(nse, GPM_pred, GPM_obs)[0]
        
        
        NSE_tot_CDR.append(my_nse_CDR)
        NSE_tot_CCS.append(my_nse_CCS)
        NSE_tot_CR2.append(my_nse_CR2)
        NSE_tot_GPM.append(my_nse_GPM)
        code.append(column)
    except:
        # plt.close()
        pass

df_validation = pd.DataFrame(data = {'NSE-CDR': NSE_tot_CDR,
                                     'NSE-CCS': NSE_tot_CCS,
                                     'NSE-CR2': NSE_tot_CR2,
                                     'NSE-GPM': NSE_tot_GPM}, index = code)

print(df_validation.mean())

#%% Plot NSE values in map

# stations_gdf.index = stations_gdf['Codigo Estacion']
stations_gdf.drop_duplicates(inplace = True)
df_validation = df_validation[df_validation > -1000]
df_validation = df_validation[df_validation.notna().any(axis = 1)]

fig = plt.figure(figsize = (17,17))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.set_title('NSE: PERSIANN-CDR')
ax2.set_title('NSE: PERSIANN-CCS')
ax3.set_title('NSE: CR2MET')

catchment_gdf.plot(ax = ax1)
catchment_gdf.plot(ax = ax2)
catchment_gdf.plot(ax = ax3)
stations_gdf.plot(ax = ax1, color = 'black')
stations_gdf.plot(ax = ax2, color = 'black')
stations_gdf.plot(ax = ax3, color = 'black')

for station in df_validation.index:
    x = float(stations_gdf.loc[station].geometry.x)
    y = float(stations_gdf.loc[station].geometry.y)
    if math.isnan(df_validation.loc[station, 'NSE-CDR']):
        s1 = ''
    else:
        s1 = str(round(df_validation.loc[station, 'NSE-CDR'],2))
    if math.isnan(df_validation.loc[station, 'NSE-CCS']):
        s2 = ''
    else:
        s2 = str(round(df_validation.loc[station, 'NSE-CCS'],2))
    if math.isnan(df_validation.loc[station, 'NSE-CR2']):
        s3 = ''
    else:
        s3 = str(round(df_validation.loc[station, 'NSE-CR2'],2))
    ax1.annotate(text=s1, xy=(x,y), ha='center')
    ax2.annotate(text=s2, xy=(x,y), ha='center')
    ax3.annotate(text=s3, xy=(x,y), ha='center')
    




