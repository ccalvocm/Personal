# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 08:27:24 2021

@author: Carlos
"""

#########################
###     Preámbulo     ###
#########################

import scipy
import pandas as pd
import numpy as np
import os
import lxml
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import geopandas
import modules_CCC
import contextily as ctx
import modules_FAA

#########################
## Graficar con comas ###
#########################
#Locale settings
import locale
# Set to Spanish locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "es_ES")

plt.rcdefaults()

# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True


def copyIndex(df):
    for col in df:
        df.loc[df.index,col] = df.index.daysinmonth
    return df
    
def pronostico(df,meses, orden):    # AR example
    # SARIMA 
    # fit model
    model = SARIMAX(df, order = orden, seasonal_order=(1, 1, 3, 12))
    model_fit = model.fit(disp=False)
    # make prediction
    return model_fit.forecast(meses)

def pronostico_ARMA(df,meses, orden):
    # fit model
    model = ARIMA(df, order=orden)
    model_fit = model.fit()
    # make prediction
    return model_fit.forecast(meses), model_fit.get_forecast(meses).conf_int(alpha=0.3)

def flags_mon(df):
    df_flag = df.copy()
    df_flag[:] = 1
    df_flag[df.isnull()] = 0
    df_flag = df_flag.resample('MS').sum()
    df_mon = df.copy().apply(pd.to_numeric).resample('MS').mean()[df_flag > 20]
    return df_mon

def completarVIC(ruta_VIC, df, estacion):
    # ------completar con VIC
    q_VIC = pd.read_csv(ruta_VIC, parse_dates = True, index_col = 0)
    
    df_2 = pd.DataFrame([], index = pd.date_range('1979-01-01',max(max(df.index),pd.to_datetime('2015-12-01')),freq = 'MS'), columns = [estacion])
    df_2.loc[df.index,estacion] = df
    
    idx = df_2.isnull().index.intersection(q_VIC.index)
    df_2.loc[idx,estacion] = q_VIC.loc[idx,'Salida_'+estacion[1:-2]]
    
    return df_2

def parseVIC(ruta):
    df = pd.read_csv(ruta)
    fechas = pd.to_datetime(df['Year'].astype(str) + df['month'].astype(str), format='%Y%m')
    df.index = fechas.values
    df = df.drop(['month', 'Year'], axis = 1)
    return df

# ======================================================
def main():
# ------------------------------------------------------
# funcion para calcular caudales
# ------------------------------------------------------

    # -----limpiar plots
    plt.close("all")

    # primero rutas
    ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2'
    os.chdir(ruta_OD.replace('Etapa 1 y 2','')+r'\scripts')
    
    # --------cuenca
    cuenca = 'Maipo'
    cuenca = 'Rapel'
    cuenca = 'Mataquito'
    cuenca = 'Maule'
    
    ruta_fb_mon = ruta_OD + '\\datos\\VIC\\'+cuenca+'_CNC\\FB _Salidas_monthly.csv'
    ruta_ro_mon = ruta_OD + '\\datos\\VIC\\'+cuenca+'_CNC\\runoff _Salidas_monthly.csv'
    
    cnc ={'Maipo' : ['Salida_18', 'Salida_19'] , 'Rapel' : ['Salida_'+str(x) for x in [2,3,4]],
          'Mataquito' : ['Salida_'+str(x) for x in [0,2,4,5]], 'Maule' : ['Salida_'+str(x) for x in [1,15,
   18,21]]}
    
    cnc_names= {'Maipo' : ['RIO PEUCO EN CHADA TRONCO','RIO LONGITUDINAL EN TUC AGUILA NORTE AGUILA SUR'],
                'Rapel' : ['Estero Antivero en El Rincón Antivero', 'Estero Zamorano en Alto Requena', 'Estero Rigolemo en Rigolemo'], 
                'Mataquito' : ['Estero Potrero Grande en CERRO DE POTRERO GRANDE CHICO','Estero Potrero Grande en Toma DOS',
                               'Río El Manzano en VERTIENTE DOS','EStero La Palmilla en TILICURA ALTO'], 
                'Maule' : ['Río Claro en Las Mercedes Claro',
                           'QUEBRADA TEATINOS EN LOS MAITENES CORINTO', 'Río catillo en matriz digua',
                           'Río Lircay en Nunez']}
   #  cnc ={'Maipo' : ['Salida_18', 'Salida_19'] , 'Rapel' : ['Salida_'+str(x) for x in [2,3,4]],
   #        'Mataquito' : ['Salida_'+str(x) for x in [0,2,4,5]], 'Maule' : ['Salida_'+str(x) for x in [1,12,15,
   # 18,19,20,21,27,28]]}
    
   #  cnc_names= {'Maipo' : ['RIO PEUCO EN CHADA TRONCO','RIO LONGITUDINAL EN TUC AGUILA NORTE AGUILA SUR'],
   #              'Rapel' : ['Estero Antivero en El Rincón Antivero', 'Estero Zamorano en Alto Requena', 'Estero Rigolemo en Rigolemo'], 
   #              'Mataquito' : ['Estero Potrero Grande en CERRO DE POTRERO GRANDE CHICO','Estero Potrero Grande en Toma DOS',
   #                             'Río El Manzano en VERTIENTE DOS','EStero La Palmilla en TILICURA ALTO'], 
   #              'Maule' : ['Río Claro en Las Mercedes Claro','Estero Picazo en RUEDECILLAS',
   #                         'QUEBRADA TEATINOS EN LOS MAITENES CORINTO', 'Río catillo en matriz digua',
   #                         'río Rari en Rojas Uno', 'Estero Vilchez en MOLINO VIEJO',
   #                         'Río Lircay en Nunez','Estero las Toscas en Tránsito', 'Estero Los Pellines en El Espinal']}
    
    q_vic = parseVIC(ruta_fb_mon)+parseVIC(ruta_ro_mon)
    q_vic = q_vic[(q_vic.index >= '1980-04-01') & (q_vic.index <= '2015-03-01')][cnc[cuenca]]
    
    q_vic.columns = cnc_names[cuenca]
    
    q_vic.to_csv(ruta_OD+'\\datos\\q_'+cuenca+'_CNC_mon.csv')
    
    plt.close("all")
    
    caudales_nam = modules_CCC.CDA('q_'+cuenca+'_CNC_mon.csv')
    
    if cuenca == 'Maule':
        for i in range(4,len(caudales_nam.columns)+4,4):
            # caudales medios anuales
            modules_CCC.CMA(caudales_nam.iloc[:,i-4:i], 10 ,22, 2, 2)
            
            # # curvas de duración de caudales
            fig, axes = plt.subplots(2,2,figsize=(10, 22))
            modules_CCC.CDQ(caudales_nam.iloc[:,i-4:i], 4, fig,  axes)
            
            # # curvas de variación estacional
            fig, axes = plt.subplots(2,2,figsize=(10, 22))
            axes = axes.reshape(-1)
            modules_CCC.CVE_1979_2019_mon(caudales_nam.iloc[:,i-4:i], fig, axes, 4, 1980, 2015)
    
            # anomalias
            fig, axes = plt.subplots(2,2,figsize=(10, 22))
            axes = axes.reshape(-1)
            modules_CCC.ANOM(caudales_nam.iloc[:,i-4:i], 20, 11, 0.72, 0.02, 110, 'MS', fig, axes)
    else:
        # curvas de variación estacional
        fig, axes = plt.subplots(2,2,figsize=(10, 22))
        axes = axes.reshape(-1)
        modules_CCC.CVE_1979_2019_mon(caudales_nam, fig, axes,4, 1980, 2015)
        
        # # curvas de duración de caudales
        fig, axes = plt.subplots(2,2,figsize=(10, 22))
        modules_CCC.CDQ(caudales_nam, 4, fig,  axes)
        
        # anomalias
        fig, axes = plt.subplots(2,2,figsize=(10, 22))
        axes = axes.reshape(-1)
        modules_CCC.ANOM(caudales_nam, 20, 11, 0.72, 0.02, 115, 'MS', fig, axes)
                        
        
def plots(cuenca, cnc):
    
    #rutas
    # --------------------------------------------------------------------
    ruta_VB = r'D:\Compartida-VB\AOHIAZC'
    ruta_hidrograph = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\GIS\Hidrografia\Red_Hidrografica.shp'
    ruta_cuencas = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Mapoteca DGA\Mapoteca_DGA\02_DGA\Cuencas\Cuencas_DARH_2015_Cuencas.shp'
    # --------------------------------------------------------------------
    
    # diccionario Cuencas
    cnc_shp = {'Maipo' : ['CNC_Maipo.shp','1300'], 'Rapel' : ['CNC_Rapel.shp', '0600'], 'Mataquito' : ['CNC_Mataquito.shp', '0701'],
               'Maule' : ['CNC_Maule.shp', '0703']}
    
    # hidrografia
    hidrograph = geopandas.read_file(ruta_hidrograph)
    hidrograph = hidrograph.to_crs(epsg = 32719)
        
    # cuenca
    cuenca_shp = geopandas.read_file(ruta_cuencas)    
    cuenca_shp = cuenca_shp[cuenca_shp['COD_CUENCA'] == cnc_shp[cuenca][1]]
            
    gdf = geopandas.read_file(ruta_VB + '\\' + cnc_shp[cuenca][0])
    if cuenca not in ['Rapel']:
        gdf.crs = {'init': 'epsg:4326'} 
        gdf = gdf.to_crs({'init': 'epsg:32719'})
    est_gdf = [int(x.split('_')[-1]) for x in cnc[cuenca]]
    gdf = gdf[gdf['COD_est'].isin(est_gdf)]
    
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    
    modules_FAA.plot_catchment_map(cuenca_shp, bsn_N = cnc_shp[cuenca][1], ax = axes, basemap = True, linewidth = 5, edgecolor = 'black', label = 'Limite cuenca')
    gdf.plot(ax = axes, color = 'blue', zorder = 3)
    gdf_hidrograf = geopandas.read_file(ruta_hidrograph)
    gdf_hidrograf = hidrograph.to_crs(epsg = 32719)
    # clip y recuento de hidrografía
    gdf_hidrograf = geopandas.clip(gdf_hidrograf, cuenca_shp)
    gdf_hidrograf.plot(ax = axes)
    axes.set_xlabel('Coordenada Este WGS84 19S (m)')
    axes.set_ylabel('Coordenada Norte WGS84 19S (m)')
    axes.legend(['Cuenca Río Maipo','Subcuencas no controladas','Hidrografía'], loc = 'upper left')



