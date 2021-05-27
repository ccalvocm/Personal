# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:26:37 2021

@author: Carlos
"""

#########################
###     Preámbulo     ###
#########################

import pandas as pd 
import matplotlib.pyplot as plt
import os
from hydroeval import evaluator, nse
import numpy as np
from textwrap import wrap
from scipy.signal import find_peaks
from scipy import interpolate
from scipy.optimize import curve_fit
from hydrobox import toolbox
import fiscalyear
import scipy.stats as st
import freqAnalysis
import modules_CCC
import geopandas
import modules_FAA
from matplotlib.lines import Line2D
import matplotlib
from matplotlib.patches import Patch
from matplotlib.patches import Circle

fiscalyear.START_MONTH = 4    

    
def main():
    
    ##############
    #   Cuenca   #
    ############## 
    cuenca = 'Maipo'
    # cuenca = 'Rapel'
    # cuenca = 'Mataquito'
    # cuenca = 'Maule'

    
    ############
    #  Rutas   #
    ############
    
    ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\scripts'
    os.chdir(ruta_OD)
    dict_rutas = { 'Maipo' : '..\\Etapa 1 y 2\\datos\\Q_relleno_MLR_Maipo_1980-2020_monthly_PROT.csv',
        'Maule' : '..\\Etapa 1 y 2\\datos\\Q_relleno_MLR_Maule_1980-2020_monthly_NAT.csv'  ,
        'Mataquito' : '..\\Etapa 1 y 2\\datos\\Q_relleno_MLR_Mataquito_1980-2020_monthly_NAT.csv',
        'Rapel' : '..\\Etapa 1 y 2\\datos\\Q_relleno_MLR_Rapel_1980-2020_monthly_NAT.csv'}
    dicc_caudales = {'Maipo' : ['../Etapa 1 y 2/datos/Maipo_cr2corregido_Q.xlsx','Q_relleno_MLR_Maipo_1980-2020_monthly_NAT.csv'], 
                 'Rapel' : ['../Etapa 1 y 2/datos/RIO RAPEL_Q_mensual.xlsx','Q_relleno_MLR_Rapel_1980-2020_monthly_NAT.csv'], 
         'Mataquito' : ['../Etapa 1 y 2/datos/RIO MATAQUITO_mensual.xlsx','Q_relleno_MLR_Mataquito_1980-2020_monthly_NAT_v2.csv'], 
         'Maule' : [r'../Etapa 1 y 2/datos/RIO MAULE_mensual.xlsx','Q_relleno_MLR_Maule_1980-2020_monthly_NAT.csv']}

    # df con Estaciones
    df_estaciones = pd.read_excel(dicc_caudales[cuenca][0], sheet_name='info estacion')
    
    ruta = dict_rutas[cuenca]
    
    #################
    #    Caudales   #
    #################
    
    q_RN = pd.read_csv(ruta, parse_dates = True, index_col = 0)
    caudales_nam = modules_CCC.get_names(q_RN,df_estaciones[['Nombre estacion', 'Codigo Estacion']])
    
    # q_RN = pd.DataFrame(q_RN['05710001-K'], columns = ['05710001-K'])
    # q_RN = pd.DataFrame(q_RN['07321002-K'], columns = ['07321002-K'])
    q_medio_anual = caudales_nam.mean()
    
    ####################################
    #  Distribuciones de probabilidad  #
    ####################################
    
    pbb_exc = [0.2,0.5,.95]
    q_pbb_exc = {0.2 : pd.DataFrame([],index = range(1,13), columns= caudales_nam.columns),
                 0.5 : pd.DataFrame([],index = range(1,13), columns= caudales_nam.columns),
                 0.95 : pd.DataFrame([],index = range(1,13), columns= caudales_nam.columns)}
    
    distros = []
    for col in caudales_nam:
        print(col)
        for pbb in pbb_exc:
            print(pbb)
            caudales = pd.DataFrame(caudales_nam[col], columns = [col])
            # q_pbb_exc[pbb][col] = freqAnalysis.CVE_pdf(caudales, pbb, distros, col)[1]
            q_pbb_exc[pbb][col] = freqAnalysis.CVE_pdf(caudales, pbb, distros, col)[1]
    
    #################
    #    Graficar   #
    #################
    
    plt.close("all")   
    for pbb in q_pbb_exc.keys(): q_pbb_exc[pbb].plot()
    
    ###################
    #   Q Ecológico   #
    ###################
    
    q_eco_a = pd.DataFrame([], index = range(1,13), columns = q_pbb_exc[0.2].columns)
    q_eco_b = pd.DataFrame([], index = range(1,13), columns = q_pbb_exc[0.2].columns)
    q_eco = pd.DataFrame([], index = range(1,13), columns = q_pbb_exc[0.2].columns)

    for col in q_eco_a.columns:
        for i in range(1,13):
            
    ###################
    #   Qeco a) D71   #
    ###################
    
            q95_50_mes = 0.5*q_pbb_exc[0.95].loc[i,col]
            qma_10_mes = 0.1*q_medio_anual[col]
            qma_20_mes = 0.2*q_medio_anual[col]
            
            if q95_50_mes < qma_10_mes:
                q_eco_a.loc[i,col] = qma_10_mes
            else:
                q_eco_a.loc[i,col] = min(q95_50_mes,2.*qma_10_mes)
   
    ###################
    #   Qeco b) D71   #
    ###################
           
            if q95_50_mes < qma_20_mes:
                q_eco_b.loc[i,col] = q95_50_mes
            else:
                q_eco_b.loc[i,col] = qma_20_mes
                
            q_eco.loc[i,col] = min(q_eco_a.loc[i,col], q_eco_b.loc[i,col])
    
    # q_eco = q_eco.reindex(index = [4,5,6,7,8,9,10,11,12,1,2,3])
    # q_eco.index = ['Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic','Ene','Feb','Mar']

    # fig, ax = plt.subplots(1)
    # q_eco.plot(ax = ax)
    
    # os.chdir(ruta_OD)
    q_eco.to_csv('..\\Etapa 1 y 2\\datos\q_eco_'+cuenca+'.csv')
    
    ############################
    #   Q Protección ambiental #
    ############################
    
    q_prot = pd.DataFrame([], index = range(1,13), columns = q_pbb_exc[0.2].columns)
        
    for col in q_prot:
        for i in range(1,13):
            q_prot.loc[i,col] = q_pbb_exc[0.2].loc[i,col] - q_eco.loc[i,col]

    q_prot = q_prot.reset_index(drop=True)
    q_prot = q_prot.reindex(index = [3,4,5,6,7,8,9,10,11,0,1,2])
    q_prot.index = ['Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic','Ene','Feb','Mar']
    q_prot.plot()

    q_prot.to_csv('..\\Etapa 1 y 2\\datos\q_protAmb_'+cuenca+'.csv')
    
    ##########################
    #   Gráficos finales
    ##########################
    
    # Grupos Maipo
    
    grupos = {'Maipo Alto' : ['RIO MAIPO EN LAS HUALTATAS','RIO COLORADO ANTES JUNTA RIO OLIVARES','RIO VOLCAN EN QUELTEHUES'],
              'Maipo Medio' : ['RIO MAIPO EN EL MANZANO','RIO ANGOSTURA EN VALDIVIA DE PAINE'],
              'Río Mapocho Alto' : ['RIO MAPOCHO EN LOS ALMENDROS','ESTERO YERBA LOCA ANTES JUNTA SAN FRANCISCO'],
              'Río Mapocho Bajo' : ['RIO COLINA EN PELDEHUE']}
    
      # Grupos Rapel
     
    grupos = {'Cachapoal Bajo' : ['RIO CLARO EN HACIENDA LAS NIEVES'], 
                'Río Tinguiririca Alto' : ['RIO TINGUIRIRICA BAJO LOS BRIONES','RIO CLARO EN EL VALLE']}
    
    # # Grupos Mataquito
    
    grupos = {'Río Teno' : ['RIO TENO DESPUES DE JUNTA CON CLARO'], 
                'Río Lontué' : ['ESTERO UPEO EN UPEO']}
    
    # Grupos Maule
    grupos = {'Rio Loncomilla' : ['RIO LONGAVI EN EL CASTILLO'], 
                'Río Maule' : ['RIO MAULE EN ARMERILLO']}
       
    df_grupos = pd.DataFrame([], index = q_prot.index, columns = grupos.keys())
    
    for key in grupos.keys():
        df_grupos[key] = q_prot[grupos[key]].sum(axis = 1)
        
    ax = df_grupos.plot.bar(rot=0)
    ax.set_title('Caudal de reserva de protección ambiental cuenca del Río '+cuenca)
    ax.set_ylabel('Caudal $m^3/s$')
    ax.grid()

    return None

def plots():
    
    plt.close("all")
    # cuenca y diccionarios
    cuenca = 'Maipo'
    # cuenca = 'Rapel'
    # cuenca = 'Mataquito'
    cuenca = 'Maule'

    
    dicc_ruta = {'Maipo' : ['\\Maipo\\RIO MAIPO_Q_mensual.xlsx','1300','../Etapa 1 y 2/datos/datosDGA/Q/Maipo/Maipo_cr2corregido_Q.xlsx'], 
                 'Rapel' : ['\\Rapel\\RIO RAPEL_Q_mensual.xlsx','0600','../Etapa 1 y 2/datos/datosDGA/Q/Rapel/RIO RAPEL_Q_mensual.xlsx'], 
                 'Mataquito' : ['\\Mataquito\\RIO MATAQUITO_Q_mensual.xlsx','0701','../Etapa 1 y 2/datos/datosDGA/Q/Mataquito/RIO MATAQUITO_mensual.xlsx'],
                 'Maule' : [r'../Etapa 1 y 2/datos/RIO MAULE_mensual.xlsx','0703',r'../Etapa 1 y 2/datos/RIO MAULE_mensual.xlsx']}

    # Rutas
    # --------------------------------------------------
    path_excel =  dicc_ruta[cuenca][2]
    path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
    ruta_cuencas = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\SIG\Mapoteca DGA\Mapoteca_DGA\02_DGA\Cuencas\Cuencas_DARH_2015_Cuencas.shp'
    # ----------------------------------------------------
    
    # cuenca
    cuenca_shp = geopandas.read_file(ruta_cuencas)    
    cuenca_shp = cuenca_shp[cuenca_shp['COD_CUENCA'] == dicc_ruta[cuenca][1]]
    
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    
    SPrior_Maipo = geopandas.read_file(r'..\SIG\02_CUENCA_Maipo\01_CAPAS_VECTORES\12_MedioAmbiente\Sitios_prioritarios_Maipo_fixed.shp')
    # SPrior_Maipo = geopandas.read_file(r'..\SIG\03_CUENCA_Rapel\01_CAPAS_VECTORES\12_MedioAmbiente\Sitios_prioritarios_Rapel_fixed.shp')
    # SPrior_Maipo = geopandas.read_file(r'..\SIG\04_CUENCA_Mataquito\01_CAPAS_VECTORES\12_MedioAmbiente\Sitios_prioritarios_Mataquito_fixed.shp')
    SPrior_Maipo = geopandas.read_file(r'..\SIG\05_CUENCA_Maule\01_CAPAS_VECTORES\12_MedioAmbiente\Sitios_prioritarios_Maule_fix.shp')

    SPrior_Maipo = geopandas.clip(SPrior_Maipo, cuenca_shp)
    SPrior_Maipo.plot(ax = axes, color = 'yellow', zorder=2)
    
    SNASPE_Maipo = geopandas.read_file(r'..\SIG\02_CUENCA_Maipo\01_CAPAS_VECTORES\12_MedioAmbiente\SNASPE_Maipo.shp')
    SNASPE_Maipo = geopandas.read_file(r'..\SIG\03_CUENCA_Rapel\01_CAPAS_VECTORES\12_MedioAmbiente\SNASPE_Rapel.shp')
    SNASPE_Maipo = geopandas.read_file(r'..\SIG\05_CUENCA_Maule\01_CAPAS_VECTORES\12_MedioAmbiente\SNASPE_Maule_fixed.shp')

    SNASPE_Maipo = geopandas.clip(SNASPE_Maipo, cuenca_shp)
    SNASPE_Maipo.plot(ax = axes, color = 'orange', zorder=3)
    
    
    est_q = pd.read_excel(path_excel, sheet_name = 'info estacion')
        
    if cuenca in ['Mataquito', 'Maule']:
        gdf =  geopandas.GeoDataFrame(est_q, geometry=geopandas.points_from_xy(est_q['Lon'], est_q['Lat']))
        gdf.set_crs(epsg=4326 , inplace=True)
        gdf.to_crs(epsg=32719 , inplace=True)
    else:
        gdf =  geopandas.GeoDataFrame(est_q, geometry=geopandas.points_from_xy(est_q['UTM Este'], est_q['UTM Norte']))
        gdf.set_crs(epsg=32719, inplace=True)
    gdf.to_file(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\QAQC\Q\gdf'+cuenca+'.shp')
    
    basin = geopandas.read_file(path)
            
    gdf.plot(ax = axes, color = 'cyan', zorder=4)
    modules_FAA.plot_catchment_map(basin, bsn_N = dicc_ruta[cuenca][1], ax = axes, basemap = True, linewidth = 5, edgecolor = 'black', label = 'Limite cuenca')
    axes.set_xlabel('Coordenada Este WGS84 19S (m)')
    axes.set_ylabel('Coordenada Norte WGS84 19S (m)')
    legend = [Patch(facecolor='gray', edgecolor=None, label='Cuenca del Río '+cuenca),
              Circle([0], [0], color="cyan", lw=2),
               Patch(facecolor='yellow', edgecolor=None, label='SNASPE'),
              Patch(facecolor='orange', edgecolor=None, label='Sitios Prioritarios')]
    axes.legend(legend,['Cuenca Río '+cuenca,'Estaciones fluviométricas','Sitios Prioritarios','SNASPE'], loc = 'upper left')

    
if __name__ == '__main__':
    main()