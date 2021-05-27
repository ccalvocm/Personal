# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:19:22 2021

@author: ccalvo
"""
import scipy
import pandas as pd
import numpy as np
import os
import lxml
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from plotly.tools import FigureFactory as FF
import geopandas
import matplotlib
import contextily as ctx
import modules_FAA

def pronostico_ARMA(df,meses, orden):
    # fit model
    model = ARIMA(df, order=orden)
    model_fit = model.fit()
    # make prediction
    return model_fit.forecast(meses), model_fit.get_forecast(meses).conf_int(alpha=0.3)

#############
### Maipo ###
#############
    
oroMaipo = [1.3289, 1.3672, 0.8733, 1.1358, 1.648, 1.7283, 1.8243, 1.9714, 2.2218, 2.337, 2.258, 1.912, 1.744, 2.3, 2.945, 2.702, 2.782, 2.984, 3.102, 3.501, 3.324, 2.873, 2.6, 2.29] # 1996-2019
cobreMaipo = [149838, 155843, 163151, 85875, 185005, 183117, 181362, 207848, 231578, 227262, 226017, 229305, 233689, 235490, 217266, 198119, 362707, 415784, 404492, 401715, 307203, 308255, 369542, 334256] # 1996-2019
concentradoMaipo = [174.7, 178.7, 322, 378, 368.3, 366.7, 271.1, 270, 330.6, 296] #2010-2019
catodosMaipo = [42.6, 38.4, 40.7, 37.8, 36.2, 35, 36, 38.3, 39, 39] # 2010-2019
plataMaipo = [73.1878, 61.0702, 45.3868, 24.798, 45.6709, 35.9698, 31.2453, 46.1296, 45.3403, 66.858, 41.052, 43.737, 45.146, 49.735, 44.947, 50.842, 78.724, 54.984, 105.346, 90.081, 69.854, 74.317, 78.735, 66.651] #1996-2019
pumicitaMaipo = [424388, 391702, 826295, 876785, 664217, 631404, 622586, 1018413, 1317088, 1385975, 1201043, 883058, 797966, 605886, 519967, 481503, 502110, 507666, 531661, 547281, 576029, 615305, 669948, 601632]  #1996-2019
demandaCuMaipo = [546, 466, 588, 685, 794, 711, 708, 979, 1097, 789] #2009-2018

#############
### Rapel ###
#############

oroRapel = [0.0852, 0.1855, 0.3123, 0.218, 0.2488, 0.2421, 0.2807, 0.2895, 0.4396, 0.604, 0.591, 0.673, 0.721, 0.807, 0.845, 0.878, 0.905, 0.969, 1.165, 1.131, 1.35, 1.059, 1.069, 1.049] # 1996-2019
cobreRapel = [353277, 361239, 353441, 354932, 366013, 365679, 344676, 351162, 435658, 450927, 429497, 420016, 397208, 421919, 426892, 420220, 440814, 470596, 473286, 487153, 499752, 464548, 465289, 459993] # 1996-2019
cobreTeniente = [344700, 343200, 338600, 346300, 355700, 355600, 334300, 339400, 435600, 437400, 418300, 404700, 381200, 404100, 403600, 400300, 417200, 450400, 455500, 471200, 475300, 464300, 465000, 459700] # 1996-2019
pumicitaRapel = [0, 0, 0, 0, 84206, 69108, 110276, 141835, 141887, 141228, 142555, 158322, 167204, 194073, 185940, 201343, 178851, 152117, 181684, 173489, 191673, 159468, 77470, 0] # 1996-2019
demandaCuRapel = [1668, 1626, 1735, 1659, 1628, 1598, 2013, 2238, 2268, 2344] #2009-2018

#############
### Maule ###
#############

cobreMaule = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 138, 17, 0, 0, 0, 0] # 1996-2019
oroMaule = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.017, 0, 0, 0, 0] # 1996-2019
plataMaule = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.484, 0.477, 0, 0, 0, 0]   #1996-2019
pumicitaMaule = [0, 0, 0, 0, 0, 26428, 6924, 0, 0, 0, 0, 0, 0, 0, 22086, 33879, 26565, 36657, 29668, 27031, 29975, 21028, 15817, 18909]  # 1996-2019

################
### Consumos ###
################

consumoMaipoHidro = [0.01, 0.02, 0.11, 0.06, 0.04, 0.05, 0.01, 0.01, 0.01, 0.01] # 2009-2018
consumoMaipoCon = [0.8, 0.73, 0.55, 0.41, 0.41, 0.35, 0.41, 0.53, 0.64, 0.33] # 2009-2018
consumoRapelCon = [0.66, 0.71, 0.7, 0.81, 0.85, 0.79, 0.81, 0.73, 0.78, 0.78] # 2009-2018
consumoMauleCon = [ 0, 0, 0, 0, 0, 0.79, 0.81, 0, 0, 0] # 1996-2018

consumos = pd.DataFrame([consumoMaipoHidro,consumoMaipoCon,consumoRapelCon,consumoMauleCon])
consumos = consumos.transpose()
consumos.index = range(2009,2019)
consumos.columns = ['MaipoHidro','MaipoCon','RapelCon','MauleCon']

for col in consumos:
    z = np.polyfit(range(2009,2019), consumos.loc[2009:2018,col], 1)
    f = np.poly1d(z)
    for i in range(3):
        consumos.loc[2019+i,col] = f(2019+i)

produccion_minera = pd.DataFrame([oroMaipo,cobreMaipo,concentradoMaipo,catodosMaipo,plataMaipo,pumicitaMaipo,oroRapel,cobreRapel,cobreTeniente,pumicitaRapel,
cobreMaule,  oroMaule,    plataMaule,   pumicitaMaule ])
produccion_minera = produccion_minera.transpose()

produccion_minera.columns = ['AuMaipo', 'CuMaipo', 'CuConMaipo', 'CuCatMaipo', 'AgMaipo', 'SiO2Maipo', 'Aurapel', 'CuRapel', 'CuTeniente', 'SiO2Rapel',
                             'CuMaule','AuMaule','AgMaule','SiO2Maule']

aux = produccion_minera['CuConMaipo']
aux[-10:] = produccion_minera['CuConMaipo'][0:10]
aux[0:10] = np.nan
produccion_minera['CuConMaipo'] = aux


aux = produccion_minera['CuCatMaipo']
aux[-10:] = produccion_minera['CuCatMaipo'][0:10]
aux[0:10] = np.nan
produccion_minera['CuCatMaipo'] = aux

produccion_minera_lcl = produccion_minera.copy()
produccion_minera_ucl = produccion_minera.copy()

agnos = len(produccion_minera.index)

for col in produccion_minera.columns:
    prod_min_2021,prod_min_2021_ci = pronostico_ARMA(produccion_minera[col], 2, (1,2,13)) #1,1,18
    for i in range(2):
        produccion_minera.loc[agnos+i,col] = prod_min_2021.iloc[i]
        produccion_minera_lcl.loc[agnos+i,col] = prod_min_2021_ci.iloc[i,0]
        produccion_minera_ucl.loc[agnos+i,col] = prod_min_2021_ci.iloc[i,1]

produccion_minera.index = range(1996,2022)
produccion_minera[produccion_minera < 0] = 0
produccion_minera_lcl[produccion_minera_lcl < 0] = 0
produccion_minera_ucl[produccion_minera_ucl < 0] = 0

produccion_minera['CuRapel']-produccion_minera['CuTeniente']
produccion_minera.plot()
# produccion_minera.to_csv(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\MIN\proyeccion_minera.csv')
produccion_minera_lcl.plot()
produccion_minera_ucl.plot()


# =====================================
#   Demanda total
# =====================================

# ----Iniciar los df

demanda = pd.DataFrame([demandaCuMaipo, demandaCuRapel ])
demanda = demanda.transpose()
demanda.columns = ['CuMaipo', 'CuRapel']
demanda.index = demanda.index = range(2009,2019)

# ----------------I.C.
demanda_lcl = demanda.copy()
demandaa_ucl = demanda.copy()

agnos_pronostico = 3
agnos = 2019

for col in demanda.columns:
    demanda_2021, demanda_2021_ci = pronostico_ARMA(demanda[col], agnos_pronostico, (1,2,10)) #1,1,18
    for i in range(agnos_pronostico):
        demanda.loc[agnos+i,col] = demanda_2021.iloc[i]
        demanda_lcl.loc[agnos+i,col] = demanda_2021_ci.iloc[i,0]
        demandaa_ucl.loc[agnos+i,col] = demanda_2021_ci.iloc[i,1]
        
def plots():
    
    # rutas
    # Paths y carga de datos genericos
    path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Cuencas/Cuencas_DARH_2015.shp'
    basin = geopandas.read_file(path)
    path = '../Etapa 1 y 2/GIS/Cuencas_DARH/Subcuencas/SubCuencas_DARH_2015.shp'
    subbasin = geopandas.read_file(path)
    ruta_Git = r'C:\Users\ccalvo\Documents\GitHub'
    ruta_mensual_Cu = r'..\Etapa 1 y 2\Demanda\MIN\Fuentes informacion\Dda_mensual_Cu.csv'
    ruta_mensual_AuAg = r'..\Etapa 1 y 2\Demanda\MIN\Fuentes informacion\Dda_mensual_AuAg.csv'
    ruta_mensual_Cu_Rapel = r'..\Etapa 1 y 2\Demanda\MIN\Fuentes informacion\Dda_mensual_Rapel.csv'
    ruta_mensual_Au_Maule = r'..\Etapa 1 y 2\Demanda\MIN\Fuentes informacion\Dda_mensual_Maule.csv'
    
    # plots de faenas
    plt.close("all")
    # plot centros acuícolas
    fig = plt.figure(figsize=(11, 8.5))
    axes = fig.add_subplot(111)
    
    # faenas Maipo
    fM_RM = geopandas.read_file('..//SIG//02_CUENCA_Maipo//01_CAPAS_VECTORES//04_AguaSuperficial//01_Demanda//Faenas_mineras//Faenas_de_Chile_RMaipo.shp')
    fM_RM.plot(ax = axes, color = 'k')
     
    # Faenas Rapel
    fM_RM = geopandas.read_file('..//SIG//03_CUENCA_Rapel//01_CAPAS_VECTORES//04_AguaSuperficial//01_Demanda//Faenas_mineras//Faenas_de_Chile_RRapel.shp')
    fM_RM.plot(ax = axes, color = 'k')
    subbasin.style.set_properties(**{'text-align': 'left'})
    
    # Faenas Mataquito
    fM_RM = geopandas.read_file('..//SIG//04_CUENCA_Mataquito//01_CAPAS_VECTORES//04_AguaSuperficial//01_Demanda//Faenas_mineras//Faenas_de_Chile_RMataquito.shp')
    fM_RM.plot(ax = axes, color = 'k')
    subbasin.style.set_properties(**{'text-align': 'left'})
    
    # Faenas Maule
    fM_RM = geopandas.read_file('..//SIG//05_CUENCA_Maule//01_CAPAS_VECTORES//04_AguaSuperficial//01_Demanda//Faenas_mineras//Faenas_de_Chile_RMaule.shp')
    fM_RM.plot(ax = axes, color = 'k')
    subbasin.style.set_properties(**{'text-align': 'left'})
    
    plt.legend(['Faenas mineras'])
    
    cuencas = geopandas.read_file('..//Etapa 1 y 2//GIS//Cuencas_DARH//Cuencas//Cuencas_DARH_2015.shp')
    cuencas = cuencas.loc[(cuencas['COD_CUENCA'] == '0703') | (cuencas['COD_CUENCA'] == '0701') | (cuencas['COD_CUENCA'] == '0600') | (cuencas['COD_CUENCA'] == '1300')]
    cuencas.plot(ax = axes, alpha=0.4)
    ctx.add_basemap(ax = axes, crs= cuencas.crs.to_string(),
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

    # gráficos de demanda por cuencas
    
    # Maipo
    dda_min_Maipo = pd.read_excel(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\MIN\00 DDA MIN ACT_VI_VII_XIII.xlsm', sheet_name = 'Resumen XIII REGIÓN  ')
    columnas = dda_min_Maipo['Minera'].tolist()
    fechas = dda_min_Maipo.columns
    dda_min_Maipo = dda_min_Maipo.transpose()
    dda_min_Maipo.index = fechas
    dda_min_Maipo.columns = columnas
    dda_min_Maipo = dda_min_Maipo.iloc[1:]
    
    dda_mensual = pd.read_csv(ruta_mensual_Cu, parse_dates = True, index_col = 0, dayfirst=True)
    dda_mensual = dda_mensual.loc[dda_mensual.index < '01-01-2021']
    fechas = pd.date_range(freq='MS', start = '1996-01-01', end = '2021-12-31')
    dda_min_Maipo_mon = pd.DataFrame([], index = fechas, columns = dda_min_Maipo.columns)
    dda_min_Maipo_yr = dda_mensual.resample('YS').sum()
    dda_min_Maipo_yr.index = dda_min_Maipo_yr.index.year
    dda_min_Maipo_yr = dda_min_Maipo_yr.loc[dda_mensual.index.year]
    dda_min_Maipo_yr.index = dda_mensual.index
    
    dda_prorrateada = 365.*dda_mensual.div(dda_min_Maipo_yr, axis = 'index')
    dda_prorrateada = dda_prorrateada.div(dda_prorrateada.index.daysinmonth, axis = 'index')
    
    for ind, col in dda_min_Maipo_mon.iterrows():
        yr = ind.year
        prod_anual = dda_min_Maipo.loc[yr].values
        if yr < 2000:
            index_aux = pd.to_datetime('2000-'+str(ind.month)+'-01')
            dda_min_Maipo_mon.loc[ind] = prod_anual*dda_prorrateada.loc[index_aux].values
        elif yr > 2020:
            index_aux = pd.to_datetime('2020-'+str(ind.month)+'-01')
            dda_min_Maipo_mon.loc[ind] = prod_anual*dda_prorrateada.loc[index_aux].values
        else:
            dda_min_Maipo_mon.loc[ind] = prod_anual*dda_prorrateada.loc[ind].values
        
    # Rapel
    dda_min_Rapel = pd.read_excel(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\MIN\00 DDA MIN ACT_VI_VII_XIII.xlsm', sheet_name = 'Resumen VI REGIÓN')
    columnas = dda_min_Rapel['Minera'].tolist()
    fechas = dda_min_Rapel.columns
    dda_min_Rapel = dda_min_Rapel.transpose()
    dda_min_Rapel.index = fechas
    dda_min_Rapel.columns = columnas
    dda_min_Rapel = dda_min_Rapel.iloc[1:]
    
    dda_mensual = pd.read_csv(ruta_mensual_Cu_Rapel, parse_dates = True, index_col = 0, dayfirst=True)
    dda_mensual = dda_mensual.loc[dda_mensual.index < '01-01-2021']
    fechas = pd.date_range(freq='MS', start = '1996-01-01', end = '2021-12-31')
    dda_min_Rapel_mon = pd.DataFrame([], index = fechas, columns = dda_min_Rapel.columns)
    dda_min_Rapel_yr = dda_mensual.resample('YS').sum()
    dda_min_Rapel_yr.index = dda_min_Rapel_yr.index.year
    dda_min_Rapel_yr = dda_min_Rapel_yr.loc[dda_mensual.index.year]
    dda_min_Rapel_yr.index = dda_mensual.index
    
    dda_prorrateada = 365.*dda_mensual.div(dda_min_Rapel_yr, axis = 'index')
    dda_prorrateada = dda_prorrateada.div(dda_prorrateada.index.daysinmonth, axis = 'index')
    
    for ind, col in dda_min_Rapel_mon.iterrows():
        yr = ind.year
        prod_anual = dda_min_Rapel.loc[yr].values
        if yr < 2000:
            #print(dda_prorrateada.loc[index_aux].values)
            index_aux = pd.to_datetime('2000-'+str(ind.month)+'-01')
            dda_min_Rapel_mon.loc[ind] = prod_anual*dda_prorrateada.loc[index_aux].values
        elif yr > 2020:
            index_aux = pd.to_datetime('2020-'+str(ind.month)+'-01')
            dda_min_Rapel_mon.loc[ind] = prod_anual[0]*dda_prorrateada.loc[index_aux].values[0]
        else:
            dda_min_Rapel_mon.loc[ind] = prod_anual*dda_prorrateada.loc[ind].values[0]
    
    # Maule
    dda_min_Maule = pd.read_excel(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Demanda\MIN\00 DDA MIN ACT_VI_VII_XIII.xlsm', sheet_name = 'Resumen VII REGIÓN ')
    columnas = dda_min_Maule['Minera'].tolist()
    fechas = dda_min_Maule.columns
    dda_min_Maule = dda_min_Maule.transpose()
    dda_min_Maule.index = fechas
    dda_min_Maule.columns = columnas
    dda_min_Maule = dda_min_Maule.iloc[1:]
    
    #dda_min_Maule.plot()
    plt.legend(title='Minera', bbox_to_anchor=(0.5, -0.5), loc='lower center')
    plt.ylabel('Q ($l$/s)')
    
    #Dda prorrateada
    
    dda_mensual = pd.read_csv(ruta_mensual_Au_Maule, parse_dates = True, index_col = 0, dayfirst=False)
    dda_mensual = dda_mensual.loc[dda_mensual.index < '01-01-2021']
    fechas = pd.date_range(freq='MS', start = '1996-01-01', end = '2021-12-31')
    dda_min_Maule_mon = pd.DataFrame([], index = fechas, columns = dda_min_Maule.columns)
    dda_min_Maule_yr = dda_mensual.resample('YS').sum()
    dda_min_Maule_yr.index = dda_min_Maule_yr.index.year
    dda_min_Maule_yr = dda_min_Maule_yr.loc[dda_mensual.index.year]
    dda_min_Maule_yr.index = dda_mensual.index
    
    dda_prorrateada = 365.*dda_mensual.div(dda_min_Maule_yr, axis = 'index')
    dda_prorrateada = dda_prorrateada.div(dda_prorrateada.index.daysinmonth, axis = 'index')
    
    #Calcular
    
    fechas = pd.date_range(freq='MS', start = '1996-01-01', end = '2021-12-31')
    dda_min_Maule_mon = pd.DataFrame([], index = fechas, columns = dda_min_Maule.columns)
    
    for ind, col in dda_min_Maule_mon.iterrows():
        yr = ind.year
        prod_anual = dda_min_Maule.loc[yr].values
        if yr < 2003:
            #print(dda_prorrateada.loc[index_aux].values)
            index_aux = pd.to_datetime('2003-'+str(ind.month)+'-01')
            dda_min_Maule_mon.loc[ind] = prod_anual*dda_prorrateada.loc[index_aux].values
        elif yr > 2020:
            index_aux = pd.to_datetime('2020-'+str(ind.month)+'-01')
            dda_min_Maule_mon.loc[ind] = prod_anual[0]*dda_prorrateada.loc[index_aux].values[0]
        else:
            dda_min_Maule_mon.loc[ind] = prod_anual*dda_prorrateada.loc[ind].values[0]
    
    # Plotear las demandas por cuenca
    dicc_dda = {'Río Maipo' : dda_min_Maipo_mon['Total'], 'Río Rapel' : dda_min_Rapel_mon['Total'], 'Río Maule' : dda_min_Maule_mon['Total']}
    cuencas = ['Río Maipo','Río Rapel','Río Maule']
    fig, axes = plt.subplots(2,2,figsize=(10, 22))
    axes = axes.reshape(-1)
    for i in range(0,3):
        # plt.figure()
        dicc_dda[cuencas[i]].plot(ax = axes[i])
        dicc_dda[cuencas[i]].to_csv('.\outputs\Demanda_minería_mon_cuenca '+cuencas[i]+'.csv')
        axes[i].set_title('Demanda de uso minero en la cuenca del '+cuencas[i])
        axes[i].set_ylabel('Demanda media mensual ($l/s$)')
        axes[i].grid()
