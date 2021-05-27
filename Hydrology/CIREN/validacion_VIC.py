# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:47:28 2021

@author: Carlos
"""

# =================
#     Liberías
# =================

import pandas as pd
import os
import matplotlib.pyplot as plt
from hydroeval import evaluator, nse
import numpy as np

# =================
#     Rutas
# =================   

ruta_Maipo_VIC = r'..\QAQC\VIC'
ruta_VIC = r'.\VIC'
# =================
#     Rutas
# =================
ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos'
os.chdir(ruta_OD)

# =================
#     Funciones
# =================

def NSE(nse, sim_flow, obs_flow, axis=1):
    serie_sim = sim_flow.values.ravel()
    serie_obs = obs_flow.values.ravel()
    my_nse = evaluator(nse, serie_sim, serie_obs, axis=1)
    return my_nse

def formato(df):
    df.index = pd.to_datetime(df['Year'].astype(str)  +'-'+ df['month'].astype(str), format='%Y-%m')
    for i in ['month','Year']:
        del df[i]
    return df


def mainMaipo():
        
    ruta_Maipo_BNA = r'\RIO MAIPO_Q_mensual.xlsx'
    
    q_Maipo = pd.read_excel(ruta_OD+ruta_Maipo_BNA, sheet_name = 'data', parse_dates = True, index_col = 0)
    q_Maipo_flags = pd.read_excel(ruta_OD+ruta_Maipo_BNA, sheet_name = 'info data', parse_dates = True, index_col = 0)
    q_Maipo_VIC_FB = pd.read_csv('..\\QAQC\\VIC\\FB _Salidas_monthly.csv')
    q_Maipo_VIC_FB.index = pd.to_datetime(q_Maipo_VIC_FB['Year'].astype(str)  +'-'+ q_Maipo_VIC_FB['month'].astype(str), format='%Y-%m')
    q_Maipo_VIC_runoff = pd.read_csv('..\\QAQC\\VIC\\runoff _Salidas_monthly.csv')
    q_Maipo_VIC_runoff.index = pd.to_datetime(q_Maipo_VIC_runoff['Year'].astype(str)  +'-'+ q_Maipo_VIC_runoff['month'].astype(str), format='%Y-%m')
    q_Maipo_VIC_Q = pd.read_csv('..\\QAQC\\VIC\\Q _Salidas_monthly.csv')
    q_Maipo_VIC_Q.index = pd.to_datetime(q_Maipo_VIC_Q['Year'].astype(str)  +'-'+ q_Maipo_VIC_Q['month'].astype(str), format='%Y-%m')
    
    for i in ['Ano','Mes']:
        del q_Maipo[i]
        del q_Maipo_flags[i]
        
    for i in ['month','Year']:
        del q_Maipo_VIC_FB[i]
        del q_Maipo_VIC_runoff[i]
        del q_Maipo_VIC_Q[i]
    
    q_Maipo_Manzano_VIC = q_Maipo_VIC_FB + q_Maipo_VIC_runoff
    
    q_Maipo_fr = q_Maipo[q_Maipo_flags.div(q_Maipo_flags.index.daysinmonth, axis = 'index') >= 0.8]
    q_Maipo_fr = q_Maipo_fr.loc[(q_Maipo_fr.index > '1980-01-01') & (q_Maipo_fr.index < '2016-01-01')]
    
    
    # =================
    #     Validacion
    # =================
    
    #runoff y FB
    fig, ax = plt.subplots(1)
    q_Maipo_fr['05710001-K'].plot(ax = ax)
    q_Maipo_Manzano_VIC.loc[q_Maipo_fr['05710001-K'].index].plot(ax = ax)
    plt.title('')
    plt.ylabel('Caudal ($m^3/s$)')
    plt.legend(['Maipo en El Manzano BNA','VIC'])
    props = dict(boxstyle='round', facecolor='wheat', alpha=.7)
    ax.text(0.05, 0.05, 'N-SE = '+str(np.round(NSE(nse, q_Maipo_Manzano_VIC.loc[q_Maipo_fr['05710001-K'].index], q_Maipo_fr['05710001-K'], axis=1),2)), transform=ax.transAxes, fontsize=10,
          verticalalignment='bottom', bbox=props)


def validador(cuenca = 'Maule', ruta_cuenca = 'RIO MAULE_mensual.xlsx', estacion = '07321002-K', 
              estacion_alias = 'Rio Maule en Armerillo'):
    ruta_fb = ruta_VIC + '\\'+cuenca+'\\FB _Salidas_monthly.csv'
    ruta_runoff = ruta_VIC + '\\'+cuenca+'\\runoff _Salidas_monthly.csv'
    
    fb = formato(pd.read_csv(ruta_fb))
    runoff = formato(pd.read_csv(ruta_runoff))
    
    q = pd.read_excel(ruta_cuenca,index_col = 0, parse_dates = True, sheet_name = 'data')
    q_f =  pd.read_excel(ruta_cuenca,index_col = 0, parse_dates = True, sheet_name = 'info data') 
    
    q_VIC = fb+runoff
    q_VIC.to_csv('q_'+cuenca+'_VIC_mon.csv')
    
    # ---------estacion
      
    q_estacion = q[estacion]
    q_estacion[q_f[estacion] < 20] = np.nan
    q_estacion = q_estacion.loc[(q_estacion.index > '1981-04-01') & (q_estacion.index < '2015-04-01')]
        
    q_estacion_VIC = q_VIC['Salida_'+estacion[1:].split('-')[0]]
        
    fig, ax = plt.subplots(1)
    q_estacion.plot(ax = ax)
    q_estacion_VIC.loc[q_estacion.index].plot(ax = ax)
    plt.title('')
    plt.ylabel('Caudal ($m^3/s$)')
    plt.legend([estacion_alias,'VIC'])   
    props = dict(boxstyle='round', facecolor='wheat', alpha=.7)    
    ax.text(0.05, 0.05, 'N-SE = '+str(np.round(NSE(nse, q_estacion_VIC.loc[q_estacion.index], q_estacion, axis=1),2)), transform=ax.transAxes, fontsize=10,
    verticalalignment='bottom', bbox=props)
    

def mainMataquito():
    ruta_VIC = r'.\VIC'
    ruta_fb_Mataquito = ruta_VIC + r'\Mataquito\FB _Salidas_monthly.csv'
    ruta_runoff_Mataquito = ruta_VIC + r'\Mataquito\runoff _Salidas_monthly.csv'
    
    fb_Mataquito = formato(pd.read_csv(ruta_fb_Mataquito))
    runoff_mataquito = formato(pd.read_csv(ruta_runoff_Mataquito))
    
    q_Mataquito = pd.read_excel('RIO MATAQUITO_mensual.xlsx',index_col = 0, parse_dates = True)
    q_Mataquito_f =  pd.read_excel('RIO MATAQUITO_mensual.xlsx',index_col = 0, parse_dates = True, sheet_name = 'info data') 

    q_teno_quenes = q_Mataquito['07102001-0']
    q_teno_quenes_f = q_Mataquito_f
    
    for i in ['Ano','Mes']:
        del q_teno_quenes_f[i]

    
    # ---------guardar VIC
    
    q_Mataquito_VIC = fb_Mataquito+runoff_mataquito
    q_Mataquito_VIC.to_csv(r'q_Mataquito_VIC_mon.csv')

    # -------Teno en los Queñes
    
    q_teno_quenes = q_teno_quenes[q_teno_quenes_f['07102001-0'] > 20]
    q_teno_quenes_VIC = fb_Mataquito['Salida_7102001']+runoff_mataquito['Salida_7102001']
    
    fig, ax = plt.subplots(1)
    q_teno_quenes.plot(ax = ax)
    q_teno_quenes_VIC.plot(ax = ax)
    plt.title('')
    plt.ylabel('Caudal ($m^3/s$)')
    plt.legend(['Río Teno en Los Queñes BNA','VIC'])       
        
    # -------Claro en los Queñes
    
    q_claro_quenes = q_Mataquito[q_Mataquito_f['07103001-6'] >= 20]['07103001-6']
    q_claro_quenes_VIC = fb_Mataquito['Salida_7103001']+runoff_mataquito['Salida_7103001']
    
    fig, ax = plt.subplots(1)
    q_claro_quenes.plot(ax = ax)
    q_claro_quenes_VIC.plot(ax = ax)
    plt.title('')
    plt.ylabel('Caudal ($m^3/s$)')
    plt.legend(['Río Claro en Los Queñes BNA','VIC'])

    


    

    

    
    