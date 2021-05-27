# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:36:17 2021

@author: Carlos
"""

# ===============
#   librerías
# ===============

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas
import rasterio
from rasterio.plot import show

# ===============
#   rutas
# ===============

ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos'
ruta_CF = ruta_OD+r'\Datos CF\Datos_BNA_EstacionesDGA\BNAT_CaudalDiario.txt'
ruta_CC = ruta_OD+r'\RIO MAIPO_Q_mensual.xlsx'
ruta_CC_Rapel = ruta_OD+r'\RIO RAPEL_Q_mensual.xlsx'
ruta_CC_Mataquito = ruta_OD+r'\RIO MATAQUITO_mensual.xlsx'
ruta_CC_Maule = ruta_OD+r'\RIO MAULE_mensual.xlsx'


def ordenarDGA(df,df_DGA):
    output = pd.DataFrame([], index = pd.date_range(start='1900-01-01', end='2020-12-31', closed=None))

    for ind, est in enumerate(df):
        q_est = df_DGA[df_DGA.iloc[:,0] == df[ind]]
        nombre = q_est.iloc[0,1]
        fechas = pd.to_datetime(q_est.iloc[:,2], dayfirst = True)
        caudal = q_est.iloc[:,3]
        flags = q_est.iloc[:,4]
        
        q_est_df = pd.DataFrame(caudal.values, index = fechas, columns = [est])
            
        output.loc[output.index,est] = q_est_df[est]
    
    return output
    
def calidadpp(df):
    df_calidad = df.copy()
    df_calidad[:] = 1
    df_calidad[df.isnull()] = 0
    df_mon = df.copy().resample('MS').sum()
    df_mon[df_calidad.resample('MS').sum() <= 20] = np.nan
    return df_mon

def calidadq(df):
    df_calidad = df.copy()
    df_calidad[:] = 1
    df_calidad[df.isnull()] = 0
    df_mon = df.copy().resample('MS').mean()
    df_mon[df_calidad.resample('MS').sum() <= 20] = np.nan
    return df_mon
    
def QAQC_q():
    q_CF = pd.read_csv(ruta_CF, sep = ';')
    q_CF['region'] = [x[:3] for x in q_CF.iloc[:,0]]
    est_CF_Maipo = q_CF[q_CF['region'] == '057'].iloc[:,0].drop_duplicates()
    est_CF_Rapel = q_CF[q_CF['region'] == '060'].iloc[:,0].drop_duplicates()
    q_CF['region'] = [x[:3] for x in q_CF.iloc[:,0]]
    est_CF_Mataquito = q_CF[q_CF['region'] == '071'].iloc[:,0].drop_duplicates()
    est_CF_Maule = q_CF[q_CF['region'] == '073'].iloc[:,0].drop_duplicates()
    
    ######################
    ##       Maipo      ##
    ######################
    
    q_CC_Maipo = pd.read_excel(ruta_CC, parse_dates = True, index_col = 0, sheet_name = 'data')
    q_CC_Rapel = pd.read_excel(ruta_CC_Rapel, parse_dates = True, index_col = 0, sheet_name = 'data')
    q_CC_Mataquito = pd.read_excel(ruta_CC_Mataquito, parse_dates = True, index_col = 0, sheet_name = 'data')
    q_CC_Maule = pd.read_excel(ruta_CC_Maule, parse_dates = True, index_col = 0, sheet_name = 'data')
    
    est_CC_maipo = q_CC_maipo.columns[2:]
    est_CC_Rapel = q_CC_Rapel.columns[2:]
    est_CC_Mataquito = q_CC_Mataquito.columns[2:]
    est_CC_Maule = q_CC_Maule.columns[2:]
    
    # Maipo
    est_diff_Maipo = [x for x in est_CF_maipo if x not in est_CC_maipo]
    q_est_diff_Maipo = ordenarDGA(est_diff_Maipo, q_CF)
    q_est_diff_Maipo = q_est_diff_Maipo.loc[q_est_diff_Maipo.index >= '1979-01-01']
    q_est_diff_Maipo_mon = q_est_diff_Maipo.resample('MS').mean()
    
    q_est_diff_Maipo_calidad = q_est_diff_Maipo.copy()
    q_est_diff_Maipo_calidad[q_est_diff_Maipo_calidad > 0] = 1
    q_est_diff_Maipo_calidad[q_est_diff_Maipo_calidad.isna()] = 0
    q_est_diff_Maipo_calidad = q_est_diff_Maipo_calidad.resample('MS').sum().div( q_est_diff_Maipo_calidad.resample('MS').sum().index.daysinmonth, axis = 'index')
    q_est_diff_Maipo_mon_flags = q_est_diff_Maipo_mon[q_est_diff_Maipo_calidad > 0.8]
    
    # writer = pd.ExcelWriter(ruta_OD+r'\q_est_diff_Maipo'+'.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    q_est_diff_Maipo_mon_flags.to_excel(writer, sheet_name= 'data')
    q_est_diff_Maipo_calidad.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    # writer.save()
    
    # Comparar con Carlos Flores
    
    est_nodiff_Maipo = [x for x in est_CF_Maipo if x in q_CC_Maipo]
    q_est_nodiff_Maipo = ordenarDGA(est_nodiff_Maipo, q_CF)
    q_est_nodiff_Maipo = q_est_nodiff_Maipo.loc[q_est_nodiff_Maipo.index >= '1979-01-01']
    
    q_est_nodiff_Maipo_calidad = Calidad(q_est_nodiff_Maipo)
    
    q_est_nodiff_Maipo_mon = q_est_nodiff_Maipo.copy().resample('MS').mean()
    q_est_nodiff_Maipo_mon = q_est_nodiff_Maipo_mon[q_est_nodiff_Maipo_calidad > 0.8]
    q_est_nodiff_Maipo_mon.plot()
    
    writer = pd.ExcelWriter(ruta_OD+r'\q_est_nodiff_Maipo_mon'+'.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    q_est_nodiff_Maipo_mon.to_excel(writer, sheet_name= 'data')
    q_est_nodiff_Maipo_calidad.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    ruta = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\RIO MAIPO_Q_mensual.xlsx'
    q_Maipo_CC = pd.read_excel(ruta, parse_dates = True, index_col = 0, sheet_name = 'data' )
    q_Maipo_CC = q_Maipo_CC[q_Maipo_CC.columns[2:]]
    calidad_CC_Maipo = pd.read_excel(ruta, parse_dates = True, index_col = 0, sheet_name = 'info data')
    calidad_CC_Maipo = calidad_CC_Maipo.div(calidad_CC_Maipo.index.daysinmonth, axis = 'index')
    q_Maipo_CC = q_Maipo_CC[calidad_CC_Maipo > 0.8]
    
    fig, ax = plt.subplots(6,5)
    ax = ax.reshape(-1)
    
    for ind, col in enumerate(q_Maipo_CC.columns):
        q_est_nodiff_Maipo_mon[col].plot(ax = ax[ind], linewidth = 1.1, color = 'r')
        q_Maipo_CC[col].plot(ax = ax[ind], color = 'b')
        ax[ind].set_title(col)
        ax[ind].set_ylabel('Q ($m^3/s$)')
        
    ax[ind].legend(['Carlos Flores','BNA'],bbox_to_anchor=(.75, .75), loc='upper left') 
    
    
    
    #############################
    ##          Rapel          ##
    #############################
    
    est_diff_Rapel = [x for x in est_CF_Rapel if x not in q_CC_Rapel]
    q_est_diff_Rapel = ordenarDGA(est_diff_Rapel, q_CF)
    q_est_diff_Rapel = q_est_diff_Rapel.loc[q_est_diff_Rapel.index >= '1979-01-01']
    q_est_diff_Rapel_mon = q_est_diff_Rapel.resample('MS').mean()
    
    q_est_diff_Maipo_calidad = q_est_diff_Maipo.copy()
    q_est_diff_Maipo_calidad[q_est_diff_Maipo_calidad > 0] = 1
    q_est_diff_Maipo_calidad[q_est_diff_Maipo_calidad.isna()] = 0
    q_est_diff_Maipo_calidad = q_est_diff_Maipo_calidad.resample('MS').sum().div( q_est_diff_Maipo_calidad.resample('MS').sum().index.daysinmonth, axis = 'index')
    q_est_diff_Maipo_mon_flags = q_est_diff_Maipo_mon[q_est_diff_Maipo_calidad > 0.8]
    
    # writer = pd.ExcelWriter(ruta_OD+r'\q_est_diff_Maipo'+'.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    q_est_diff_Maipo_mon_flags.to_excel(writer, sheet_name= 'data')
    q_est_diff_Maipo_calidad.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    # writer.save()
    
    # Comparar con Carlos Flores
    
    est_nodiff_Rapel = [x for x in est_CF_Rapel if x in q_CC_Rapel]
    q_est_nodiff_Rapel = ordenarDGA(est_nodiff_Rapel, q_CF)
    q_est_nodiff_Rapel = q_est_nodiff_Rapel.loc[q_est_nodiff_Rapel.index >= '1979-01-01']
    
    q_est_nodiff_Rapel_calidad = calidadq(q_est_nodiff_Rapel)
    
    q_est_nodiff_Rapel_mon = q_est_nodiff_Rapel.copy().resample('MS').mean()
    q_est_nodiff_Rapel_mon = q_est_nodiff_Rapel_mon[q_est_nodiff_Rapel_calidad > 0.8]
    q_est_nodiff_Rapel_mon.plot()
    
    writer = pd.ExcelWriter(ruta_OD+r'\q_est_nodiff_Rapel_mon'+'.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    q_est_nodiff_Rapel_mon.to_excel(writer, sheet_name= 'data')
    q_est_nodiff_Rapel_calidad.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    ruta = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\RIO RAPEL_Q_mensual.xlsx'
    q_Rapel_CC = pd.read_excel(ruta, parse_dates = True, index_col = 0, sheet_name = 'data' )
    q_Rapel_CC = q_Rapel_CC[q_Rapel_CC.columns[2:]]
    calidad_CC_Rapel = pd.read_excel(ruta, parse_dates = True, index_col = 0, sheet_name = 'info data')
    calidad_CC_Rapel = calidad_CC_Rapel.div(calidad_CC_Rapel.index.daysinmonth, axis = 'index')
    q_Rapel_CC = q_Rapel_CC[calidad_CC_Rapel > 0.8]
    
    fig, ax = plt.subplots(6,4)
    ax = ax.reshape(-1)
    
    for ind, col in enumerate(q_Rapel_CC.columns):
        q_est_nodiff_Rapel_mon[col].plot(ax = ax[ind], linewidth = 3, color = 'r')
        q_Rapel_CC[col].plot(ax = ax[ind], color = 'b')
        ax[ind].set_title(col)
        ax[ind].set_ylabel('Q ($m^3/s$)')
        
    ax[ind].legend(['Carlos Flores','BNA'],bbox_to_anchor=(.75, .75), loc='upper left')    
    
    
    
    #############################
    ##        Mataquito        ##
    #############################
    
    est_diff_Mataquito = [x for x in est_CF_Mataquito if x not in q_CC_Mataquito]
    q_est_diff_Mataquito = ordenarDGA(est_diff_Mataquito, q_CF)
    q_est_diff_Mataquito = q_est_diff_Mataquito.loc[q_est_diff_Mataquito.index >= '1979-01-01']
    
    est_nodiff_Mataquito = [x for x in est_CF_Mataquito if x in q_CC_Mataquito]
    q_est_nodiff_Mataquito = ordenarDGA(est_nodiff_Mataquito, q_CF)
    q_est_nodiff_Mataquito = q_est_nodiff_Mataquito.loc[q_est_nodiff_Mataquito.index >= '1979-01-01']
    
    # Comparar con Carlos Flores
    q_est_nodiff_Mataquito_calidad = calidadq(q_est_nodiff_Mataquito)
    
    q_est_nodiff_Mataquito_mon = q_est_nodiff_Mataquito.copy().resample('MS').mean()
    q_est_nodiff_Mataquito_mon = q_est_nodiff_Mataquito_mon[q_est_nodiff_Mataquito_calidad > 0.8]
    q_est_nodiff_Mataquito_mon.plot()
    
    writer = pd.ExcelWriter(ruta_OD+r'\q_est_nodiff_Mataquito_mon'+'.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    q_est_nodiff_Mataquito_mon.to_excel(writer, sheet_name= 'data')
    q_est_nodiff_Mataquito_calidad.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    ruta = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\RIO MATAQUITO_mensual.xlsx'
    q_Mataquito_CC = pd.read_excel(ruta, parse_dates = True, index_col = 0, sheet_name = 'data' )
    q_Mataquito_CC = q_Mataquito_CC[q_Mataquito_CC.columns[2:]]
    calidad_CC_Mataquito = pd.read_excel(ruta, parse_dates = True, index_col = 0, sheet_name = 'info data')
    calidad_CC_Mataquito = calidad_CC_Mataquito.div(calidad_CC_Mataquito.index.daysinmonth, axis = 'index')
    q_Mataquito_CC = q_Mataquito_CC[calidad_CC_Mataquito > 0.8]
    
    
    fig, ax = plt.subplots(5,3)
    ax = ax.reshape(-1)
    
    for ind, col in enumerate(q_Mataquito_CC.columns):
        q_est_nodiff_Mataquito_mon[col].plot(ax = ax[ind], linewidth = 3, color = 'r')
        q_Mataquito_CC[col].plot(ax = ax[ind], color = 'b')
        ax[ind].set_title(col)
        ax[ind].set_ylabel('Q ($m^3/s$)')
        
    ax[ind].legend(['Carlos Flores','BNA'],bbox_to_anchor=(.75, .75), loc='upper left')    
    
    # ==============================
    #             Maule
    # ==============================
    
    
    est_diff_Maule = [x for x in est_CF_Maule if x not in q_CC_Maule]
    q_est_diff_Maule = ordenarDGA(est_diff_Maule, q_CF)
    q_est_diff_Maule = q_est_diff_Maule.loc[q_est_diff_Maule.index >= '1979-01-01']
    q_est_diff_Maule_mon = q_est_diff_Maule.resample('MS').mean()
    
    q_est_diff_Maule_mon.plot()
    
    q_est_diff_Maipo_calidad = q_est_diff_Maipo.copy()
    q_est_diff_Maipo_calidad[q_est_diff_Maipo_calidad > 0] = 1
    q_est_diff_Maipo_calidad[q_est_diff_Maipo_calidad.isna()] = 0
    q_est_diff_Maipo_calidad = q_est_diff_Maipo_calidad.resample('MS').sum().div( q_est_diff_Maipo_calidad.resample('MS').sum().index.daysinmonth, axis = 'index')
    q_est_diff_Maipo_mon_flags = q_est_diff_Maipo_mon[q_est_diff_Maipo_calidad > 0.8]
    
    # writer = pd.ExcelWriter(ruta_OD+r'\q_est_diff_Maipo'+'.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    q_est_diff_Maipo_mon_flags.to_excel(writer, sheet_name= 'data')
    q_est_diff_Maipo_calidad.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    # writer.save()

def QAQC_p():
    
    # ----------- rutas
    
    ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos'
    ruta_CF = ruta_OD+r'\Datos CF\Datos_BNA_EstacionesDGA\BNAT_PrecipitacionDiaria.txt'
    ruta_metadata = ruta_OD + r'\Datos CF\Estaciones\Estaciones\Metadatos estaciones\DGA\Metadatos_DGA.xls'
    dicc_ruta = {'Maipo' : r'\datosDGA\Pp\Maipo\RIO MAIPO_P_diario.csv' ,
                 'Rapel' : r'\datosDGA\Pp\Rapel\RIO RAPEL_P_diario.csv',
                 'Mataquito' : r'\datosDGA\Pp\Mataquito\RIO Mataquito_P_diario.csv',
                 'Maule' : r'\datosDGA\Pp\Maule\RIO Maule_P_diario.csv'}

    pp_CF =  pd.read_csv(ruta_CF, sep = ';')
    pp_CF['region'] = [x[:3] for x in pp_CF.iloc[:,0]]
    est_CF_Valp = pp_CF[pp_CF['region'] == '054'].iloc[:,0].drop_duplicates()
    meta_CF_Valp = pp_CF[pp_CF['region'] == '054'].iloc[:,1].drop_duplicates()
    est_CF_Maipo = pp_CF[pp_CF['region'] == '057'].iloc[:,0].drop_duplicates()
    est_CF_Rapel = pp_CF[pp_CF['region'] == '060'].iloc[:,0].drop_duplicates()
    pp_CF['region'] = [x[:3] for x in pp_CF.iloc[:,0]]
    est_CF_Mataquito = pp_CF[pp_CF['region'] == '071'].iloc[:,0].drop_duplicates()
    est_CF_Maule = pp_CF[pp_CF['region'] == '073'].iloc[:,0].drop_duplicates()
    est_CF_Nuble = pp_CF[pp_CF['region'] == '081'].iloc[:,0].drop_duplicates()
    meta_CF_Nuble = pp_CF[pp_CF['region'] == '081'].iloc[:,1].drop_duplicates()
    
    # =======================
    #       Valparaíso      
    # =======================
    
    pp_est_diff_Valp = ordenarDGA(est_CF_Valp.values, pp_CF)
    pp_est_diff_Valp = pp_est_diff_Valp[pp_est_diff_Valp.index > '1979-01-01']
    for ind,row in meta_CF_Valp.items():
        string = meta_CF_Valp.loc[ind].replace(' ','_')
        meta_CF_Valp.loc[ind] =  string.lower().capitalize()+'_DGA'

    # ----------Cruce

    meta_Valpo = pd.read_excel(ruta_metadata, sheet_name = 'Ppt', index_col = 0)
    
    metadata_V = pd.DataFrame([])
    
    for x in meta_CF_Valp.values:
        metadata_V = metadata_V.append(meta_Valpo[meta_Valpo['Estacion'].isin([x])])
        
    metadata_V['Estacion_name'] = pp_est_diff_Valp.columns
    
    #---Guardar V region
    
    writer = pd.ExcelWriter(ruta_OD+r'\pp_est_Valpo.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    metadata_V.to_excel(writer, sheet_name= 'data')
    pp_est_diff_Valp.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()
       
    
    # =======================
    #       Maipo      
    # =======================
    
    cuenca = 'Maipo'
    
    pp_CC = pd.read_csv(ruta_OD+'\\'+dicc_ruta[cuenca], index_col = 0, parse_dates = True)
    est_CC_maipo = pp_CC.columns

    # --------------- estaciones diferentes
    
    est_diff_Maipo = [x for x in est_CF_Maipo if x not in est_CC_maipo]
    pp_est_diff_Maipo = ordenarDGA(est_diff_Maipo, pp_CF)
    pp_est_diff_Maipo = pp_est_diff_Maipo.loc[pp_est_diff_Maipo.index >= '1979-01-01']
    pp_est_diff_Maipo_mon = pp_est_diff_Maipo.resample('MS').mean()
    
    pp_est_diff_Maipo_calidad = pp_est_diff_Maipo.copy()
    pp_est_diff_Maipo_calidad[:] = 1
    pp_est_diff_Maipo_calidad[pp_est_diff_Maipo.isnull()] = 0
    pp_est_diff_Maipo_calidad_mon = pp_est_diff_Maipo_calidad.resample('MS').sum()
    pp_est_diff_Maipo_mon[pp_est_diff_Maipo_calidad_mon > 20].plot()
    
    # writer = pd.ExcelWriter(ruta_OD+r'\q_est_diff_Maipo'+'.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    q_est_diff_Maipo_mon_flags.to_excel(writer, sheet_name= 'data')
    q_est_diff_Maipo_calidad.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    # writer.save()
    
    # --------------- estaciones comunes
    
    est_nodiff_Maipo = [x for x in est_CF_Maipo if x in est_CC_maipo]
    pp_est_nodiff_Maipo = ordenarDGA(est_nodiff_Maipo, pp_CF)
    pp_est_nodiff_Maipo = pp_est_nodiff_Maipo.loc[pp_est_nodiff_Maipo.index >= '1979-01-01']
    
    pp_est_nodiff_Maipo_calidad = pp_est_nodiff_Maipo.copy()
    
    pp_est_nodiff_Maipo_calidad[:] = 1
    pp_est_nodiff_Maipo_calidad[pp_est_nodiff_Maipo.isnull()] = 0
    
    pp_est_nodiff_Maipo_mon = pp_est_nodiff_Maipo.copy().resample('MS').sum()
    pp_est_nodiff_Maipo_mon = pp_est_nodiff_Maipo_mon[pp_est_nodiff_Maipo_calidad.resample('MS').sum() > 20]
    pp_est_nodiff_Maipo_mon.plot()

    pp_CC_calidad = calidadpp(pp_CC)
    
    fig, ax = plt.subplots(10,6)
    ax = ax.reshape(-1)
    
    for ind, col in enumerate(pp_est_nodiff_Maipo_mon.columns):
        pp_est_nodiff_Maipo_mon[col].plot(ax = ax[ind], linewidth = 1.9, color = 'r')
        pp_CC_calidad[col].plot(ax = ax[ind], color = 'b')
        ax[ind].set_title(col)
        ax[ind].set_ylabel('Pp ($mm/m$)')
    
    for i in list(np.arange(56,60)):
        fig.delaxes(ax.flatten()[i])
        
    ax[ind].legend(['Carlos Flores','BNA'],bbox_to_anchor=(1.75, 0.75), loc='upper left') 
    
    
    # =======================
    #       Rapel      
    # =======================
    
    cuenca = 'Rapel'
    
    pp_CC = pd.read_csv(ruta_OD+'\\'+dicc_ruta[cuenca], index_col = 0, parse_dates = True)
    est_CC_rapel = pp_CC.columns

    # --------------- estaciones diferentes
    
    est_diff_Rapel = [x for x in est_CF_Rapel if x not in est_CC_rapel]
    pp_est_diff_Rapel = ordenarDGA(est_diff_Rapel, pp_CF)
    pp_est_diff_Rapel = pp_est_diff_Rapel.loc[pp_est_diff_Rapel.index >= '1979-01-01']
    pp_est_diff_Rapel_mon = pp_est_diff_Rapel.resample('MS').mean()
    
    pp_est_diff_Rapel_calidad = pp_est_diff_Rapel.copy()
    pp_est_diff_Rapel_calidad[:] = 1
    pp_est_diff_Rapel_calidad[pp_est_diff_Rapel.isnull()] = 0
    pp_est_diff_Rapel_calidad_mon = pp_est_diff_Rapel_calidad.resample('MS').sum()
    pp_est_diff_Rapel_mon[pp_est_diff_Rapel_calidad_mon > 20].plot()
    
    writer = pd.ExcelWriter(ruta_OD+r'\pp_est_diff_Rapel'+'.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    pp_est_diff_Rapel.to_excel(writer, sheet_name= 'data')
    pp_est_diff_Rapel_calidad.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()
    
    # --------------- estaciones comunes
    
    est_nodiff_Rapel = [x for x in est_CF_Rapel if x in est_CC_rapel]
    pp_est_nodiff_Rapel = ordenarDGA(est_nodiff_Rapel, pp_CF)
    pp_est_nodiff_Rapel = pp_est_nodiff_Rapel.loc[pp_est_nodiff_Rapel.index >= '1979-01-01']
    
    pp_est_nodiff_Rapel_calidad = pp_est_nodiff_Rapel.copy()
    
    pp_est_nodiff_Rapel_calidad[:] = 1
    pp_est_nodiff_Rapel_calidad[pp_est_nodiff_Rapel.isnull()] = 0
    
    pp_est_nodiff_Rapel_mon = pp_est_nodiff_Rapel.copy().resample('MS').sum()
    pp_est_nodiff_Rapel_mon = pp_est_nodiff_Rapel_mon[pp_est_nodiff_Rapel_calidad.resample('MS').sum() > 20]
    pp_est_nodiff_Rapel_mon.plot()

    pp_CC_calidad = calidadpp(pp_CC)
    
    fig, ax = plt.subplots(6,6)
    ax = ax.reshape(-1)
    
    for ind, col in enumerate(pp_est_nodiff_Rapel_mon.columns):
        pp_est_nodiff_Rapel_mon[col].plot(ax = ax[ind], linewidth = 1.9, color = 'r')
        pp_CC_calidad[col].plot(ax = ax[ind], color = 'b')
        ax[ind].set_title(col)
        ax[ind].set_ylabel('Pp ($mm/m$)')
    
    for i in list(np.arange(32,36)):
        fig.delaxes(ax.flatten()[i])
        
    ax[ind].legend(['Carlos Flores','BNA'],bbox_to_anchor=(1.75, 0.75), loc='upper left') 
    
    # =======================
    #       Mataquito      
    # =======================
    
    cuenca = 'Mataquito'
    
    pp_CC = pd.read_csv(ruta_OD+'\\'+dicc_ruta[cuenca], index_col = 0, parse_dates = True)
    est_CC_Mataquito = pp_CC.columns

    # --------------- estaciones diferentes
    
    est_diff_Mataquito = [x for x in est_CF_Mataquito if x not in est_CC_Mataquito]
    pp_est_diff_Mataquito = ordenarDGA(est_diff_Mataquito, pp_CF)
    pp_est_diff_Mataquito = pp_est_diff_Mataquito.loc[pp_est_diff_Mataquito.index >= '1979-01-01']
    pp_est_diff_Mataquito_mon = pp_est_diff_Mataquito.resample('MS').mean()
    
    pp_est_diff_Mataquito_calidad = pp_est_diff_Mataquito.copy()
    pp_est_diff_Mataquito_calidad[:] = 1
    pp_est_diff_Mataquito_calidad[pp_est_diff_Mataquito.isnull()] = 0
    pp_est_diff_Mataquito_calidad_mon = pp_est_diff_Mataquito_calidad.resample('MS').sum()
    pp_est_diff_Mataquito_mon[pp_est_diff_Mataquito_calidad_mon > 20].plot()

    
    # --------------- estaciones comunes
    
    est_nodiff_Mataquito = [x for x in est_CF_Mataquito if x in est_CC_Mataquito]
    pp_est_nodiff_Mataquito = ordenarDGA(est_nodiff_Mataquito, pp_CF)
    pp_est_nodiff_Mataquito = pp_est_nodiff_Mataquito.loc[pp_est_nodiff_Mataquito.index >= '1979-01-01']
    
    pp_est_nodiff_Mataquito_calidad = pp_est_nodiff_Mataquito.copy()
    
    pp_est_nodiff_Mataquito_calidad[:] = 1
    pp_est_nodiff_Mataquito_calidad[pp_est_nodiff_Mataquito.isnull()] = 0
    
    pp_est_nodiff_Mataquito_mon = pp_est_nodiff_Mataquito.copy().resample('MS').sum()
    pp_est_nodiff_Mataquito_mon = pp_est_nodiff_Mataquito_mon[pp_est_nodiff_Mataquito_calidad.resample('MS').sum() > 20]
    pp_est_nodiff_Mataquito_mon.plot()

    pp_CC_calidad = calidadpp(pp_CC)
    
    fig, ax = plt.subplots(6,2)
    ax = ax.reshape(-1)
    
    for ind, col in enumerate(pp_est_nodiff_Mataquito_mon.columns):
        pp_est_nodiff_Mataquito_mon[col].plot(ax = ax[ind], linewidth = 1.9, color = 'r')
        pp_CC_calidad[col].plot(ax = ax[ind], color = 'b')
        ax[ind].set_title(col)
        ax[ind].set_ylabel('Pp ($mm/m$)')
    
    for i in list(np.arange(32,36)):
        fig.delaxes(ax.flatten()[i])
        
    ax[ind].legend(['Carlos Flores','BNA'],bbox_to_anchor=(1.75, -0.75), loc='upper left') 
    
    
    # =======================
    #       Maule      
    # =======================
    
    cuenca = 'Maule'
    
    pp_CC = pd.read_csv(ruta_OD+'\\'+dicc_ruta[cuenca], index_col = 0, parse_dates = True)
    est_CC_Maule = pp_CC.columns

    # --------------- estaciones diferentes
    
    est_diff_Maule = [x for x in est_CF_Maule if x not in est_CC_Maule]
    pp_est_diff_Maule = ordenarDGA(est_diff_Maule, pp_CF)
    pp_est_diff_Maule = pp_est_diff_Maule.loc[pp_est_diff_Maule.index >= '1979-01-01']
    pp_est_diff_Maule_mon = pp_est_diff_Maule.resample('MS').mean()
    
    pp_est_diff_Maule_calidad = pp_est_diff_Maule.copy()
    pp_est_diff_Maule_calidad[:] = 1
    pp_est_diff_Maule_calidad[pp_est_diff_Maule.isnull()] = 0
    pp_est_diff_Maule_calidad_mon = pp_est_diff_Maule_calidad.resample('MS').sum()
    pp_est_diff_Maule_mon[pp_est_diff_Maule_calidad_mon > 20].plot()
    
    
    # --------------- estaciones comunes
    
    est_nodiff_Maule = [x for x in est_CF_Maule if x in est_CC_Maule]
    pp_est_nodiff_Maule = ordenarDGA(est_nodiff_Maule, pp_CF)
    pp_est_nodiff_Maule = pp_est_nodiff_Maule.loc[pp_est_nodiff_Maule.index >= '1979-01-01']
    
    pp_est_nodiff_Maule_calidad = pp_est_nodiff_Maule.copy()
    
    pp_est_nodiff_Maule_calidad[:] = 1
    pp_est_nodiff_Maule_calidad[pp_est_nodiff_Maule.isnull()] = 0
    
    pp_est_nodiff_Maule_mon = pp_est_nodiff_Maule.copy().resample('MS').sum()
    pp_est_nodiff_Maule_mon = pp_est_nodiff_Maule_mon[pp_est_nodiff_Maule_calidad.resample('MS').sum() > 20]
    pp_est_nodiff_Maule_mon.plot()

    pp_CC_calidad = calidadpp(pp_CC)
    
    fig, ax = plt.subplots(7,8)
    ax = ax.reshape(-1)
    
    for ind, col in enumerate(pp_est_nodiff_Maule_mon.columns):
        pp_est_nodiff_Maule_mon[col].plot(ax = ax[ind], linewidth = 3, color = 'r')
        pp_CC_calidad[col].plot(ax = ax[ind], color = 'b')
        ax[ind].set_title(col)
        ax[ind].set_ylabel('Pp ($mm/m$)')
    
    for i in list(np.arange(51,56)):
        fig.delaxes(ax.flatten()[i])
        
    ax[ind].legend(['Carlos Flores','BNA'],bbox_to_anchor=(1.75, 0.75), loc='upper left') 
    
    
    # =======================
    #       Ñuble      
    # =======================
    
    pp_est_diff_Nuble = ordenarDGA(est_CF_Nuble.values, pp_CF)
    pp_est_diff_Nuble = pp_est_diff_Nuble[pp_est_diff_Nuble.index > '1979-01-01']
    for ind,row in meta_CF_Nuble.items():
        string = meta_CF_Nuble.loc[ind].replace(' ','_')
        meta_CF_Nuble.loc[ind] =  string.lower().capitalize()+'_DGA'
        meta_CF_Nuble.loc[ind] = meta_CF_Nuble.loc[ind].replace(u"\xf1", "n")
        meta_CF_Nuble.loc[ind] = meta_CF_Nuble.loc[ind].replace('(', "")
        meta_CF_Nuble.loc[ind] = meta_CF_Nuble.loc[ind].replace(')', "")


    # ----------Cruce

    meta_Nuble = pd.read_excel(ruta_metadata, sheet_name = 'Ppt', index_col = 0)
    
    metadata_VIII = pd.DataFrame([])
    
    for x in meta_CF_Nuble.values:
        metadata_VIII = metadata_VIII.append(meta_Valpo[meta_Valpo['Estacion'].isin([x])])
            
    metadata_VIII['Estacion_name'] = pp_est_diff_Nuble.columns
    
    #---Guardar VIII region
    
    writer = pd.ExcelWriter(ruta_OD+r'\pp_est_Nuble.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    metadata_VIII.to_excel(writer, sheet_name= 'data')
    pp_est_diff_Nuble.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()
    
def ppFN():
    
    # =====================
    #         rutas
    # =====================
    
    ruta_FN = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\QAQC\Pp\20201215_reporte_web.xlsx'
    pp_FN = pd.read_excel(ruta_FN, skiprows = 1, index_col = 0, parse_dates = True)
    
    # =====================
    # comparar estaciones
    # =====================
    
    est_FN = pp_FN.columns
    est_CC = list(est_FN)
    
    for ind,est in enumerate(est_CC):
        if '-' in str(est):
            est_CC[ind] = str(est_CC[ind]).split('-')[0]
        if str(est)[0] == '0':
            est_CC[ind] = str(est_CC[ind][1:])
            
        est_CC[ind] = str(est_CC[ind])
    
    est_FN = list(est_FN)
    
    for ind,est in enumerate(est_FN):
        est_FN[ind] = str(est_FN[ind])
    
    est_diff = np.setdiff1d(est_FN,est_CC)
    est_diff = np.setdiff1d(est_CC,est_FN)
    
    
    # ================================
    #     validar raster AgroClima
    # ================================
    
    plt.close("all")
    ruta_pp = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\AgroClima\HISTORICO\PPA\lmStepAIC_srtm_ppa_historico_ppa_hist.tif'
    raster_pp = rasterio.open(ruta_pp)
    puntos_pp = geopandas.read_file(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\AgroClima\HISTORICO\PPA\PPA_hist_QAQC_clip.shp')

    puntos_pp_utm = puntos_pp.to_crs({'init': 'epsg:32719'})

    fig,ax = plt.subplots(1)
    
    xlim = ([160_000,  500_000])
    ylim = ([5.95*1e6,  6.35*1e6])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    c = show(raster_pp, cmap='pink', vmin=0, ax = ax)
    c = ax.imshow(raster_pp.read(1), cmap ='pink', vmin = 0, vmax = 1800,
              interpolation ='nearest')
    cb = fig.colorbar(c, ax = ax)
    cb.ax.set_ylabel('Pp media (mm/yr)', rotation=90)

    
    puntos_pp_utm['PPA_hist'] = puntos_pp_utm['PPA_hist'].astype(float)
    puntos_pp_utm.plot(column='PPA_hist', ax = ax, vmin = 0, vmax = 1800, cmap ='pink')
    
    
def parseDMC_T():
    """
    función para ordenar la temperatura del aire de la DMC

    Returns
    -------
    None.

    """
    # librerias
    import os
    
    path = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\datosDMC\T'
    
    writer = pd.ExcelWriter(ruta_OD+r'\Tmed_DMC.xlsx', engine='xlsxwriter')

    # df para guardar
    df_tmed = pd.DataFrame([], index = pd.date_range('1979-01-01','2020-04-01',freq = '1d'))
    metadata_tmed = pd.DataFrame([], columns = ['Codigo Estacion', 'Nombre estacion',	'Codigo Estacion',	'Cuenca',	'Subcuenca',	'UTM Este',
                                             'UTM Norte',	'Latitud',	'Longitud',	'Altitud'])
    for filename in os.listdir(path):
       if 'Temperatura del Aire Seco_estacion' in filename:
           # nombre estacion
           est = filename.replace('.xlsx','')[-6:]
           
           # datos T
           df_t = pd.read_excel(path+'\\'+filename, sheet_name = 'Datos', index_col = 0, parse_dates = True)
           # formato DMC
           df_t['Ts'][df_t['Ts'] == '.'] = np.nan
           df_t = df_t['Ts'].astype(float)
           
           # patos DMC
           df_t[df_t < -10] = -1*df_t[df_t < -10]
           df_t_day = df_t.resample('1d').mean() 
           df_t_day = df_t_day.loc[(df_t_day.index >= '1979-01-01') & (df_t_day.index <= '2020-04-01')]
           df_tmed[est] = np.nan

           # guardar
           df_tmed.loc[df_t_day.index,est] = df_t_day.values

           # metadatos
           metadata_t = pd.read_excel(path+'\\'+filename, sheet_name = 'ficha_est', index_col = 1)
           metadata_t = metadata_t.loc[:, ~metadata_t.columns.str.contains('^Unnamed', na=False)]
           metadata_t.loc['Altitud',:] = float(metadata_t.loc['Altitud',:].values[0][:-5])
           campos = ['Código Nacional','Nombre de la Estación','Código Nacional','Zona Geográfica','Zona Geográfica',
                     'Latitud','Longitud','Altitud']
           metadatos = pd.DataFrame(index = [list(range(0,10))], columns = ['1'])
           metadatos.loc[[0,1,2,3,4,7,8,9],:] = metadata_t.loc[campos].values
           metadatos.index = ['Codigo Estacion', 'Nombre estacion',	'Codigo Estacion',	'Cuenca',	'Subcuenca',	'UTM Este',
                                             'UTM Norte',	'Latitud',	'Longitud',	'Altitud']
           metadata_tmed = pd.concat([metadata_tmed,metadatos.transpose()], ignore_index=True)
                     
    # Write each dataframe to a different worksheet.
    metadata_tmed.to_excel(writer, sheet_name= 'data')
    df_tmed.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()
    
    # -------------------
    #     Maipo
    # -------------------
    
    estaciones_Maipo = [330171,330172,330173,330174,330167,330168,330169,330170,330180,330181,330190,330175,330177,330178,330114,330118,330121,330122,330093,330111,330112,330113,330162,330164,330165,330166,330141,330151,330160,330020,330021,330023,330026,330019,330075,330076,330077,330083,330027,330065,330071]
    tmed_Maipo = pd.read_excel(ruta_OD+r'\Tmed_DMC.xlsx', sheet_name = 'info data', index_col = 0, parse_dates = True)
    tmed_Maipo = tmed_Maipo.loc[:, ~tmed_Maipo.columns.str.contains('^Unnamed', na=False)]
    tmed_Maipo = tmed_Maipo[[str(x) for x in estaciones_Maipo]]


    tmed_Maipo_metadata = pd.read_excel(ruta_OD+r'\Tmed_DMC.xlsx', sheet_name = 'data', index_col = 1)
    tmed_Maipo_metadata = tmed_Maipo_metadata.loc[:, ~tmed_Maipo_metadata.columns.str.contains('^Unnamed', na=False)]
    tmed_Maipo_metadata = tmed_Maipo_metadata.loc[estaciones_Maipo]
    
    # guardar
    
    writer = pd.ExcelWriter(ruta_OD+r'\Tmed_DMC_Maipo.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    tmed_Maipo.to_excel(writer, sheet_name= 'data')
    tmed_Maipo_metadata.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()
    
    # -------------------
    #     Rapel
    # -------------------    
    
    estaciones_Rapel = [340102,340113,340116,340117,340118,340119,340120,340121,340122,340123,340124,340125,340126,340127,340128,340129,340130,340132,340134,340136,340137,340139,330158,340146,340147,340149,330179,330182,330192,340032,340039,340042,340045,340048,340050,340052,340053,340077,340093,340098]
    tmed_Rapel = pd.read_excel(ruta_OD+r'\Tmed_DMC.xlsx', sheet_name = 'info data', index_col = 0, parse_dates = True)
    tmed_Rapel = tmed_Rapel.loc[:, ~tmed_Rapel.columns.str.contains('^Unnamed', na=False)]
    tmed_Rapel = tmed_Rapel[[str(x) for x in estaciones_Rapel]]


    tmed_Rapel_metadata = pd.read_excel(ruta_OD+r'\Tmed_DMC.xlsx', sheet_name = 'data', index_col = 1)
    tmed_Rapel_metadata = tmed_Rapel_metadata.loc[:, ~tmed_Rapel_metadata.columns.str.contains('^Unnamed', na=False)]
    tmed_Rapel_metadata = tmed_Rapel_metadata.loc[estaciones_Rapel]
    
    # guardar
    
    writer = pd.ExcelWriter(ruta_OD+r'\Tmed_DMC_Rapel.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    tmed_Rapel.to_excel(writer, sheet_name= 'data')
    tmed_Rapel_metadata.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()
    
    
    #%% Exportar para Climatol
    
    #.dat data
    ruta_data = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Clima\Tmed\Consolidado\Rapel\Rapel_Tmed_consolidado.xlsx'
    ruta_data = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Clima\Tmed\Consolidado\Maipo\RIO MAIPO_Tmed_diario_consolidado.xlsx'
    ruta_data  = r'C:\Users\ccalvo\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Clima\Pp\Consolidado\Maipo\Maipo_consolidado.xlsx'

    data = pd.read_excel(ruta_data, sheet_name = 'data', index_col = 0, parse_dates = True)
    
    data_climatol = data.replace(" ", 'NA')
    data_climatol = data_climatol.replace(np.nan, 'NA')
    
    data_climatol = data_climatol[(data_climatol.index.year >= 1990) & (data_climatol.index <= '2019-12-31')]
    
    ruta_data_out = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Clima\Tmed\Consolidado\Rapel\Rapel_Tmed_consolidado_2000-2019.dat'
    ruta_data_out =  r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Clima\Tmed\Consolidado\Maipo\Maipo_Tmed_consolidado_2000-2019.dat'
    ruta_data_out = r'C:\Users\ccalvo\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Clima\Pp\Consolidado\Maipo\Maipo_consolidado_1990-2019.dat'
    
    data_climatol.to_csv(ruta_data_out, index = False, sep = " ", header = False)
    
    # .est estaciones
    ruta_consolidado = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Clima\Tmed\Consolidado\Rapel\Rapel_Tmed_consolidado.xlsx'
    est_tmed = pd.read_excel(ruta_consolidado, sheet_name = 'info estacion')
    columnas = ['Long','Lat','Altitud','Codigo Estacion','Nombre estacion']
    est_tmed = est_tmed[columnas]
    alias = est_tmed['Codigo Estacion']
    nombres = est_tmed['Nombre estacion']
    
    for i in range(len(alias)):
        alias_i = alias[i]
        nombres[i] = "\""+nombres[i]+"\""
        if '-' in str(alias_i):
            alias_i = alias_i[:-2]
            alias[i] = alias_i  
        if "\"" not in [alias_i]:
            alias_i = "\""+str(alias_i)+"\""
            alias[i] = alias_i
    
    ruta_salida = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\Clima\Tmed\Consolidado\Rapel\RIO Rapel_Tmed_diario_consolidado.est'
    est_tmed.to_csv(ruta_salida, index = False, header = None, sep = ' ')

    
    
def QAQC_t():
    import os
 # ----------- rutas
    
    ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos'
    ruta_CF = ruta_OD+r'\Datos CF\Datos_BNA_EstacionesDGA\BNAT_TemperaturaDiaria.txt'
    dicc_rutaDGA = {'Maipo' : r'\datosDGA\Tmed\Maipo\RIO MAIPO_Tmed_diario.csv' ,
                 'Rapel' : r'\datosDGA\Tmed\Rapel\RIO RAPEL_Tmed_diario.csv',
                 'Mataquito' : r'\datosDGA\Tmed\Mataquito\RIO Mataquito_Tmed_diario.csv',
                 'Maule' : r'\datosDGA\Tmed\Maule\RIO Maule_Tmed_diario.csv'} 

    
    pp_CF =  pd.read_csv(ruta_CF, sep = ';')
    pp_CF['region'] = [x[:3] for x in pp_CF.iloc[:,0]]
    est_CF_Valp = pp_CF[pp_CF['region'] == '054'].iloc[:,0].drop_duplicates()
    meta_CF_Valp = pp_CF[pp_CF['region'] == '054'].iloc[:,1].drop_duplicates()
    est_CF_Maipo = pp_CF[pp_CF['region'] == '057'].iloc[:,0].drop_duplicates()
    est_CF_Rapel = pp_CF[pp_CF['region'] == '060'].iloc[:,0].drop_duplicates()
    pp_CF['region'] = [x[:3] for x in pp_CF.iloc[:,0]]
    est_CF_Mataquito = pp_CF[pp_CF['region'] == '071'].iloc[:,0].drop_duplicates()
    est_CF_Maule = pp_CF[pp_CF['region'] == '073'].iloc[:,0].drop_duplicates()
    est_CF_Nuble = pp_CF[pp_CF['region'] == '081'].iloc[:,0].drop_duplicates()
    meta_CF_Nuble = pp_CF[pp_CF['region'] == '081'].iloc[:,1].drop_duplicates()
    
    # =======================
    #       Valparaíso      
    # =======================
    
    pp_est_diff_Valp = ordenarDGA(est_CF_Valp.values, pp_CF)
    pp_est_diff_Valp = pp_est_diff_Valp[pp_est_diff_Valp.index > '1979-01-01']
    for ind,row in meta_CF_Valp.items():
        string = meta_CF_Valp.loc[ind].replace(' ','_')
        meta_CF_Valp.loc[ind] =  string.lower().capitalize()+'_DGA'

    # ----------Cruce

    meta_Valpo = pd.read_excel(ruta_metadata, sheet_name = 'Ppt', index_col = 0)
    
    metadata_V = pd.DataFrame([])
    
    for x in meta_CF_Valp.values:
        metadata_V = metadata_V.append(meta_Valpo[meta_Valpo['Estacion'].isin([x])])
        
    metadata_V['Estacion_name'] = pp_est_diff_Valp.columns
    
    #---Guardar V region
    
    writer = pd.ExcelWriter(ruta_OD+r'\pp_est_Valpo.xlsx', engine='xlsxwriter')
            
    # Write each dataframe to a different worksheet.
    metadata_V.to_excel(writer, sheet_name= 'data')
    pp_est_diff_Valp.to_excel(writer, sheet_name= 'info data')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()
