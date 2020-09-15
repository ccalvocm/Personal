# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author: CCCM

"""

# Limpiar entorno

%reset -f

import win32com.client
import time
import os, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

#%% Copiapo

# Carpetas de entrada 

path_escenarios = r'D:\ARClim\Extraer_series\Salidas\futuro'
path_climate = r'D:\WEAP\Weap areas\Copiapo_WEAP_MODFLOW_PEGRH_Cal\MV2'

# Carpetas de salida 
    
output_folder = r'D:\ARClim\Copiapo\Salidas'
ruta_modelo = r'D:\WEAP\Weap areas\Copiapo_WEAP_MODFLOW_PEGRH_Cal'

overview = ['Default']

#Abrir WEAP y crear objeto WEAP

WEAP = win32com.client.Dispatch('WEAP.WEAPApplication')

#Seleccionar Area

#time.sleep(5)

# Funcion para borrar inputs antiguos como buenas prácticas

def borrar(folder,file):
    for filename in os.listdir(folder):
        if filename == file:
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

# función para rellenar años vacíos
            
def extenderMensual(dataframe, ini):
    
    indiceInicial = len(dataframe)
    start = dataframe.index.max().date() 
    end = pd.to_datetime('2064-12-01',format='%Y-%m-%d')
    nuevasFechas = pd.DataFrame(pd.date_range(start, end, freq='MS', closed='right'),  columns =['fecha'])
    nuevasFechas.set_index(pd.date_range(start, end, freq='MS', closed='right'), inplace=True)
    dataframe = dataframe.append(nuevasFechas)
    dataframe = dataframe.drop('fecha', axis = 1)
    dataframe = dataframe.reset_index()
   
    #repetir años
    yearHalf= 8
    indiceInicial = indiceInicial
    chunk = dataframe.loc[indiceInicial-yearHalf*12:indiceInicial-1]
    contador = 0
    dataframe.rename(columns={'index':'fecha'}, inplace=True)
    fechas = pd.to_datetime(dataframe['fecha'])
        
    # inicio del completado    
    for j in range(4):
        for i in range(len(chunk)):      
            aux = chunk.iloc[i]
            dataframe.loc[indiceInicial+contador] = aux[:]
            contador = contador + 1 
            
    dataframe['fecha'] = fechas        
    dataframe['Year'] = pd.DatetimeIndex(fechas).year
    dataframe['Month'] = pd.DatetimeIndex(fechas).month
    yr = dataframe['Year']
    mnth = dataframe['Month']
    dataframe.drop(labels=['Year'], axis=1,inplace = True)
    dataframe.drop(labels=['Month'], axis=1,inplace = True)
    dataframe.drop(labels=['fecha'], axis=1,inplace = True)
    dataframe.insert(0, 'Year', yr)
    dataframe.insert(1, 'Month', mnth)

    return dataframe
        
def extenderFuturo(df_presente,df_futuro,years,string):
    # preproceso de serie futura
    if string == 'pp':
        df_futuro = df_futuro.resample('MS').sum()
    elif string == 't':
        df_futuro = df_futuro.resample('MS').mean()
        
#    df_futuro['Mo'] = df_futuro.index.month
    df_futuro = df_futuro.drop(['Day'], axis = 1)
    df_presente = df_presente.drop(['Day'], axis = 1)
    
    # extendido
    ini = 600
    fechas = pd.to_datetime(df_presente['fecha'])
    df_futuro = df_futuro.reset_index()
    chunk = df_futuro.loc[0:years*12-1]
    contador = 0
    
    for j in range(0):
        for i in range(len(chunk)):
            aux = chunk.iloc[i].values.tolist()
            df_presente.loc[ini+contador] = aux[:]
            contador = contador + 1 
    
    inifuturo = 660
    indice = 0
    
    for i in range(len(df_futuro)):
        aux = df_futuro.iloc[indice].values.tolist()
        df_presente.loc[inifuturo+i] = aux[:]
        indice = indice + 1
    
    df_presente['fecha'] = fechas
    df_presente['Month'] = fechas.dt.month
    df_presente['Year'] = fechas.dt.year
#    del df_presente['fecha']

    return df_presente 

def extenderMensualGCM(dataframe, ini):
    
    indiceInicial = len(dataframe)
    start = dataframe.index.max().date() 
    end = pd.to_datetime('2064-12-01',format='%Y-%m-%d')
    nuevasFechas = pd.DataFrame(pd.date_range(start, end, freq='MS', closed='right'),  columns =['fecha'])
    nuevasFechas.set_index(pd.date_range(start, end, freq='MS', closed='right'), inplace=True)
#    dataframe = dataframe.append(nuevasFechas)
    dataframe = dataframe.append(nuevasFechas)[dataframe.columns.tolist()]
#    dataframe = dataframe.drop('fecha', axis = 1)
    dataframe = dataframe.reset_index()
   
    #repetir años
    yearHalf= 5
    indiceInicial = indiceInicial
    chunk = dataframe.loc[indiceInicial-yearHalf*12:indiceInicial-1]
    contador = 0
    dataframe.rename(columns={'index':'fecha'}, inplace=True)
    fechas = pd.to_datetime(dataframe['fecha'])
        
    # inicio del completado    
    for j in range(5):
        for i in range(len(chunk)):      
            aux = chunk.iloc[i]
            dataframe.loc[indiceInicial+contador] = aux[:]
            contador = contador + 1 
            
    dataframe['fecha'] = fechas        
    dataframe['Year'] = pd.DatetimeIndex(fechas).year
    dataframe['Month'] = pd.DatetimeIndex(fechas).month
    yr = dataframe['Year']
    mnth = dataframe['Month']
#    dataframe.drop(labels=[';Año'], axis=1,inplace = True)
    dataframe.drop(labels=['Month'], axis=1,inplace = True)
    dataframe.insert(1, 'Month', mnth)
    dataframe.drop(labels=['Year'], axis=1,inplace = True)
    dataframe.insert(2, 'Year', yr)

    return dataframe

def reordenarColumnas(df):
    cols = list(df.columns)
    a, b = cols.index('Year'), cols.index('Month')
    cols[b], cols[a] = cols[a], cols[b]
    df = df[cols]
    return df


#%% Iterar sobre modelos Futuro 6 escenarios

modelos_clima = next(os.walk(path_escenarios))[1]

archivos_clima = ['BandasCarrizalilloB1','BandasCarrizalilloB2','BandasCarrizalilloB3','Bandas Jorquera_B1','Bandas Jorquera_B2','Bandas Jorquera_B3','Bandas Jorquera_B4','Bandas Jorquera_B5','Bandas Manflas_B1','Bandas Manflas_B2','Bandas Manflas_B3','Bandas Manflas_B4','Bandas Paipote_B1','Bandas Paipote_B2','Bandas Paipote_B3','Bandas Pulido_B1','Bandas Pulido_B2','Bandas Pulido_B3','Bandas Pulido_B4','Bandas Pulido_B5','COPIAPO_1','COPIAPO_2','COPIAPO_3','COPIAPO_4','COPIAPO_5','COPIAPO_6','COPIAPO_7','COPIAPO_8','COPIAPO_9','COPIAPO_10','COPIAPO_11','COPIAPO_12','COPIAPO_13','COPIAPO_14']

modelo = 'inmcm4'
for modelo in modelos_clima:
        
    # Leer precipitación y temperatura historica GCM 
    path_historico = 'D:\\ARClim\\Extraer_series\\Salidas\\historico_GCM\\'+modelo+r'\historico'
    pp_historico = []
    pp_subcuencas_control = []
    tmax_historico = []
    tmin_historico = []

    for r, d, f in os.walk(path_historico):
        for file in f:
            if 'prCopiapo_control' in str(file):
                pp_historico = pd.read_csv(os.path.join(r, file))
                pp_historico.set_index(pd.to_datetime(pp_historico[['Day','Month','Year']]), inplace=True)
#            elif 'pr_hist_Limari_irrigation' in str(file):
#                pp_historico_irrigacion = pd.read_csv(os.path.join(r, file))
#                pp_historico_irrigacion.set_index(pd.to_datetime(pp_historico[['Day','Month','Year']]), inplace=True)             
            elif 'tasmaxCopiapo_control' in str(file):
                tmax_historico = pd.read_csv(os.path.join(r, file))
                tmax_historico.set_index(pd.to_datetime(tmax_historico[['Day','Month','Year']]), inplace=True)
            elif 'tasminCopiapo_control' in str(file):
                tmin_historico = pd.read_csv(os.path.join(r, file))
                tmin_historico.set_index(pd.to_datetime(tmin_historico[['Day','Month','Year']]), inplace=True)
    
    tmedia_control = (tmax_historico + tmin_historico) / 2
    
    # Crear dataframe historico

    pp_subcuencas_control =  pp_historico.resample('MS').sum()
    pp_subcuencas_control = extenderMensualGCM(pp_subcuencas_control,1980)
#    pp_subcuencas_control = pp_subcuencas_control[['fecha','Month','Year']+subcuencas_clima+['Day']]

    t_control = tmedia_control.resample('MS').mean()
    t_control = extenderMensualGCM(t_control,1980)    
    
    path_futuro = path_escenarios+'\\'+modelo+r'\future'
    pp_futuro = []
    tmax_futuro = []
    tmin_futuro = []
    
    # Leer precipitación y temperatura
    
    for r, d, f in os.walk(path_futuro):
        for file in f:
            if 'prCopiapo_futuro' in str(file):
#                pp_futuro.append(os.path.join(r, file))
                pp_futuro = pd.read_csv(os.path.join(r, file))
                pp_futuro.set_index(pd.to_datetime(pp_futuro[['Day','Month','Year']]), inplace=True)
            elif 'tasmaxCopiapo_futuro' in str(file):
                tmax_futuro = pd.read_csv(os.path.join(r, file))
                tmax_futuro.set_index(pd.to_datetime(tmax_futuro[['Day','Month','Year']]), inplace=True)
#                tmax_futuro.append(os.path.join(r, file))      
            elif 'tasminCopiapo_futuro' in str(file):
                tmin_futuro = pd.read_csv(os.path.join(r, file))
                tmin_futuro.set_index(pd.to_datetime(tmin_futuro[['Day','Month','Year']]), inplace=True)
#                tmin_futuro.append(os.path.join(r, file))     
    
    tmedia_futuro = (tmax_futuro+tmin_futuro)/ 2
    
    # Crear DataFrames con datos climáticos para exportarlos y ser usados en modelo WEAP
    
    pp_subcuencas =  pp_futuro
    pp_subcuencas = extenderFuturo(pp_subcuencas_control,pp_subcuencas,5,'pp')
    pp_subcuencas = reordenarColumnas(pp_subcuencas) 


    temp = tmedia_futuro
    temp = extenderFuturo(t_control,temp,5,'t')
    temp = reordenarColumnas(temp)    
    
    # Escribir los archivos csv que utilizará el modelo WEAP del escenario climático j
    
    integer = 0
    for col in pp_subcuencas.columns[3:]:
        pp = pp_subcuencas[['Year','Month']+[col]]
        t = temp[['Year','Month']+[col]]
        nombre = archivos_clima[integer]

        archivo_original = pd.read_csv(path_climate+'\\'+nombre+'.shp_historicoP.csv')
        archivo_original['pp'][12:-12] = pp[col]
        archivo_original['tavg'][12:-12] = t[col]
        archivo_original.to_csv(path_climate+'\\'+nombre+'.shp_historicoP.csv', index = False)
        print(col)
        print(nombre)
#        archivo.to_csv(nombre_archivo)
        integer = integer + 1
    

#    pp_subcuencas.to_csv()
#    pp_irrigacion
#    pp_estaciones
        
  
#%% calcular WEAP

    WEAP.ActiveArea = 'Copiapo_WEAP_MODFLOW_PEGRH_Cal'
    WEAP.ActiveScenario = 'MCV'
    WEAP.verbose = 0
    WEAP.Logfile = "WEAPAutomationErrors.txt"
    WEAP.Visible = True
      
    # Calcular el escenario climático j
    
    cambio = random.randint(0,1e4)
    WEAP.Branch(r"\Key\cambio").Variable("Annual Activity Level").Expression = cambio
    
    # Activar el Scenario Explorer con los favoritos definidos según la tabla y exportar
    
    for i in range(len(overview)):  
        WEAP.LoadOverview(overview[i])
        
        # Exportar el Scenario Explorer en un solo CSV
            
        WEAP.ExportResults(output_folder+'\\'+str(modelo)+'\\'+overview[i]+'_GCM_ResultadosAltas.csv')
        
    print('El escenario '+modelo+' ha terminado')
    