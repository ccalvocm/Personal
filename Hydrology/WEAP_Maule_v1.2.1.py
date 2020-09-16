# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author: CCCM

"""

import win32com.client
import re
import time
import numpy as np
import os, shutil
import pandas as pd
import random
import matplotlib.pyplot as plt


#%% Maule + Laja
    
# Carpeta de salida y favoritos
    
output_folder = r'D:\ARClim\Maule\Salidas'
model_folder = r'E:\WEAP\Weap areas\WEAP_Maule_Laja'
path_cr2met ='D:\ARClim\Extraer_series\Salidas\historico'
path_escenarios = r'D:\ARClim\Extraer_series\Salidas\futuro'

overview = 'Resultados Atlas'

#Abrir WEAP y crear objeto WEAP

WEAP = win32com.client.Dispatch('WEAP.WEAPApplication')

#Esperar a WEAP

time.sleep(5)

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
    indiceInicial = indiceInicial-3
    chunk = dataframe.loc[indiceInicial-yearHalf*12:indiceInicial-1]
    contador = 0
    dataframe.rename(columns={'index':'fecha'}, inplace=True)
    fechas = pd.to_datetime(dataframe['fecha'])
        
    # inicio del completado    
    for j in range(2):
        for i in range(len(chunk)):      
            aux = chunk.iloc[i]
            dataframe.loc[indiceInicial+contador] = aux[:]
            contador = contador + 1 
            
    dataframe['fecha'] = fechas        
    dataframe['Year'] = pd.DatetimeIndex(fechas).year
    dataframe['Month'] = pd.DatetimeIndex(fechas).month
    yr = dataframe['Year']
    mnth = dataframe['Month']
    dataframe.drop(labels=[';Año'], axis=1,inplace = True)
    dataframe.drop(labels=['Month'], axis=1,inplace = True)
    dataframe.insert(1, 'Month', mnth)
    dataframe.drop(labels=['Year'], axis=1,inplace = True)
    dataframe.insert(2, 'Year', yr)

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
    df_presente['Year'] = fechas.dt.year
    df_presente['Month'] = fechas.dt.month
    
    return df_presente 

def extenderMensualGCM(dataframe, ini):
    
    indiceInicial = len(dataframe)
    start = dataframe.index.max().date() 
    end = pd.to_datetime('2064-12-01',format='%Y-%m-%d')
    nuevasFechas = pd.DataFrame(pd.date_range(start, end, freq='MS', closed='right'),  columns =['fecha'])
    nuevasFechas.set_index(pd.date_range(start, end, freq='MS', closed='right'), inplace=True)
#    dataframe = dataframe.append(nuevasFechas)
#    dataframe = dataframe.drop('fecha', axis = 1)
    dataframe = dataframe.append(nuevasFechas)[dataframe.columns.tolist()]
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

#%% Leer precipitación y temperatura DGA

ruta_ppMaule_DGA = r'D:\WEAP\Weap areas\WEAP_Maule_Laja\Datos_DGA\Datos\Precipitaciones\PpMauleAlto.csv'
ruta_ppLaja_DGA = r'D:\WEAP\Weap areas\WEAP_Maule_Laja\Datos_DGA\Data\Pp.txt'
ruta_t_DGA = r'D:\WEAP\Weap areas\WEAP_Maule_Laja\Datos_DGA\Datos\Temperatura\TempMauleAlto_2.csv'
ruta_tLaja_DGA = r'D:\WEAP\Weap areas\WEAP_Maule_Laja\Datos_DGA\Data\t.txt'

ppMaule_DGA = pd.read_csv(ruta_ppMaule_DGA, encoding = "ISO-8859-1")
ppMaule_DGA = ppMaule_DGA.set_index(pd.to_datetime(ppMaule_DGA[';Año'].astype(str)+'-'+ppMaule_DGA['Mes'].astype(str).str.zfill(2), format='%Y-%m'))

ppLaja_DGA = pd.read_csv(ruta_ppLaja_DGA, sep = '\t', encoding = "ISO-8859-1")
ppLaja_DGA = ppLaja_DGA.set_index(pd.to_datetime(ppLaja_DGA[';Año'].astype(str)+'-'+ppLaja_DGA['Mes'].astype(str).str.zfill(2), format='%Y-%m'))

tMaule_DGA = pd.read_csv(ruta_t_DGA, encoding = "ISO-8859-1")
tMaule_DGA = tMaule_DGA.set_index(pd.to_datetime(tMaule_DGA[';Año'].astype(str)+'-'+tMaule_DGA['Mes'].astype(str).str.zfill(2), format='%Y-%m'))

tLaja_DGA = pd.read_csv(ruta_tLaja_DGA, sep = '\t', encoding = "ISO-8859-1")
tLaja_DGA = tLaja_DGA.set_index(pd.to_datetime(tLaja_DGA[';Año'].astype(str)+'-'+tLaja_DGA['Mes'].astype(str).str.zfill(2), format='%Y-%m'))

# crear los archivos y escribirlos

pp_armerillo_DGA = extenderMensual(ppMaule_DGA[[';Año','Mes','Armerillo']],1980)
pp_cipreses_DGA =  extenderMensual(ppMaule_DGA[[';Año','Mes','Cipreses']],1980)
pp_melado_DGA = extenderMensual(ppMaule_DGA[[';Año','Mes','Melado']],1980)
pp_abanico_DGA = extenderMensual(ppLaja_DGA[[';Año','Mes','Abanico']],1980)

t_armerillo_DGA = extenderMensual(tMaule_DGA[[';Año','Mes','Armerillo']],1980)
t_diguillin_DGA =  extenderMensual(tLaja_DGA[[';Año','Mes','Diguillin']],1980)

pp_maule_alto = pp_armerillo_DGA[['Year','Month','Armerillo']]
pp_maule_alto['Rio_Cipreses_En_Dasague_Laguna_La_Invernada'] = pp_cipreses_DGA[['Cipreses']]
pp_maule_alto['Rio_Melado_En_La_Lancha_Dga'] = pp_melado_DGA['Melado']
pp_maule_alto.to_csv(model_folder+'\Datos\Precipitaciones\PpMauleAlto.csv', index=False)

pp_laja = pp_abanico_DGA[['Year','Month','Abanico']]
pp_laja.to_csv(model_folder+'\Data\Pp.txt', header=None, index=None, sep=' ')

t_armerillo = t_armerillo_DGA[['Year','Month','Armerillo']]
t_armerillo.to_csv(model_folder+'\Datos\Temperatura\TempMauleAlto_2.csv', index=False)

t_diguillin = t_diguillin_DGA[['Year','Month','Diguillin']]
t_diguillin.to_csv(model_folder+'\Data\T.txt',  header=None, index=None, sep=' ')


#%% Iterar sobre modelos Futuro 6 escenarios

modelos_clima = next(os.walk(path_escenarios))[1]
estaciones_futuras = ['Pabellon','Recoleta_Embalse','Ovalle_Escuela_Agricola','Paloma_Embalse','El_Tome','Punitaqui_DGA','Las_Ramadas','Tascadero','Cogoti_18','Cerro_Calan','San_Jose_De_Maipo_Reten','El_Yeso_Embalse','Rio_Cipreses_En_Dasague_Laguna_La_Invernada','Armerillo','Rio_Melado_En_La_Lancha_Dga','Diguillin','Abanico'] 

modelo = 'IPSL-CM5B-LR'
for modelo in modelos_clima:
        
    # Leer precipitación y temperatura historica GCM 
    path_historico = 'D:\\ARClim\\Extraer_series\\Salidas\\historico_GCM\\'+modelo+r'\historico'
    pp_historico = []
    tmax_historico = []
    tmin_historico = []

    for r, d, f in os.walk(path_historico):
        for file in f:
            if 'pr_' in str(file):
                pp_historico = pd.read_csv(os.path.join(r, file))
                pp_historico.set_index(pd.to_datetime(pp_historico[['Day','Month','Year']]), inplace=True)
            elif 'tasmax' in str(file):
                tmax_historico = pd.read_csv(os.path.join(r, file))
                tmax_historico.set_index(pd.to_datetime(tmax_historico[['Day','Month','Year']]), inplace=True)
            elif 'tasmin' in str(file):
                tmin_historico = pd.read_csv(os.path.join(r, file))
                tmin_historico.set_index(pd.to_datetime(tmin_historico[['Day','Month','Year']]), inplace=True)
    
    tmedia_historico = (tmax_historico + tmin_historico) / 2
    
    # Crear dataframe historico
    
    pp_armerillo_historico = pp_historico[['Day','Month','Year','Armerillo']].resample('MS').sum()
    pp_armerillo_historico = extenderMensualGCM(pp_armerillo_historico,1980)
    
    pp_cipreses_historico = pp_historico[['Day','Month','Year','Rio_Cipreses_En_Dasague_Laguna_La_Invernada']].resample('MS').sum()
    pp_cipreses_historico = extenderMensualGCM(pp_cipreses_historico,1980)

    pp_melado_historico = pp_historico[['Day','Month','Year','Rio_Melado_En_La_Lancha_Dga']].resample('MS').sum()
    pp_melado_historico = extenderMensualGCM(pp_melado_historico,1980)   

    pp_abanico_historico = pp_historico[['Day','Month','Year','Abanico']].resample('MS').sum()
    pp_abanico_historico = extenderMensualGCM(pp_abanico_historico,1980)
        
    t_armerillo_historico = tmedia_historico[['Day','Month','Year','Armerillo']].resample('MS').mean() 
    t_armerillo_historico = extenderMensualGCM(t_armerillo_historico,1980)
    
    t_diguillin_historico = tmedia_historico[['Day','Month','Year','Diguillin']].resample('MS').mean()   
    t_diguillin_historico = extenderMensualGCM(t_diguillin_historico,1980)
    
    
    path_futuro = path_escenarios+'\\'+modelo+r'\future'
    pp_futuro = []
    tmax_futuro = []
    tmin_futuro = []
    
    # Leer precipitación y temperatura
    
    for r, d, f in os.walk(path_futuro):
        for file in f:
            if 'pr_' in str(file):
#                pp_futuro.append(os.path.join(r, file))
                pp_futuro = pd.read_csv(os.path.join(r, file))
                pp_futuro.set_index(pd.to_datetime(pp_futuro[['Day','Month','Year']]), inplace=True)
            elif 'tasmax' in str(file):
                tmax_futuro = pd.read_csv(os.path.join(r, file))
                tmax_futuro.set_index(pd.to_datetime(tmax_futuro[['Day','Month','Year']]), inplace=True)
#                tmax_futuro.append(os.path.join(r, file))      
            elif 'tasmin' in str(file):
                tmin_futuro = pd.read_csv(os.path.join(r, file))
                tmin_futuro.set_index(pd.to_datetime(tmin_futuro[['Day','Month','Year']]), inplace=True)
#                tmin_futuro.append(os.path.join(r, file))     
    
    tmedia_futuro = (tmax_futuro+tmin_futuro)/ 2
    
    # Crear DataFrames con datos climáticos para exportarlos y ser usados en modelo WEAP
    
    pp_armerillo = pp_futuro[['Day','Month','Year','Armerillo']]
    pp_armerillo = extenderFuturo(pp_armerillo_historico,pp_armerillo,5,'pp')
    
    pp_cipreses = pp_futuro[['Day','Month','Year','Rio_Cipreses_En_Dasague_Laguna_La_Invernada']]
    pp_cipreses = extenderFuturo(pp_cipreses_historico,pp_cipreses,5,'pp')

    pp_melado = pp_futuro[['Day','Month','Year','Rio_Melado_En_La_Lancha_Dga']]
    pp_melado = extenderFuturo(pp_melado_historico,pp_melado,5,'pp')    

    pp_abanico = pp_futuro[['Day','Month','Year','Abanico']]
    pp_abanico = extenderFuturo(pp_abanico_historico,pp_abanico,5,'pp') 
        
    t_armerillo = tmedia_futuro[['Day','Month','Year','Armerillo']]  
    t_armerillo = extenderFuturo(t_armerillo_historico,t_armerillo,5,'t')
    
    t_diguillin = tmedia_futuro[['Day','Month','Year','Diguillin']]  
    t_diguillin = extenderFuturo(t_diguillin_historico,t_diguillin,5,'t')
  
    # Borrar inputs antiguos para QA
        
    borrar(model_folder+r'\Datos\Precipitaciones','PpMauleAlto.csv')
    borrar(model_folder+r'\Datos\Temperatura','TempMauleAlto_2.csv')
    borrar(model_folder+r'\Data','T.txt')
    borrar(model_folder+r'\Data','PP.txt')
        
    # Escribir los archivos csv que utilizará el modelo WEAP del escenario climático j
        
    pp_maule_alto = pp_armerillo[['Year','Month','Armerillo']]
    pp_maule_alto['Rio_Cipreses_En_Dasague_Laguna_La_Invernada'] = pp_cipreses['Rio_Cipreses_En_Dasague_Laguna_La_Invernada']
    pp_maule_alto['Rio_Melado_En_La_Lancha_Dga'] = pp_melado['Rio_Melado_En_La_Lancha_Dga']
    pp_maule_alto.to_csv(model_folder+'\Datos\Precipitaciones\PpMauleAlto.csv', index=False)
    
    pp_laja = pp_abanico[['Year','Month','Abanico']]
    pp_laja.to_csv(model_folder+'\Data\Pp.txt', header=None, index=None, sep=' ')
    
    t_armerillo = t_armerillo[['Year','Month','Armerillo']]
    t_armerillo.to_csv(model_folder+'\Datos\Temperatura\TempMauleAlto_2.csv', index=False)
    
    t_diguillin = t_diguillin[['Year','Month','Diguillin']]
    t_diguillin.to_csv(model_folder+'\Data\T.txt', header=None, index=None, sep=' ')


#%% calcular WEAP
   
    WEAP.ActiveArea = 'WEAP_Maule_Laja'
    
    WEAP.ActiveScenario = "historico mas futuro"
    
    WEAP.verbose = 0
    WEAP.Logfile = "WEAPAutomationErrors.txt"
    WEAP.Visible = True
    
    # Calcular el escenario climático j
    
    cambio = random.randint(0,1e4)
    WEAP.Branch(r"\Key\cambio").Variable("Annual Activity Level").Expression = cambio
    
    # Activar el Scenario Explorer con los favoritos definidos según la tabla y exportar
        
    WEAP.LoadOverview(overview)
    
    # Exportar el Scenario Explorer en un solo CSV
        
    WEAP.ExportResults(output_folder+'\\'+str(modelo)+'_ResultadosAltas_historico.csv')
    
    print('El escenario '+modelo+' ha terminado')
    

    
