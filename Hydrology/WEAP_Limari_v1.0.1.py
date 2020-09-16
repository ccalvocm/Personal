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


#%% Limarí

# Carpetas de entrada 
path_historico ='D:\ARClim\Limari\Clima_DGA'

# Carpetas de salida 
    
output_folder = r'D:\ARClim\Limari\Salidas'
ruta_hidrologico = r'D:\WEAP\Weap areas\Limari_Paper_V4'
ruta_modflow = r'D:\WEAP\Weap areas\Limari_WEAP_MODFLOW_DICTUC_SEI'

overview = ['Default','ANCL']
path_escenarios = r'D:\ARClim\Extraer_series\Salidas\futuro'
folder_DICTUC = r'D:\WEAP\Weap areas\Limari_WEAP_MODFLOW_DICTUC_SEI\Datos'

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
    del df_presente['fecha']

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

def reordenarColumnas(df):
    cols = list(df.columns)
    a, b = cols.index('Year'), cols.index('Month')
    cols[b], cols[a] = cols[a], cols[b]
    df = df[cols]
    return df

#%% Leer precipitación y temperatura DGA

ruta_pp_DGA = path_historico+r'\PrecCuencasAltasb.csv'
ruta_t_DGA = path_historico+r'\TempCuencasAltas.csv'

ppLimari_DGA =  pd.read_csv(ruta_pp_DGA)
ppLimari_DGA = ppLimari_DGA.set_index(pd.to_datetime(ppLimari_DGA['Year'].astype(str)+'-'+ppLimari_DGA['Month'].astype(str).str.zfill(2), format='%Y-%m'))

tLimati_DGA = pd.read_csv(ruta_t_DGA)
tLimati_DGA = tLimati_DGA.set_index(pd.to_datetime(tLimati_DGA['Year'].astype(str)+'-'+tLimati_DGA['Month'].astype(str).str.zfill(2), format='%Y-%m'))

# crear los archivos y escribirlos
# Precipitacion
pp_Limari_DGA = extenderMensual(ppLimari_DGA,1980)
pp_Paloma = pp_Limari_DGA[['Year','Month','La Paloma Embalse']]

pp_Limari_DGA.to_csv(ruta_hidrologico+r'\Datos\PrecCuencasAltasb.csv', index=False)
pp_Limari_DGA.to_csv(ruta_modflow+r'\Datos\PrecCuencasAltasb.csv', index=False)
pp_Paloma.to_csv(ruta_modflow+r'\Datos\PrecLaPaloma.csv', index=False)

#temperatura

t_Limari_DGA = extenderMensual(tLimati_DGA,1980)
t_Limari_DGA.to_csv(ruta_hidrologico+r'\Datos\TempCuencasAltas.csv', index=False)
t_Limari_DGA.to_csv(ruta_modflow+r'\Datos\TempCuencasAltas.csv', index=False)


#%% Caudales modelo hidrológico

#filas de todos los caudales

row_q_ancl = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 59, 62, 65, 68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 95, 96, 97, 98, 101, 104, 107, 108, 111, 112, 115, 116, 119, 122, 125, 126, 129, 132, 135, 138, 139, 142, 143, 146]

folder_ANCL = output_folder + r'\ANCL.xlsx'
ancl = pd.read_excel(folder_ANCL)
q_ancl = ancl.iloc[row_q_ancl,2:-1]
q_ancl = q_ancl.transpose() 
q_ancl = q_ancl.set_index(q_ancl[1])    
del q_ancl[1] 

#aportes naturales
an = q_ancl.loc[:,[3,6,9,12,15,18,21,24,27,30,33]]
cl = q_ancl.copy(deep=True)
cl.columns = np.arange(0,len(cl.columns),1)
cl = cl.loc[:,0:34]
cl.loc[:] = 0

#cuencas laterales cl[0] = CL_1
cl[0] = q_ancl[37]-q_ancl[36]
cl[1] = q_ancl[41]-q_ancl[40]
cl[2] = q_ancl[45]-q_ancl[44]
cl[3] = q_ancl[49]-q_ancl[48]
cl[4] = q_ancl[53]-q_ancl[52]+q_ancl[56]
cl[5] = q_ancl[59]
cl[6] = q_ancl[62]
cl[7] = q_ancl[65]
#cl9 es cero
cl[9] = q_ancl[69]-q_ancl[68]
cl[10] = q_ancl[73]-q_ancl[72]
cl[11] = q_ancl[77]-q_ancl[76]
cl[12] = q_ancl[81]-q_ancl[80]
cl[13] = q_ancl[85]-q_ancl[84]
cl[14] = q_ancl[89]-q_ancl[88]
cl[15] = q_ancl[92]
cl[16] = q_ancl[96]-q_ancl[95]+q_ancl[98]-q_ancl[97]
cl[17] = q_ancl[101]
cl[18] = q_ancl[104]
# El modelo hidrológico no considera caudales en cl 20 al 23
cl[23] = q_ancl[108]-q_ancl[107]
cl[24] = q_ancl[112]-q_ancl[111]
cl[25] = q_ancl[116]-q_ancl[115]
cl[26] = q_ancl[119]
#cl 28_1
cl[27] = q_ancl[122]
#cl 28_2
cl[28] = q_ancl[126]-q_ancl[125]
#cl 29
cl[29] = q_ancl[129]
cl[30] = q_ancl[132]
cl[31] = q_ancl[135]
cl[32] = q_ancl[139]-q_ancl[138]
cl[33] = q_ancl[143]-q_ancl[142]
cl[34] = q_ancl[146]

cl = cl[cl > 0]
cl = cl.fillna(0)
an_cl_final = pd.concat([an, cl], axis=1)
an_cl_final.insert(0, 'Year', an_cl_final.index.year)
an_cl_final.insert(1, 'Month', an_cl_final.index.month)
an_cl_final.to_csv(folder_DICTUC+r'\AN_Q.csv',index = False)


#%% Iterar sobre modelos Futuro 6 escenarios

modelos_clima = next(os.walk(path_escenarios))[1]
estaciones_futuras = ['Pabellon','Recoleta_Embalse','Ovalle_Escuela_Agricola','Paloma_Embalse','El_Tome','Punitaqui_DGA','Las_Ramadas','Tascadero','Cogoti_18','Cerro_Calan','San_Jose_De_Maipo_Reten','El_Yeso_Embalse','Rio_Cipreses_En_Dasague_Laguna_La_Invernada','Armerillo','Rio_Melado_En_La_Lancha_Dga','Diguillin','Abanico'] 
estaciones_pp = ['Pabellon','Las_Ramadas','Tascadero','Cogoti_18','Recoleta_Embalse','Paloma_Embalse','El_Tome','Punitaqui_DGA','Ovalle_Escuela_Agricola']
# rellenar las estaciones que no se usan para el archivo PrecCuencasAltasb.csv
estaciones_pp_exportadas = ['Pabellon','Pabellon','Pabellon','Las_Ramadas','Tascadero','Cogoti_18','Cogoti_18','Recoleta_Embalse','Paloma_Embalse','El_Tome','Punitaqui_DGA','Ovalle_Escuela_Agricola']

modelo = 'ACCESS1-3'
for modelo in modelos_clima:
        
    # Leer precipitación y temperatura historica GCM 
    path_historico = 'D:\\ARClim\\Extraer_series\\Salidas\\historico_GCM\\'+modelo+r'\historico'
    pp_historico = []
    pp_subcuencas_irrigacion = []
    pp_subcuencas_control = []
    pp_estaciones_control = []
    tmax_historico = []
    tmin_historico = []

    for r, d, f in os.walk(path_historico):
        for file in f:
            print(file)
            if 'pr_per_control_GCM' in str(file):
                pp_historico = pd.read_csv(os.path.join(r, file))
                pp_historico.set_index(pd.to_datetime(pp_historico[['Day','Month','Year']]), inplace=True)
#            elif 'pr_hist_Limari_irrigation' in str(file):
#                pp_historico_irrigacion = pd.read_csv(os.path.join(r, file))
#                pp_historico_irrigacion.set_index(pd.to_datetime(pp_historico[['Day','Month','Year']]), inplace=True)             
            elif 'tasmax_per_control_' in str(file):
                tmax_historico = pd.read_csv(os.path.join(r, file))
                tmax_historico.set_index(pd.to_datetime(tmax_historico[['Day','Month','Year']]), inplace=True)
            elif 'tasmin_per_control_' in str(file):
                tmin_historico = pd.read_csv(os.path.join(r, file))
                tmin_historico.set_index(pd.to_datetime(tmin_historico[['Day','Month','Year']]), inplace=True)
    
    tmedia_control = (tmax_historico + tmin_historico) / 2
    
    # Crear dataframe historico

    subcuencas_clima = ['AN-07','CL-12','CL-11','AN-01','AN-03','AN-04','AN-09','CL-05','CL-20','CL-21','CL-22','CL-06','CL-23','CL-26','AN-10','AN-11','CL-13','CL-15','CL-04','CL-09','CL-08','CL-07','CL-32','CL-31','CL-30','CL-27','CL-19','CL-34','CL-01','AN-02','CL-03','CL-02','CL-18','CL-10','CL-29','CL-33','CL-25','CL-24','CL-16','CL-17','AN-08','AN-05','AN-06','CL-281']
    catchments_clima = ['Los Molles Alto 8','HurtadoIntermedioSanAgustin_Angostura','HurtadoAntesRecoleta Riego','Rapel Riego','MostazalDesembocadura Riego','GrandeAMostazal Riego arr. GC','GrandePSJuan Riego','CogotiEmbalseRiego','Combarbala Riego','Pama Intermedio Riego','AfluentePalomaNorte Riego','R. PalquiFRU','R. CauchilFRU','R. Huatulame antes BCCFRU','R. HuatulameFRU','R. Paloma PonienteFRU','R. Hurtado OvalleFRU','R. TalhuenFRU','R. Villalon IngenioFRU','R. Canal VillalonFRU','R. M. PalomaFRU','R. Grande A. PalomaFRU','R. M. CogotiFRU','R. Camarico _HualliFRU','R. M. Cogoti Pun.FRU','R. Limari_IngenioFRU','R. Cam ClindoANU','R. Der Cogoti 1FRU','R. Cam MatrizANU','R. Der Cogoti 2FRU','R. Camarico PunitaquiFRU','R. Canal PunitaquiFRU','R. TabaliFRU','Punitaqui S Chalinga riego','R. Limari abajo IngenioFRU','Punitaqui antes Limari riego','R. Cam ClindoFRU','R. Cam MatrizFRU','R. TalhuenANU','R. Limari_IngenioANU','R. Villalon IngenioANU','R. Der Cogoti 1ANU','R. M. Cogoti Pun.ANU','R. Camarico_HualliANU','R. M. CogotiANU','R. Grande A. PalomaANU','R. M. PalomaANU','R. Hurtado OvalleANU','R. Paloma PonienteANU','R. Der Cogoti 2ANU','R. Canal PunitaquiANU','R. Camarico PunitaquiANU','R. TabaliANU','R. Limari abajo IngenioANU','R. Canal VillalonANU','R. CauchilANU','R. PalquiANU','R. HuatulameANU','R. Huatulame antes BCCANU','GrandeAMostazal Riego ab. GC']

    pp_subcuencas_control =  pp_historico[['Day','Month','Year']+subcuencas_clima].resample('MS').sum()
    pp_subcuencas_control = extenderMensualGCM(pp_subcuencas_control,1980)
    pp_subcuencas_control = pp_subcuencas_control[['fecha','Month','Year']+subcuencas_clima+['Day']]

    pp_subcuencas_irrigacion =  pp_historico[['Day','Month','Year']+catchments_clima].resample('MS').sum()
    pp_subcuencas_irrigacion = extenderMensualGCM(pp_subcuencas_irrigacion,1980)
    pp_subcuencas_irrigacion = pp_subcuencas_irrigacion[['fecha','Month','Year']+catchments_clima+['Day']]
    
    pp_estaciones_control =  pp_historico[['Day','Month','Year']+estaciones_pp].resample('MS').sum()
    pp_estaciones_control = extenderMensualGCM(pp_estaciones_control,1980)   
    pp_estaciones_control = pp_estaciones_control[['fecha','Month','Year']+estaciones_pp+['Day']]

    t_control = tmedia_control[['Day','Month','Year','Las_Ramadas','Paloma_Embalse']].resample('MS').mean()
    t_control = extenderMensualGCM(t_control,1980)    
    
    path_futuro = path_escenarios+'\\'+modelo+r'\future'
    pp_futuro = []
    tmax_futuro = []
    tmin_futuro = []
    
    # Leer precipitación y temperatura
    
    for r, d, f in os.walk(path_futuro):
        for file in f:
            if 'pr_per_fut_' in str(file):
#                pp_futuro.append(os.path.join(r, file))
                pp_futuro = pd.read_csv(os.path.join(r, file))
                pp_futuro.set_index(pd.to_datetime(pp_futuro[['Day','Month','Year']]), inplace=True)
            elif 'tasmax_per_fut_' in str(file):
                tmax_futuro = pd.read_csv(os.path.join(r, file))
                tmax_futuro.set_index(pd.to_datetime(tmax_futuro[['Day','Month','Year']]), inplace=True)
#                tmax_futuro.append(os.path.join(r, file))      
            elif 'tasmin_per_fut_' in str(file):
                tmin_futuro = pd.read_csv(os.path.join(r, file))
                tmin_futuro.set_index(pd.to_datetime(tmin_futuro[['Day','Month','Year']]), inplace=True)
#                tmin_futuro.append(os.path.join(r, file))     
    
    tmedia_futuro = (tmax_futuro+tmin_futuro)/ 2
    
    # Crear DataFrames con datos climáticos para exportarlos y ser usados en modelo WEAP
    
    pp_subcuencas =  pp_futuro[['Day','Month','Year']+subcuencas_clima]
    pp_subcuencas = extenderFuturo(pp_subcuencas_control,pp_subcuencas,5,'pp')
    pp_subcuencas = reordenarColumnas(pp_subcuencas) 

    pp_irrigacion =  pp_futuro[['Day','Month','Year']+catchments_clima]
    pp_irrigacion = extenderFuturo(pp_subcuencas_irrigacion,pp_irrigacion,5,'pp') 
    pp_irrigacion = reordenarColumnas(pp_irrigacion) 

    pp_estaciones =  pp_futuro[['Day','Month','Year']+estaciones_pp]
    pp_estaciones = extenderFuturo(pp_estaciones_control,pp_estaciones,5,'pp')   
    pp_estaciones_final = pp_estaciones[['Year','Month']+estaciones_pp_exportadas]

    temp = tmedia_futuro[['Day','Month','Year','Las_Ramadas','Paloma_Embalse']]
    temp = extenderFuturo(t_control,temp,5,'t')
    temp = reordenarColumnas(temp)    
    
    # Escribir los archivos csv que utilizará el modelo WEAP del escenario climático j
    
    pp_estaciones_final.to_csv(ruta_hidrologico+'\Datos\PrecCuencasAltasb.csv', index=None)
    temp.to_csv(ruta_hidrologico+'\Datos\TempCuencasAltas.csv', index=None)

#    pp_subcuencas.to_csv()
#    pp_irrigacion
#    pp_estaciones
        
  
#%% calcular WEAP

    WEAP.ActiveArea = 'Limari_Paper_V4'
    WEAP.ActiveScenario = "Referencia"
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
    