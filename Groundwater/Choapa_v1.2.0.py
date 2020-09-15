# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author: CCCM

"""

# Limpiar entorno
#%reset -f

import pandas as pd
import math
import statsmodels.api as sm
import numpy as np
import win32com.client
import time
import os, shutil
import matplotlib.pyplot as plt
import random
import fiscalyear
fiscalyear.START_MONTH = 4

# Preámbulo

es_futuro = 1 #Flag para cambiar a modelo de control o futuro
overview = ['Default']
path_escenarios = r'D:\ARClim\Extraer_series\Salidas\futuro'
ruta_WEAP = r'D:\WEAP\Weap areas\Choapa_WEAP_MODFLOW_DICTUC_SEI_2019_corrales\Datos'
if es_futuro > 0:
    ruta_WEAP = r'D:\WEAP\Weap areas\Choapa_WEAP_MODFLOW_DICTUC_SEI_2019_corrales_futuro\Datos'

output_folder = r'D:\ARClim\Choapa\Salidas'
WEAP = win32com.client.Dispatch('WEAP.WEAPApplication')

time.sleep(5)

#% funciones

def agnohidrologico(year_,month_):
    cur_dt = fiscalyear.FiscalDate(year_, month_, 1) 
    retornar = cur_dt.fiscal_year - 1
    return retornar
    
   
def extenderFuturo(df_presente,df_futuro,years):
        
#    df_futuro['Mo'] = df_futuro.index.month
#    df_futuro = df_futuro.drop(['Day'], axis = 1)
#    df_presente = df_presente.drop(['Day'], axis = 1)
    
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

    return reordenarColumnas(df_presente)

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
    
#%% Calcular correlaciones (solo 1 vez)

AN_Q = []
ruta = r'G:\Mi unidad\2020 RRHH ARCLIM COMPARTIDA\Resultados\Resultados Choapa\Correlaciones\AN_Q_rev.xlsx'
AN_Q = pd.read_excel(ruta, sheetname= 'Todos los datos', skiprows = 1)

for j, row in AN_Q.iterrows():
    year0 = AN_Q.loc[j,'AGNO_CALEND']
    month0 = AN_Q.loc[j,'MES_No'] 
    agno = agnohidrologico(year0,month0)
    AN_Q.loc[j,'AGNO_CALEND'] = agno
    
AN_Q = AN_Q.set_index(pd.to_datetime(AN_Q['AGNO_CALEND'].astype(str)+'-'+AN_Q['MES_No'].astype(str).str.zfill(2),  format='%Y-%m'))


meses = [4,5,6,7,8,9,10,11,12,1,2,3]

coef_m_mensuales = pd.DataFrame( index = meses, columns = AN_Q.columns[3:])
coef_n_mensuales = pd.DataFrame( index = meses, columns = AN_Q.columns[3:])
coef_r2_mensuales = pd.DataFrame( index = meses, columns = AN_Q.columns[3:])

coefs_m_anuales = pd.DataFrame(index = [0], columns = AN_Q.columns[3:])
coefs_n_anuales = pd.DataFrame(index = [0], columns = AN_Q.columns[3:])
coefs_r2_anuales = pd.DataFrame(index = [0], columns = AN_Q.columns[3:])

for i in range(12):

    mask = AN_Q[AN_Q['MES_No'] == meses[i]]

    AN_Q1 = mask['AN-01']
    AN_Q1 = sm.add_constant(AN_Q1)

    for col in mask.columns[3:]:

        subcuenca_col =  mask[col]
   
        model = sm.OLS(subcuenca_col, AN_Q1)
        results = model.fit()
        
        n = results.params[0]
        m = results.params[1]
        r2 = results.rsquared
        print(r2)
#        if r2 > 0:
        coef_m_mensuales.loc[meses[i]][col] = results.params[1]
        coef_n_mensuales.loc[meses[i]][col] = results.params[0]
        coef_r2_mensuales.loc[meses[i]][col] = results.rsquared
#        else:        
            
        AN_Q_anual = AN_Q.resample('YS').mean()
        
        AN_Q1_anual = AN_Q_anual['AN-01']
        AN_Q1_anual = sm.add_constant(AN_Q1_anual)

        subcuenca_anual = AN_Q_anual[col]
        
        model = sm.OLS(subcuenca_anual, AN_Q1_anual)
        results_anual = model.fit()
        n_anual = results_anual.params[0]
        m_anual = results_anual.params[1]
        r2_anual = results_anual.rsquared
        
        coefs_m_anuales.loc[0][col] = m_anual
        coefs_n_anuales.loc[0][col] = n_anual
        coefs_r2_anuales.loc[0][col] = r2_anual
                        
#coef_m_mensuales.to_csv('coef_m_mensuales.csv',index_label = 'Mes' )
#coef_n_mensuales.to_csv('coef_n_mensuales.csv',index_label = 'Mes' )
#coef_r2_mensuales.to_csv('coef_r2_mensuales.csv',index_label = 'Mes' )
#
#coefs_m_anuales.to_csv('coef_m_anuales.csv',index = False)
#coefs_n_anuales.to_csv('coef_n_anuales.csv',index = False)
#coefs_r2_anuales.to_csv('coef_r2_anuales.csv',index = False)

#%% Completar areas futuras - ESTO YA ESTÁ HECHO

folder = r'D:\WEAP\Weap areas\Choapa_WEAP_MODFLOW_DICTUC_SEI_2019_corrales_futuro\Datos\AREAS_CULTIVOS'

for r, d, f in os.walk(folder):
    for file in f:
        if "._" not in file:
            archivo = pd.read_csv(r+'\\'+file ) 
            indice = 2
            tecnificado = 0
            if "TECNIFICADO" in file and "NO_TECNIFICADO" not in file:
                indice = 1
                tecnificado = 1
            year1 = []
            for i in range(1,12):
                year1 = archivo.loc[indice].tolist()
                year1[0] = str(int(int(year1[0]) -1))
                archivo.loc[-1] = year1
                as_list = archivo.index.tolist()
                indice = (as_list[1]+as_list[2])/2.0
                if tecnificado == 1:
                    indice = (as_list[0]+as_list[1])/2.0
                as_list[-1] = indice
                archivo.index = as_list   
                archivo = archivo.sort_index()
            if not "TECNIFICADO" in file:
                archivo = archivo.drop(indice)
            else:
                None
            archivo.columns = archivo.columns.str.replace('Unnamed.*', 'Year')
            if "TECNIFICADO" in file and "NO_TECNIFICADO" not in file:
                archivo.columns = archivo.columns.str.replace('0', 'Year')
            archivo = archivo.replace(np.nan, 0, regex=True)
            archivo["Year"].astype(int)
            
            for i in range(1,20):
                indice_f = archivo.index[-1]
                yearf = archivo.loc[indice_f].tolist()
                yearf[0] = str(int(int(yearf[0]) + 1))
                archivo.loc[indice_f+1] = yearf
#                as_list = archivo.index.tolist()
#                indice = as_list[-1]+1
#                as_list[-1] = indice
#                archivo.index = as_list   
#                archivo = archivo.sort_index()
            
            archivo = archivo.replace(0, np.nan, regex=True)
            archivo.to_csv(r+'\\'+file,index = False)
            
    
#%% Desmarques
            
# Calcular Q50 REAL
ruta_Q_hidrologico = r'G:\Mi unidad\2020 RRHH ARCLIM COMPARTIDA\Resultados\Resultados Choapa\Caudales\Modelo Hidrologia Cabecera'

file = ruta_Q_hidrologico + r'\\ARCLIM-ResultadosCordillera-Choapa_Hist-DGA.xlsx'
Q_hidro = pd.read_excel(file, sep = ',', sheetname = "WEAP Export - PEGAR Aqui", skiprows=1)
Choapa_Cuncumen = pd.DataFrame(Q_hidro.loc[2][1:-1]).astype(float)
Choapa_Cuncumen.columns = ['Q']
Choapa_Cuncumen_sort = Choapa_Cuncumen.sort_values(by='Q',ascending=False)
Choapa_Cuncumen = Choapa_Cuncumen
Choapa_Cuncumen = pd.DataFrame(Choapa_Cuncumen_sort)
n = Choapa_Cuncumen.size
Choapa_Cuncumen.insert(0, 'rank', range(1, 1 + n))
Choapa_Cuncumen["probability"] = (
    1- (n - Choapa_Cuncumen["rank"] + 1) / (n + 1))

Q50 = Choapa_Cuncumen.iloc[(Choapa_Cuncumen["probability"]-0.5).abs().argsort()[:2]]
Q50 = Q50["Q"].iloc[0]+(Q50["Q"].iloc[1]-Q50["Q"].iloc[0])*(.5-Q50["probability"].iloc[0])/(Q50["probability"].iloc[1]-Q50["probability"].iloc[0])


#%% Iterar sobre modelos Futuro 6 escenarios de control

modelos_clima = next(os.walk(path_escenarios))[1]
catchments_orden = ['NOZR_CL050FRU','NOZR_CL13ANU','NOZR_CL13FRU','ZR_01ANU','ZR_01FRU','ZR_02ANU','ZR_02FRU','ZR_03ANU','ZR_03FRU','ZR_04ANU','ZR_04FRU','ZR_050ANU','ZR_050FRU','ZR_051ANU','ZR_051FRU','ZR_052ANU','ZR_052FRU','ZR_06ANU','ZR_06FRU','ZR_07ANU','ZR_07FRU','ZR_08ANU','ZR_08FRU','ZR_09ANU','ZR_09FRU','ZR_10ANU','ZR_10FRU','ZR_11ANU','ZR_11FRU','ZR_12ANU','ZR_12FRU','ZR_13ANU','ZR_13FRU','ZR_14ANU','ZR_14FRU','ZR_15ANU','ZR_15FRU','ZR_16ANU','ZR_16FRU','ZR_17ANU','ZR_17FRU','ZR_18ANU','ZR_18FRU','ZR_19ANU','ZR_19FRU','ZR_20ANU','ZR_20FRU','ZR_21ANU','ZR_21FRU','ZR_22ANU','ZR_22FRU','ZR_23ANU','ZR_23FRU','ZR_24ANU','ZR_24FRU','ZR_25ANU','ZR_25FRU','ZR_26ANU','ZR_26FRU','ZR_27ANU','ZR_27FRU','ZR_28ANU','ZR_28FRU','ZR_29','ZR_Canelillo']

modelo = 'IPSL-CM5A-MR'    

for modelo in modelos_clima:

    # Calcular Q/Q50 y desmarques serie climatica 
    
    for r, d, f in os.walk(ruta_Q_hidrologico):
            for file in f:
                if modelo in file:
                    Choapa_Cuncumen_GCM = pd.read_csv(r+'\\'+file, sep = ',', skiprows = 1).loc[1][1:-1].astype(float)
                    CHOAPA_desmarques = Choapa_Cuncumen_GCM/Q50
                    CHOAPA_desmarques[CHOAPA_desmarques > 1.] = 1.    
                    CHOAPA_desmarques = pd.DataFrame(CHOAPA_desmarques)
                    CHOAPA_desmarques = CHOAPA_desmarques.set_index(pd.to_datetime(CHOAPA_desmarques.index))
                    
    desmarques_gcm = pd.read_csv(ruta_WEAP +'\\'+'DESMARQUES_Q_original.csv')
    acciones = desmarques_gcm.loc[0]
    desmarques_gcm = desmarques_gcm.iloc[2:].copy()
    agnos = desmarques_gcm.ix[:,0]
    meses = desmarques_gcm.ix[:,1]
    desmarques_gcm = desmarques_gcm.set_index(pd.to_datetime(agnos.astype(str)+'-'+meses.astype(str).str.zfill(2), format='%Y-%m'))
    
    for index, row in desmarques_gcm.iterrows():
        if index in CHOAPA_desmarques.index:
            desmarques_gcm.at[index,'1'] = CHOAPA_desmarques.loc[index].values[0]
            for col in desmarques_gcm.columns:
    #                for i in range(len(acciones)):
                if not "Unnamed" in col:
                    if int(col) > 4 and index.year > 2000: #año consistente con el historico de DICTUC
                        accion_col = float(acciones.loc[col])/1000.
                        desmarques_gcm.at[index, col] = desmarques_gcm.at[index,'1'].copy()*accion_col
    
    desmarques_gcm.loc[pd.Timestamp('1900-01-01T12')] = acciones
    desmarques_gcm = desmarques_gcm.sort_index()
    desmarques_gcm.to_csv(ruta_WEAP+'\\'+'DESMARQUES_Q.csv', index = False)


    path_historico = 'D:\\ARClim\\Extraer_series\\Salidas\\historico_GCM\\'+modelo+r'\historico' 
    pp_control = []
    pp_control_orden = []

    for r, d, f in os.walk(path_historico):
        for file in f:
            if 'prChoapa_catchments_control' in str(file):
                pp_control = pd.read_csv(os.path.join(r, file))
                pp_control.set_index(pd.to_datetime(pp_control[['Day','Month','Year']]), inplace=True)
    
    pp_control_orden = pp_control[catchments_orden]
    pp_control_orden = pp_control_orden.resample('MS').sum()
    pp_control_orden.insert(loc=0,column = 'Year', value = pp_control_orden.index.year)
    pp_control_orden.insert(loc=1,column = 'Month', value = pp_control_orden.index.month)
    
    pp_control_ext = extenderMensualGCM(pp_control_orden,1980)
        
    path_futuro = path_escenarios+'\\'+modelo+r'\future'
    pp_futuro = []
    pp_futuro_orden = []
    
    # Leer precipitación y temperatura
    
    for r, d, f in os.walk(path_futuro):
        for file in f:
            if 'prChoapa_catchments_futuro' in str(file):
                pp_futuro = pd.read_csv(os.path.join(r, file))
                pp_futuro.set_index(pd.to_datetime(pp_futuro[['Day','Month','Year']]), inplace=True)
      
    pp_futuro_orden = pp_futuro[catchments_orden]
    pp_futuro_orden = pp_futuro_orden.resample('MS').sum()
    pp_futuro_orden.insert(loc=0,column = 'Year', value = pp_futuro_orden.index.year)
    pp_futuro_orden.insert(loc=1,column = 'Month', value = pp_futuro_orden.index.month)
    
    pp_futuro_orden = extenderFuturo(pp_control_ext,pp_futuro_orden,5)

    pp_futuro_orden.to_csv(ruta_WEAP+'\\'+'Precip_VIC.csv', index = False)
    
        
    #%% Generar caudales sintéticos (por escenario climático)
    
    ruta_temp = r'G:\Mi unidad\2020 RRHH ARCLIM COMPARTIDA\Resultados\Resultados Choapa\Correlaciones'
    
    cuencas_m = ['AN-02','AN-03','AN-04','AN-05','AN-06','AN-08','AN-09','CL-01','CL-02','CL-050','CL-09','CL-16','CL-17','CL-18']
    cuencas_a = ['AN-07','AN-10','CL-03','CL-04','CL-051','CL-052','CL-06','CL-07','CL-08','CL-10','CL-11','CL-12','CL-13','CL-14','CL-15','CL-19','CL-20','CL-21','CL-22','CL-23','CL-24','CL-25']
    
    ruta_Q_hidrologico = r'G:\Mi unidad\2020 RRHH ARCLIM COMPARTIDA\Resultados\Resultados Choapa\Caudales\Modelo Hidrologia Cabecera'
    ruta_template = r'G:\Mi unidad\2020 RRHH ARCLIM COMPARTIDA\Resultados\Resultados Choapa\Correlaciones\Template.xlsx'
    
    template = pd.read_excel(ruta_template, sheetname = "Hoja1", skiprows=1)
    razones_template = template.loc[325:]
    razones_template.columns = template.columns
    razones_template.set_index('MES_No', inplace=True)
    
    for r, d, f in os.walk(ruta_Q_hidrologico):
        for file in f:
            if modelo in file:
                Q_cabecera = pd.read_csv(r+'\\'+file, sep = ',', skiprows = 1)
                
    Q_AN1 = Q_cabecera.loc[9][1:-1]
    fechas =  pd.to_datetime(Q_cabecera.columns[1:-1])
    Q_AN1 = pd.DataFrame(Q_AN1).astype(float)
    Q_AN1 = Q_AN1.set_index(fechas)
    Q_AN1.columns = ['QAN1']
    if es_futuro > 0:
        Q_anual_1 = Q_AN1[(Q_AN1.index > '2034-12-31') & (Q_AN1.index <= '2064-12-31')].mean() #Periodo FUTURO
    else:
        Q_anual_1 = Q_AN1[(Q_AN1.index > '1979-12-31') & (Q_AN1.index <= '2009-12-31')].mean()    
    years = []
    months = []
    for i in range(len(fechas)):
        years.append(fechas[i].year)
        months.append(fechas[i].month)
    
    AN_Q_sintetico = pd.DataFrame(index = fechas, columns = AN_Q.columns)
    AN_Q_sintetico["AGNO_CALEND"] = years
    AN_Q_sintetico["MES_No"] = months
    
    Q = 0
    
    for index, row in AN_Q_sintetico.iterrows():
        for col in AN_Q_sintetico.columns:
            mes = row["MES_No"]
            if col in cuencas_m:
                coef_m_mens = coef_m_mensuales.loc[mes][col]
                coef_n_mens = coef_n_mensuales.loc[mes][col]
                Q = float(Q_AN1.loc[index])*coef_m_mens+coef_n_mens
                Q = max(Q,0)
                AN_Q_sintetico = AN_Q_sintetico.copy()
                AN_Q_sintetico.at[index,col] = Q
            elif col in cuencas_a:
                coef_m_anu = coefs_m_anuales[col].values
                coef_n_anu = coefs_n_anuales[col].values
                Q_anual_2 = Q_anual_1*coef_m_anu+coef_n_anu
                razon = razones_template.loc[mes][col]
                razon = razon.copy()
                Q = Q_anual_2*razon
                Q = max(Q.values,0)
                AN_Q_sintetico = AN_Q_sintetico.copy()
                AN_Q_sintetico.at[index,col] = Q[0]
                
    AN_Q_sintetico['AN-01'] = Q_AN1      
    AN_Q_sintetico.to_csv(ruta_WEAP+'\\AN_Q.csv', index = False)
    print(ruta_WEAP)

  
#%% calcular WEAP

    WEAP.ActiveArea = 'Choapa_WEAP_MODFLOW_DICTUC_SEI_2019_corrales_futuro'
    WEAP.ActiveScenario = "Embalses libres"
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
            
        WEAP.ExportResults(output_folder+'\\'+str(modelo)+'\\'+overview[i]+'_GCM_ResultadosAltas_futuro.csv')
        
    print('El escenario '+modelo+' ha terminado')
    

#ruta_modelo = r''

#AN_Q_sintetico.to_csv(ruta_temp+'\\AN_Q_sintetico.csv',index = None)






