# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author: CCCM

"""

#%% Preambulo

def limpiar_kernel():    
    try:
        from IPython import get_ipython
        get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass
    
limpiar_kernel()

import pandas as pd
import math
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

import fiscalyear
fiscalyear.START_MONTH = 4

#% funciones

def agnohidrologico(year_,month_):
    cur_dt = fiscalyear.FiscalDate(year_, month_, 1) 
    retornar = cur_dt.fiscal_year - 1
    return retornar


def regresion(x_,y_):
    X = sm.add_constant(x_)
    resultados_fit = sm.OLS(y_,X,missing='drop').fit()
    N = resultados_fit.params[0]
    M = resultados_fit.params[1]
    R2 = resultados_fit.rsquared
    return [M,N,R2]
    
#%%    
    
def main():    

#%%    
#    ruta_GitHub = r'D:\GitHub'
    ruta_GitHub = r'C:\Users\ccalvo\Documents\GitHub'

    ruta_Q = ruta_GitHub+r'\Analisis-Oferta-Hidrica\DGA\datosDGA\Q\Maule\Q_Maule_1900-2020.csv'
    Q_daily = pd.read_csv(ruta_Q, index_col = 0, sep =";")
    Q_daily.index = pd.to_datetime(Q_daily.index)
    
    # ver entrada gráfica
    Q_daily.plot()
    
    #meses
    
    meses = [4,5,6,7,8,9,10,11,12,1,2,3]


  #%%Crear indice de fechas
    
    #Convertir años a int y calcular frecuencia de datos para mapa de colores

    data = Q_daily.notnull().astype('int')
    data = data.groupby(Q_daily.index.year)  
    data_anual = data.aggregate(np.sum)
    data_anual = data_anual/(365*0.8)  
    data_anual = data_anual.apply(lambda x: [y if y < 1 else 1 for y in x])
    data_anual = data_anual.transpose()
   
    data_anual = data_anual.sort_index()
    estaciones_minimas = pd.DataFrame(data_anual.sum(axis=1), columns = ['registro'])
    estaciones_minimas = estaciones_minimas[estaciones_minimas['registro']>= 15]
    
    Q_daily_filtradas = Q_daily[estaciones_minimas.index]
    
    coef_m_mensuales = pd.DataFrame( index = meses, columns = Q_daily_filtradas.columns)
    coef_n_mensuales = pd.DataFrame( index = meses, columns = Q_daily_filtradas.columns)
    coef_r2_mensuales = pd.DataFrame( index = meses, columns = Q_daily_filtradas.columns)
   

    for j, row in Q_daily_filtradas.iterrows():
        yr = j.year
        mnth = j.month
        hydro_yr = agnohidrologico(yr,mnth)
        # Q_daily.loc[j,'AGNO_HIDRO'] = hydro_yr
    
    # Est con mejor correlación diaria
    correl = Q_daily_filtradas.corr()
    correl = correl.replace(1,-9999)
    idx = correl.idxmax()
    r = correl.max()
    Q_daily_mon = Q_daily_filtradas.groupby(Q_daily_filtradas.index.month)
    
    for indice in idx.index:
        print(indice)
        for mes in meses:
            y = Q_daily_mon[indice].apply(list).loc[mes] #mes 1
            est_indep = idx.loc[indice]
            x =  Q_daily_mon[est_indep].apply(list).loc[mes]  #mes 1
            try:
                M, N, R2 = regresion(x,y)
                coef_m_mensuales.loc[mes][indice] = M
                coef_n_mensuales.loc[mes][indice] = N
                coef_r2_mensuales.loc[mes][indice] = R2
            except:
                print('No hay datos para el mes '+str(mes))
        
        for index, row in Q_daily_filtradas.iterrows():
            for col in Q_daily_filtradas.columns:
                mes = index.month
                m = coef_m_mensuales.loc[mes][col]
                n = coef_n_mensuales.loc[mes][col]
                Q_x =  Q_daily_filtradas.loc[index,idx.loc[col]]
                Q_daily_filtradas.loc[index,col] = Q_x*m+n
#                r2 = coef_r2_mensuales.loc[mes][index]

                
            
            for col in AN_Q_sintetico.columns:
                mes = row["MES_No"]
                if col in cuencas_m:
                    coef_m = coef_m_mensuales.loc[mes][col]
                    coef_n = coef_n_mensuales.loc[mes][col]
                    Q = Q_AN1.loc[index]*coef_m+coef_n
                    Q = max(Q.values,0)
                    AN_Q_sintetico = AN_Q_sintetico.copy()
                    AN_Q_sintetico.at[index,col] = Q
                elif col in cuencas_a:
                    coef_m = coefs_m_anuales[col].values
                    coef_n = coefs_n_anuales[col].values
                    Q_anual_1 = Q_AN1.mean()
                    Q_anual_2 = Q_anual_1*coef_m+coef_n
                    razon = razones_template.loc[mes][col]
                    razon = razon.copy()
                    Q = Q_anual_2*razon
                    Q = max(Q.values,0)
                    AN_Q_sintetico = AN_Q_sintetico.copy()
                    AN_Q_sintetico.at[index,col] = Q
            
        
       
        
    # regresiones diarias por mes
    model = sm.OLS(subcuenca_col, AN_Q1)
    results = model.fit()
    
    n = results.params[0]
    m = results.params[1]
    
    # La regresión se hará por mes        
    idx = correl.groupby('Fecha').transform(max) == correl.index
    max_r = correl.groupby('Fecha').max() 
    


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

#%%

ruta_temp = r'G:\Mi unidad\2020 RRHH ARCLIM COMPARTIDA\Resultados\Resultados Choapa\Correlaciones'

cuencas_m = ['AN-02','AN-03','AN-04','AN-05','AN-06','AN-08','AN-09','CL-01','CL-02','CL-050','CL-09','CL-16','CL-17','CL-18']
cuencas_a = ['AN-07','AN-10','CL-03','CL-04','CL-051','CL-052','CL-06','CL-07','CL-08','CL-10','CL-11','CL-12','CL-13','CL-14','CL-15','CL-19','CL-20','CL-21','CL-22','CL-23','CL-24','CL-25']

ruta_Q_hidrologico = r'G:\Mi unidad\2020 RRHH ARCLIM COMPARTIDA\Resultados\Resultados Choapa\Caudales\Modelo Hidrologia Cabecera'
ruta_template = r'G:\Mi unidad\2020 RRHH ARCLIM COMPARTIDA\Resultados\Resultados Choapa\Correlaciones\Template.xlsx'

template = pd.read_excel(ruta_template, sheetname = "Hoja1", skiprows=1)
razones_template = template.loc[325:]
razones_template.columns = template.columns
razones_template.set_index('MES_No', inplace=True)

test = r'ARCLIM-ResultadosCordillera-Choapa_Hist-DGA.xlsx'

Q_cabecera = pd.read_excel(ruta_Q_hidrologico+'\\'+test, sep = ',', sheetname = "WEAP Export - PEGAR Aqui")
Q_AN1 = Q_cabecera.loc[11][1:-1]
fechas =  Q_cabecera.loc[1][1:-1]
Q_AN1 = pd.DataFrame(Q_AN1)
Q_AN1 = Q_AN1.set_index(fechas)

years = []
months = []
for i in range(len(fechas)):
    years.append(fechas[i].year)
    months.append(fechas[i].month)

AN_Q_sintetico = pd.DataFrame(index = fechas, columns = AN_Q.columns)
AN_Q_sintetico["AGNO_CALEND"] = years
AN_Q_sintetico["MES_No"] = months


for index, row in AN_Q_sintetico.iterrows():
    print(row['column'])

Q = 0

for index, row in AN_Q_sintetico.iterrows():
    for col in AN_Q_sintetico.columns:
        mes = row["MES_No"]
        if col in cuencas_m:
            coef_m = coef_m_mensuales.loc[mes][col]
            coef_n = coef_n_mensuales.loc[mes][col]
            Q = Q_AN1.loc[index]*coef_m+coef_n
            Q = max(Q.values,0)
            AN_Q_sintetico = AN_Q_sintetico.copy()
            AN_Q_sintetico.at[index,col] = Q
        elif col in cuencas_a:
            coef_m = coefs_m_anuales[col].values
            coef_n = coefs_n_anuales[col].values
            Q_anual_1 = Q_AN1.mean()
            Q_anual_2 = Q_anual_1*coef_m+coef_n
            razon = razones_template.loc[mes][col]
            razon = razon.copy()
            Q = Q_anual_2*razon
            Q = max(Q.values,0)
            AN_Q_sintetico = AN_Q_sintetico.copy()
            AN_Q_sintetico.at[index,col] = Q
            
AN_Q_sintetico['AN-01'] =     Q_AN1      
AN_Q_sintetico[AN_Q_sintetico < 0] = 0
#ruta_modelo = r''

AN_Q_sintetico.to_csv(ruta_temp+'\\AN_Q_sintetico.csv',index = None)
    


        
