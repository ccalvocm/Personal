# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author: CCCM

"""

# Limpiar entorno
%reset -f

import pandas as pd
import math
import statsmodels.api as sm
import numpy as np

import fiscalyear
fiscalyear.START_MONTH = 4

#% funciones

def agnohidrologico(year_,month_):
    cur_dt = fiscalyear.FiscalDate(year_, month_, 1) 
    retornar = cur_dt.fiscal_year - 1
    return retornar
    
    
#%%

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
    


        
