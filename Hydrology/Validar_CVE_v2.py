# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:47:17 2020

@author: ccalvo
"""

def limpiar_kernel():    
    try:
        from IPython import get_ipython
        get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass
 
limpiar_kernel()

#%%
import pandas as pd 
import matplotlib.pyplot as plt
import os
from hydroeval import evaluator, nse

def parametros_distribucion(data_, distr_):
    params = distr_.fit(data_)

    return params

def CVEParser(txt_):
    CVE = dict()
    n = 0
    station = None
    data = None
    with open(txt_, 'r') as f:
        for line in f:
            L = line.split()
            if len(L) == 1:
                station = L[0]
                data = []
            elif len(L) > 1:
                data.append(L)
                n +=1
                print(n)
                print(L)
            if n >= 5:
                n = 0
                CVE[station] = data
            else:
                pass
    
        return CVE


def CVE(dataframe, quantiles, months = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                                        'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                                        'Nov', 'Dec'],
        hydroyear = True, aggregate = True):
    '''
    

 

    Parameters
    ----------
    dataframe : Pandas dataframe
        Pandas dataframe with index as datestamps and columns as stations
    quantiles : array or list
        List of quantiles from 0 to 1 e.g [0.85, 0.9]. 85 and 90 quantile
    months : array or list of months in english with 3 characters e.g 'Jan'
        List of months of the year.
    hydroyear : boolean, default to False
        Set to true if you wish to use the hydrologic year in the southern
        hemisphere: ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                     'Nov', 'Dec', 'Jan', 'Feb', 'Mar']

 

    Returns
    -------
    mdf : list of Pandas dataframes
        returns list of dataframes showing the quantile month calculation
        for all stations.

 

    '''
    if hydroyear:
        months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                     'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
    else:
        pass
    
    mdf = dict()
    if aggregate:
        for qtile in quantiles:
            MonthAV = dataframe.groupby(dataframe.index.month).quantile(q = 1-qtile)
            MonthAV['Total'] = MonthAV.mean(axis = 1)
            MonthAV.index = pd.to_datetime(MonthAV.index, format = '%m')
            MonthAV.index = MonthAV.index.month_name().str.slice(stop=3)
            MonthAV = MonthAV.reindex(months)
            mdf[str(qtile)] = MonthAV
    else:
        mdf = dict()
        for col in dataframe.columns:
            qtl_group = dict()
            for qtile in quantiles:
                MonthAV = dataframe[col].groupby(dataframe.index.month).quantile(q = 1-qtile)
                # MonthAV['Total'] = MonthAV.mean(axis = 1)
                MonthAV.index = pd.to_datetime(MonthAV.index, format = '%m')
                MonthAV.index = MonthAV.index.month_name().str.slice(stop=3)
                MonthAV = MonthAV.reindex(months)
                qtl_group[str(qtile)] = MonthAV
            mdf[col] = qtl_group
    return mdf

def NSE(nse, sim_flow, obs_flow, axis=1):
    serie_sim = sim_flow.values.ravel()
    serie_obs = obs_flow.values.ravel()
    my_nse = evaluator(nse, serie_sim, serie_obs, axis=1)
    return my_nse
    

#%%
def main():
  
    #%%
# Informe: 
# DIAGNOSTICO Y CLASIFICACION DE LOS CURSOS Y CUERPOS DE AGUA SEGUN OBJETIVOS DE CALIDAD
# CUENCA DEL RIO MAIPO
# JULIO 2004
# URL: https://mma.gob.cl/wp-content/uploads/2017/12/Maipo.pdf
  
  # inputs:
  ruta_Git = r'C:\Users\ccalvo\Documents\GitHub'
#  ruta_Git = 'D:\GitHub'
  ruta_Q_rellenos = ruta_Git+r'\Analisis-Oferta-Hidrica\Hidrología\Caudales\Validacion\Q_relleno_MLR_Maipo_1950-2008_outlier_in_correction_sin_saltar.csv'
  # ruta_Q_rellenos = ruta_Git+r'\Analisis-Oferta-Hidrica\Hidrología\Caudales\Q_relleno_MLR_Maipo_1950-2008_outlier_in_correction.csv'


  # ruta_Q_rellenos = ruta_Git+r'\WEAP-MODFLOW-personal\Hydrology\Q_relleno_MLR_Maipo_1950-2001.csv'
  probabilidades_excedencia_grupo1 = [.05, .1, .5, .85, .95]
  probabilidades_excedencia_grupo2 = [.05, .2, .5, .85, .95]
  
  # Grupo de estaciones
  
  grupo2 = ['05722001-5', '05722002-3', '05737002-5', '05746001-6']

  # estaciones_date = {'05710001-K': ['1950-01-01', '2001-12-31'],
  #      '05701001-0': ['1979-01-01', '2001-12-31'],
  #      '05701002-9': ['1962-01-01', '1993-12-31'],
  #      '05702001-6': ['1950-01-01', '2001-12-31'],
  #      '05704002-5': ['1950-01-01', '2001-12-31'],
  #      '05705001-2': ['1977-01-01', '2001-12-31'],
  #      '05706001-8': ['1977-01-01', '2001-12-31'],
  #      '05707002-1': ['1950-01-01', '2001-12-31'],
  #      '05721001-K': ['1986-01-01', '2001-12-31'],
  #      '05722001-5': ['1952-01-01', '2001-12-31'],
  #      '05722002-3': ['1950-01-01', '2001-12-31'],
  #      '05716001-2': ['1981-01-01', '2001-12-31'],
  #      '05735001-6': ['1980-01-01', '2001-12-31'],
  #      '05737002-5': ['1959-01-01', '2001-12-31'],
  #      '05741001-9': ['1950-01-01', '2001-12-31'],
  #      '05746001-6': ['1986-01-01', '2001-12-31'],
  #      '05748001-7': ['1979-01-01', '2001-12-31']}
  
  estaciones_date = {'05710001-K': ['1950-01-01', '1998-12-31'],
       '05722001-5': ['1950-01-01', '1998-12-31'],
       '05722002-3': ['1950-01-01', '1998-12-31'],
       '05737002-5': ['1950-01-01', '1998-12-31'],
       '05716001-2': ['1981-01-01', '2007-12-31'],
       '05748001-7': ['1980-01-01', '2007-12-31'],
       # '05717005-0': ['1950-01-01', '1960-12-31'],
       '05746001-6': ['1987-01-01', '2005-12-31']
       } 
  ruta = ruta_Git+r'\Analisis-Oferta-Hidrica\Hidrología\Caudales\Validacion'
  os.chdir(ruta)

  # ruta_datos = 'Validacion_Maipo_DGA_2004.txt'
  ruta_datos = 'Validacion_Maipo_DGA_2015.txt'
  
  cve = CVEParser(ruta_datos)
  
  Q_relleno = pd.read_csv(ruta_Q_rellenos, index_col = 0)[estaciones_date.keys()]
  Q_relleno.index = pd.to_datetime(Q_relleno.index)
  inicio = pd.to_datetime('1950-01-01',format='%Y-%m-%d')
  fin = pd.to_datetime('2007-12-31',format='%Y-%m-%d')
  Q_relleno = pd.DataFrame(Q_relleno[Q_relleno.index <= fin ],  index = pd.date_range(inicio, fin, freq='D', closed='right'))
  
   # crear lista de pbb de excedencia de meses por estaciones

  caudales_pbb_mes = {x:'' for x in estaciones_date.keys()}

  # calcular CVE

  pbb_mensuales = pd.DataFrame(columns=range(4), index = [0,1,2,3,4,5,6,7,8,9,10,11])

  r = -1
  c = 0
  plt.close("all")
  fig, axes = plt.subplots(3,3)
  
  N_SE = []
  # iterar sobre estaciones
  for i,estacion in enumerate(estaciones_date):
                  
    cve_informe = pd.DataFrame(cve[estacion]).transpose().astype(float)
    
    probabilidades_excedencia = probabilidades_excedencia_grupo1
    
    if estacion in grupo2:
    
        probabilidades_excedencia = probabilidades_excedencia_grupo2
        
    for index, col in enumerate(probabilidades_excedencia):
        
        fechas = pd.to_datetime(estaciones_date[estacion])
    
        CVE_rellenada = CVE(pd.DataFrame(Q_relleno[estacion].loc[(Q_relleno.index <= fechas[-1]) & (Q_relleno.index >= fechas[0])], columns = [estacion], index = Q_relleno.index), probabilidades_excedencia)[str(probabilidades_excedencia[index])][estacion]

        pbb_mensuales.loc[pbb_mensuales.index, index] =  CVE_rellenada.to_list()
        
    caudales_pbb_mes[estacion] = pbb_mensuales
    
    
#    print(caudales_pbb_mes[estacion])
#    
#    print(cve_informe)
    
    N_SE.append(NSE(nse, caudales_pbb_mes[estacion], cve_informe, axis=1))
    
    if estacion == '05748001-7':

        print(caudales_pbb_mes[estacion])
        print(cve_informe)

    
#Graficar

    if (i+1)%3 == 0:
        r += 1
        c = 0
    
    axis = axes[r,c]
    cve_informe.plot(color = 'r', ax = axis, legend=False, linewidth = 3)
    caudales_pbb_mes[estacion].plot(ax = axis, color = 'b', legend=False, linewidth = 3)

    axis.set_xticks(range(13)) 
    axis.set_xticklabels(['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'])
    axis.set_ylabel('Q $m^3/s$')
    axis.set_title('Estación '+estacion)

    c += 1
        
  axis.legend(['Informe','Rellenada'],bbox_to_anchor=(1.05, 1.05), loc='upper left')    
  
  
#%%
if __name__ == '__main__':
    main()
        
        






