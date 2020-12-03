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
import numpy as np

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
            else:
                pass
            if n >= 7:
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
    
def Qmm(df_, estacion):
  df = df_.groupby(df_.index.month).mean()
  df = df[estacion].reindex([4,5,6,7,8,9,10,11,12,1,2,3])
  df = df.reset_index()
  df = df.set_index(pd.Index(range(1,13)))
  del df['index']
  df.columns = [estacion]
  return df

#%%
  
def main():

#%%
  
  # inputs:
  ruta_Git = r'C:\Users\ccalvo\Documents\GitHub'
  ruta_Q_rellenos = ruta_Git+r'\Analisis-Oferta-Hidrica\Hidrología\Caudales\Q_relleno_MLR_Maipo_1980-2020_relleno.csv'

  probabilidades_excedencia = [.05, .1, .2, .5, .85, .95]
  

  ruta = ruta_Git+r'\Analisis-Oferta-Hidrica\Hidrología\Caudales\Validacion'
  os.chdir(ruta)

  Q_relleno = pd.read_csv(ruta_Q_rellenos, index_col = 0)
  Q_relleno.index = pd.to_datetime(Q_relleno.index)
  
  year_i = 1979
  year_f = 2019
    
  #fechas
  inicio = pd.to_datetime(str(year_i)+'-12-31',format='%Y-%m-%d')
  fin = pd.to_datetime(str(year_f)+'-12-31',format='%Y-%m-%d')
    
  Q_relleno = pd.DataFrame(Q_relleno[Q_relleno.index <= fin ],  index = pd.date_range(inicio, fin, freq='D', closed='right'))
  
  # calcular CVE

  pbb_mensuales = pd.DataFrame(columns=[probabilidades_excedencia], index = [0,1,2,3,4,5,6,7,8,9,10,11])

#%%
  r = -1
  c = 0
  plt.close("all")
  fig, axes = plt.subplots(3,6)
  
  caudales_pbb_mes = {x:'' for x in Q_relleno.columns}
    # iterar sobre estaciones
  for i,estacion in enumerate(Q_relleno.columns):
                     
    for index, col in enumerate(probabilidades_excedencia):
           
        CVE_rellenada = CVE(pd.DataFrame(Q_relleno[estacion], columns = [estacion], index = Q_relleno.index), probabilidades_excedencia)[str(probabilidades_excedencia[index])][estacion]
       
        pbb_mensuales.loc[pbb_mensuales.index, col] =  CVE_rellenada.to_list()
    
    caudales_pbb_mes[estacion] = pbb_mensuales
        
#Graficar

    if (i)%6 == 0:
        r += 1
        c = 0
    
    axis = axes[r,c]
    colores =  ['blue','magenta',  'yellow',  'cyan', 'purple', 'brown']
    caudales_pbb_mes[estacion].plot(ax = axis, color=colores, style='.-', legend=False, linewidth = 3, logy=False)
        
    axis.set_xticks(range(13)) 
    axis.set_xticklabels(['A', 'M', 'J', 'J', 'A', 'S', 'O',
                     'N', 'D', 'J', 'F', 'M'], FontSize = 8)
    axis.set_ylabel('Q $(m^3/s)$')
    axis.set_title(estacion)
    axis.set_ylim(bottom = 0)
    axis.grid()
    axis.legend(['Q5','Q10', 'Q20','Q50','Q85', 'Q95'], prop={'size': 6})

    c += 1
        
#%%
if __name__ == '__main__':
    main()
        
        






