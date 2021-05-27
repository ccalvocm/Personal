# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:13:40 2021

@author: Carlos
"""


import fiscalyear
import scipy.stats as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

fiscalyear.START_MONTH = 4    

def best_fit_distribution(data, bins, DISTRIBUTIONS):
   '''
   Parameters
   ----------
   data : DataFrame
        Datos a ajsutar.
   bins : int
        bins.
   DISTRIBUTIONS : st
        Distribuciones de probabilidad candidatas.

   Returns
   -------
   None.

   '''

   """Model data by finding best fit distribution to data"""
   # Get ^ogram of original data
   y, x = np.histogram(data, bins=bins, density=True)
   indices = np.where(y == 0)
   x = (x + np.roll(x, -1))[:-1] / 2.0
     
         # Best holders
   best_distribution = st.lognorm
   best_params = (0.0, 1.0)
   best_sse = np.inf
   best_xi2 = np.inf
   
   
   # Estimate distribution parameters from data
   for distribution in DISTRIBUTIONS:
       
       sse = -1

       # Try to fit the distribution
       try:
           # Ignore warnings from data that can't be fit
           # fit dist to data
           if distribution == 'logpearson3':
               distribution_aux = st.pearson3
               data_aux = data.copy()
               data_aux[data_aux <= 0] = 10.**-10.
               data_aux = np.log(data_aux)
               params = distribution_aux.fit(data_aux.values)
           else:
               # cambiar esto
               params = distribution.fit(data.values)

           # Separate parts of parameters
           arg = params[:-2]
           loc = params[-2]
           scale = params[-1]
           
           #Test de Chi-cuadrado
           #Lognormal está programada distinto
           
           if distribution == st.lognorm:
                # Calculate fitted PDF and error with fit in distribution
               pdf = distribution.pdf(x, loc=loc, scale = scale, s=1.)
               xi2 = np.sum(np.power(y - pdf, 2.0)/pdf)
               sse = np.sum(np.power(y - pdf, 2.0))
           elif distribution == 'logpearson3':
               y, x = np.histogram(data_aux, bins=bins, density=True)
               x = (x + np.roll(x, -1))[:-1] / 2.0
               pdf = distribution_aux.pdf(x, loc=loc, scale = scale, *arg)
               y[y <= 0] = 10.**-10.
               xi2 = np.sum(np.power(y - pdf, 2.0)/pdf)
               sse = np.sum(np.power(y - pdf, 2.0))
           else:
               # Calculate fitted PDF and error with fit in distribution
               pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
               xi2 = np.sum(np.power(y - pdf, 2.0)/pdf)
               sse = np.sum(np.power(y - pdf, 2.0))
           # identify if this distribution is better
           if best_sse > sse > 0:
               best_distribution = distribution
               best_params = params
               best_sse = sse
               best_xi2 = xi2
   
       except Exception as e:
              print(e)
              pass

   if best_distribution == 'logpearson3':
      xi2_max = st.chi2.ppf(0.05, df=3)
      return ('logpearson3', best_params)
   elif best_distribution == st.pearson3:
      xi2_max = st.chi2.ppf(0.05, df=3)
      return (best_distribution.name, best_params)
   else:
      xi2_max = st.chi2.ppf(0.05, df=2)
      return (best_distribution.name, best_params)
   
def CVE_pdf(df_relleno, pbb, distr, est):  
    '''
    
    Parameters
    ----------
    df_relleno : DataFrame
        caudales rellenados
    pbb : List
        Probabilidades de excedencia.
    distr : st 
        Funciones de distribucion de probabilidad.

    Returns
    -------
    best_fit_name : string
        Nombre de la distribucion de probabilidad ajustada.
    cve_pdf : DataFrame
        Curva de variación estacional usando la distribución de probabilidad ajustada.

    '''
    
    # ------------------------
    # distribuciones Maipo
    # ------------------------
    distr = [st.norm, st.lognorm , st.expon, st.gamma, st.gumbel_l, st.pearson3, st.weibull_min, st.weibull_max, 'logpearson3']
    
    # ------------------------
    # distribuciones Rapel
    # ------------------------
    # distr = [st.norm, st.lognorm , st.expon, st.gamma, st.gumbel_l,st.gumbel_r, st.pearson3, st.weibull_min, 'logpearson3']
    
    # ------------------------
    # distribuciones CNC Maipo
    # ------------------------
    # distr = [st.norm, st.lognorm , st.expon, st.gamma, st.gumbel_l, st.weibull_min]
    
    # ------------------------
    # distribuciones CNC Maule
    # ------------------------
    # distr = [st.norm, st.lognorm , st.expon, st.gamma, st.gumbel_l, st.weibull_min, st.weibull_max]
   
    # ------------------------
    # distribuciones MAULE
    # ------------------------
    # distr = [st.norm, st.lognorm , st.expon, st.gamma, st.gumbel_l, st.gumbel_r, st.pearson3, st.weibull_min, st.weibull_max, 'logpearson3']
    
    # ------------------------
    # distribuciones Ñuble
    # ------------------------
    # distr = [st.norm, st.lognorm , st.expon, st.gumbel_l, st.pearson3, st.weibull_min, st.weibull_max, 'logpearson3']

    
    distr_backup = distr.copy()
    
    best_dist_list = []
    
    # bins según Gabirel Castro
    bins = 200
    
    # iniciarlizar df y recorrer meses
    cve_pdf = pd.DataFrame([],index = [4,5,6,7,8,9,10,11,12,1,2,3], columns = df_relleno.columns)
    
    for mes in range(1,13):
        data = pd.DataFrame(df_relleno[df_relleno.index.month == mes ].values.ravel())
        
        if est == 'QUEBRADA RAMON EN RECINTO EMOS R.N.':
            distr.remove('logpearson3')
        if est in ['ESTERO PUANGUE EN BOQUERON R.N.','ESTERO ARRAYAN EN LA MONTOSA R.N.',
                   'ESTERO ARRAYAN EN LA MONTOSA R.N.']:
            distr.remove('logpearson3')
        if mes == 11:
            if est in ['RIO VOLCAN EN QUELTEHUES R.N.','RIO ANCOA EN EL MORRO R.N.','RIO ACHIBUENO EN LA RECOVA R.N.']:
                distr.remove(st.pearson3)
            if est == 'RIO MAULE EN ARMERILLO R.N.':
                distr = [st.gumbel_r]
        if (mes in [9,11]) & (est in ['RIO MAIPO EN LAS HUALTATAS R.N.']):
            distr.remove(st.pearson3)
        if mes == 1:
            if est in ['RIO CACHAPOAL 5 KM. AGUAS ABAJO JUNTA CORTADERAL R.N.',
                       'RIO CACHAPOAL EN PTE TERMAS DE CAUQUENES R.N.']:
                distr.remove(st.weibull_min)     
            if est == 'RIO PANGAL EN PANGAL R.N.':
                distr = [x for x in distr if x not in [st.pearson3, st.weibull_min,st.expon,st.gamma,st.norm]]
            if est in ['RIO TINGUIRIRICA BAJO LOS BRIONES R.N.','ESTERO CHIMBARONGO BAJO EMBALSE CONVENTO VIEJO R.N.']:
                distr.remove(st.pearson3)     
            if est == 'Estero Potrero Grande en Toma DOS':
                distr.remove('logpearson3')    
            if est == 'RIO LAS LEÑAS ANTE JUNTA RIO CACHAPOAL R.N.':
                distr = [x for x in distr if x not in [st.norm,st.gumbel_l,st.expon]]
        if mes == 2:    
            if est in  ['ESTERO UPEO EN UPEO R.N.','ESTERO CURIPEUMO EN LO HERNANDEZ R.N.']:
                distr = [x for x in distr if x not in [st.pearson3,'logpearson3']]     
            if est =='RIO CORTADERAL ANTE JUNTA RIO CACHAPOAL R.N.':
                distr.remove(st.norm)
        if mes == 3:
            if est == 'RIO CLARO EN TUNCA R.N.':
                distr = [x for x in distr if x not in [st.gamma,st.weibull_min]]
            if est in ['ESTERO UPEO EN UPEO R.N.','ESTERO CURIPEUMO EN LO HERNANDEZ R.N.']:
                distr = [x for x in distr if x not in [st.pearson3,'logpearson3']]
            if est == 'RIO ANCOA EN EL MORRO R.N.':
                distr.remove('logpearson3')
            if est == 'RIO COLINA EN PELDEHUE R.N.':
                distr.remove('logpearson3')
        if mes == 5:
            if est == 'RIO MELADO EN EL SALTO R.N.':
                distr = [x for x in distr if x not in [st.gamma,st.gumbel_r]]
            if est =='RIO CLARO EN TUNCA R.N.':
                distr.remove(st.norm)
            if est == 'ESTERO YERBA LOCA ANTES JUNTA SAN FRANCISCO R.N.':
                distr.remove('logpearson3')
        if mes == 6:
            if est == 'RIO COLINA EN PELDEHUE R.N.':
                distr = [x for x in distr if x not in [st.lognorm,st.expon,st.pearson3,st.weibull_min]]
            if est in ['RIO CACHAPOAL 5 KM. AGUAS ABAJO JUNTA CORTADERAL R.N.',
                       'RIO CACHAPOAL EN PTE TERMAS DE CAUQUENES R.N.']:
                distr = [x for x in distr if x not in [st.pearson3,st.weibull_min]]             
            if est == 'ESTERO UPEO EN UPEO R.N.':
                distr = [x for x in distr if x not in [st.weibull_min, st.pearson3]]
            if est == 'ESTERO PUANGUE EN BOQUERON R.N.':
                distr = [x for x in distr if x not in [st.lognorm, st.expon,st.weibull_min]]
            if est == 'RIO MAPOCHO EN LOS ALMENDROS R.N.':
                distr = [x for x in distr if x not in [st.lognorm,st.pearson3,'logpearson3']]
            if est == 'RIO MAIPO EN LAS HUALTATAS R.N.':
                distr = [x for x in distr if x not in [st.expon,st.lognorm]]
            if est == 'RIO MAIPO EN LAS MELOSAS R.N.':
                distr = [x for x in distr if x not in [st.weibull_min, st.gamma]]
            if est == 'RIO LONGAVI EN LA QUIRIQUINA R.N.':
                distr = [x for x in distr if x not in [st.gamma,st.pearson3,st.weibull_min,st.gumbel_r,st.lognorm]]
        if mes == 7:
            if est == 'RIO CLARO EN EL VALLE R.N.':
                distr = [x for x in distr if x not in [st.lognorm, st.pearson3]]      
            if est == 'RIO CACHAPOAL EN PTE TERMAS DE CAUQUENES R.N.':
                distr = [x for x in distr if x not in [st.norm,st.lognorm]]
            distr.remove(st.expon)
        if est in ['RIO LAS LEÑAS ANTE JUNTA RIO CACHAPOAL R.N.','RIO CORTADERAL ANTE JUNTA RIO CACHAPOAL R.N.']:
                distr = [x for x in distr if x not in [st.pearson3,st.weibull_min, st.gamma]]
        if est == 'RIO CORTADERAL ANTE JUNTA RIO CACHAPOAL R.N.':
            distr = [st.norm]
        if est == 'ESTERO DE LA CADENA ANTES JUNTA RIO CACHAPOAL R.N.':
            distr.remove('logpearson3')
        if est == 'RIO MAPOCHO EN LOS ALMENDROS R.N.':
            distr = [x for x in distr if x not in [st.lognorm]]
        if mes == 8:
            if est == 'RIO CLARO EN LOS QUEÑES R.N.':
                distr = [x for x in distr if x not in [st.weibull_min, st.pearson3, st.gamma,st.lognorm]]
            if est == 'RIO TENO DESPUES DE JUNTA CON CLARO R.N.':
                distr = [x for x in distr if x not in [st.expon,st.lognorm, st.pearson3]]
            if est in ['RIO LONGAVI EN LA QUIRIQUINA R.N.','RIO CLARO EN RAUQUEN R.N.']:
                distr = [x for x in distr if x not in [st.gamma, st.expon, st.pearson3,st.weibull_min]]
            if est == 'RIO LONGAVI EN EL CASTILLO R.N.':
                distr = [x for x in distr if x not in [st.lognorm, st.expon]]
            if est == 'RIO TENO DESPUES DE JUNTA CON CLARO R.N.':
                distr = [st.gumbel_r, st.uniform, st.laplace, st.rayleigh]
            if est == 'ESTERO UPEO EN UPEO R.N.':
                distr = [x for x in distr if x not in [st.gamma, st.pearson3]]
            if est == 'RIO MELADO EN EL SALTO R.N.':
                distr.remove(st.gumbel_r)
        if mes == 9:
            if est in ['RIO COLORADO ANTES JUNTA RIO OLIVARES R.N.']:
                distr = [x for x in distr if x not in [st.gumbel_l,st.pearson3]]
            if est == 'RIO MAIPO EN SAN ALFONSO R.N.':
                distr = [x for x in distr if x not in [st.pearson3]]
            if est == 'RIO LONGAVI EN LA QUIRIQUINA R.N.':
                distr = [x for x in distr if x not in [st.gumbel_r,st.pearson3]]    
        if mes == 10:
            if est in ['RIO MATAQUITO EN PUENTE LA HUERTA R.N.','RIO MATAQUITO EN LICANTEN R.N.']:
                distr.remove(st.pearson3)
            if est == 'RIO LONGAVI EN LA QUIRIQUINA R.N.':
                distr = [x for x in distr if x not in [st.weibull_min]]
            if est == 'RIO MAIPO EN LAS HUALTATAS R.N.':
                distr = [x for x in distr if x not in [st.pearson3, st.gumbel_l]]
            if est == 'RIO COLORADO ANTES JUNTA RIO OLIVARES R.N.':
                distr = [x for x in distr if x not in [st.gamma,st.pearson3]]
            if est == 'RIO LONGAVI EN LA QUIRIQUINA R.N.':
                distr = [x for x in distr if x not in [st.gamma,st.pearson3, st.gumbel_r]]
        if mes == 12:
            if est =='RIO PANGAL EN PANGAL R.N.':
                distr.remove(st.pearson3)
            if est == 'RIO CACHAPOAL EN PTE TERMAS DE CAUQUENES R.N.':
                distr = [x for x in distr if x not in [st.norm, st.pearson3]]   
            if est == 'RIO COLORADO ANTES JUNTA RIO MAIPO R.N.':
                distr = [x for x in distr if x not in [st.gamma]]                                              
        if est == 'RIO MELADO EN EL SALTO R.N.':
            distr.remove(st.pearson3)
        if est == 'QUEBRADA RAMON EN RECINTO EMOS R.N.':
            distr.remove(st.weibull_min)

        
        best_fit_name, best_fit_params = best_fit_distribution(data, bins, distr)
        
        # Separate parts of parameters
        arg = best_fit_params[:-2]
        loc = best_fit_params[-2]
        scale = best_fit_params[-1]
           
        if best_fit_name == 'logpearson3':
            best_dist = getattr(st, 'pearson3')
            cve_pdf.loc[mes, cve_pdf.columns] = np.exp(best_dist.ppf(1-pbb, loc = loc, scale =  scale, *arg))
            
        elif best_fit_name == 'lognorm':
            best_dist = getattr(st, best_fit_name)  
            cve_pdf.loc[mes, cve_pdf.columns] = best_dist.ppf(1-pbb, loc = loc, scale =  scale, s = arg)
        else:
            best_dist = getattr(st, best_fit_name)  
            cve_pdf.loc[mes, cve_pdf.columns] = best_dist.ppf(1-pbb, loc = loc, scale =  scale, *arg)
        
        distr_corr = distr.copy()
        while cve_pdf.loc[mes, cve_pdf.columns].min() < 0:
            if best_fit_name == 'logpearson3':
                distr_corr.remove('logpearson3')
            else:
                distr_corr.remove(best_dist)
            
            best_fit_name, best_fit_params = best_fit_distribution(data, bins, distr_corr)
            
            # Separate parts of parameters
            arg = best_fit_params[:-2]
            loc = best_fit_params[-2]
            scale = best_fit_params[-1]
        
            if best_fit_name == 'logpearson3':
                best_dist = getattr(st, 'pearson3')
                cve_pdf.loc[mes, cve_pdf.columns] = np.exp(best_dist.ppf(1-pbb, loc = loc, scale =  scale, *arg))
                
            elif best_fit_name == 'lognorm':
                best_dist = getattr(st, best_fit_name)
                cve_pdf.loc[mes, cve_pdf.columns] = best_dist.ppf(1-pbb, loc = loc, scale =  scale, s = arg)
            else:
                best_dist = getattr(st, best_fit_name)
                cve_pdf.loc[mes, cve_pdf.columns] = best_dist.ppf(1-pbb, loc = loc, scale =  scale, *arg)
                        
        best_dist_list.append(best_fit_name)
        
        distr = distr_backup.copy()

    return best_dist_list, cve_pdf