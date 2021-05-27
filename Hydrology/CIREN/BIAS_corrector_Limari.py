# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:12:17 2020

@author: Carlos
"""

import os
import numpy as np
import pandas as pd

def biasCorrectorPP(DGA,CR2METhist):
    CR2METhist.columns = DGA.columns
    mask_DGA = (DGA.index > '1979-12-31') & (DGA.index <= '2015-12-31')
    DGAhist = DGA[mask_DGA]
    mediamensualDGA = DGAhist.groupby(DGAhist.index.month).mean()

    mask_gcm = (CR2METhist.index > '1979-12-31') & (CR2METhist.index <= '2015-12-31')
    CR2hist = CR2METhist[mask_gcm]
    mediamensualGCM = CR2hist.groupby(CR2hist.index.month).mean()
    
    factor = pd.DataFrame(mediamensualDGA/mediamensualGCM, index=mediamensualDGA.index, columns=mediamensualDGA.columns)
    
    CR2METhist_corr = CR2METhist
    
    for index, row in CR2METhist.iterrows():
        mes = int(row['mes'])
        fila_factor = factor.loc[mes,:]
        CR2METhist_corr.loc[index,:] = CR2METhist.loc[index,:]*fila_factor
    
    return CR2METhist_corr
    
    
def biasCorrectorT(DGA_t,CR2METhist_t):
    CR2METhist_t.columns = DGA_t.columns
    mask_DGA_t = (DGA_t.index > '1979-12-31') & (DGA_t.index <= '2005-05-31')
    DGAhist_t = DGA_t[mask_DGA_t]
    mediamensualDGA_t = DGAhist_t.groupby(DGAhist_t.index.month).mean()

    mask_gcm_t = (CR2METhist_t.index > '1979-12-31') & (CR2METhist_t.index <= '2015-12-31')
    CR2hist_t = CR2METhist_t[mask_gcm_t]
    mediamensualGCM_t = CR2METhist_t.groupby(CR2METhist_t.index.month).mean()
    
    delta_t = mediamensualDGA_t-mediamensualGCM_t
    
    CR2METhist_corr_t = CR2METhist_t
    
    for index, row in CR2METhist_t.iterrows():
        mes = int(row['Month'])
        fila_factor_t = delta_t.loc[mes,:]
        CR2METhist_corr_t.loc[index,:] = CR2METhist_corr_t.loc[index,:]+fila_factor_t
    
    return CR2METhist_corr_t
    
    
#%% Inputs CR2MET
            
ruta_cr2met = r'D:\ARClim\Extraer_series\Salidas\Limari\CR2MET_v2'
ruta_DGA = r'D:\ARClim\Extraer_series\Salidas\Limari\Clima_DGA'

files_pp_cr2met = []
files_tmax_cr2met = []
fileS_tmin_cr2met = []

files_pp_DGA = []
files_t_DGA = []

for r, d, f in os.walk(ruta_cr2met):
    for file in f:
        if 'pr_' in str(file):
            files_pp_cr2met.append(os.path.join(r, file))
        if 'tmax' in file:
            files_tmax_cr2met.append(os.path.join(r, file))      
        if 'tmin' in file:
            fileS_tmin_cr2met.append(os.path.join(r, file))     
            
for r, d, f in os.walk(ruta_DGA):
    for file in f:
        if 'Precip' in str(file):
            files_pp_DGA.append(os.path.join(r, file))
        if 'Temp' in file:
            files_t_DGA.append(os.path.join(r, file))      

# Lista para extraer fechas de los inputs climáticos

keys = ['Day', 'Month', 'Year']

# Leer precipitación y temperatura

#pp_cr2met = pd.read_csv(files_pp_cr2met[0])
tmax_cr2met = pd.read_csv(files_tmax_cr2met[0])
tmin_cr2met = pd.read_csv(fileS_tmin_cr2met[0])

#pp_cr2met.set_index(pd.to_datetime(pp_cr2met[keys]), inplace=True)
#pp_cr2met = pp_cr2met.resample('MS').sum()
#pp_cr2met['Year'] = pp_cr2met.index.year
#pp_cr2met['Month'] = pp_cr2met.index.month
#pp_cr2met['Day'] = pp_cr2met.index.day

t_media_cr2met = (tmax_cr2met + tmin_cr2met) / 2
t_media_cr2met.set_index(pd.to_datetime(t_media_cr2met[keys]), inplace=True)
t_media_cr2met = t_media_cr2met.resample('MS').mean()
t_media_cr2met['Year'] = t_media_cr2met.index.year
t_media_cr2met['Month'] = t_media_cr2met.index.month
t_media_cr2met['Day'] = t_media_cr2met.index.day

#pp_DGA = pd.read_csv(files_pp_DGA[0]) 
#pp_DGA = pp_DGA.set_index(pd.to_datetime(pp_DGA['anio'].astype(str)+'-'+pp_DGA['mes'].astype(str).str.zfill(2), format='%Y-%m'))

t_DGA = pd.read_csv(files_t_DGA[0]) 
t_DGA = t_DGA.set_index(pd.to_datetime(t_DGA['Year'].astype(str)+'-'+t_DGA['Month'].astype(str).str.zfill(2), format='%Y-%m'))

#del pp_cr2met['Day']

#mask_DGA = (pp_DGA.index > '1979-12-31') & (pp_DGA.index <= '2015-12-31')
#pp_DGA = pp_DGA[mask_DGA]
    
#pp_cr2met_corr = biasCorrectorPP(pp_DGA,pp_cr2met)
#pp_cr2met_corr.to_csv('pp_CR2MET_corr.csv')

del t_media_cr2met['Day']

t_media_cr2met_corr = biasCorrectorT(t_DGA,t_media_cr2met)
t_media_cr2met_corr.to_csv('t_media_cr2met_corr.csv')

