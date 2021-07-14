# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:38:15 2021

@author: Carlos
"""

import pandas as pd
import os


#%%Río Cachapoal en Pte. Termas

#% leer archivo .par
folder = r'D:\GitHub\Entrega_E3_CNR\Cachapoal_Puente_Termas\PEST_HP\PESTHP20002020'
os.chdir(folder)
ruta_par = r'.\master20002020sce.par'
par = pd.read_csv(ruta_par, delim_whitespace=True, skiprows = 1, names= ['par','val', '1', '0'], index_col = 0)

#subir rcr
par_change = [x for x in par.index if x.find('rcr') == 0]
par.loc[par_change, 'val'] = par.loc[par_change, 'val']*3
par.columns = ['single', 'point', '']
par.index.name = ''
par.to_csv(r'.\master20002020sce.par', sep = ' ')

#%%Río Claro Hacienda las Nieves
# bajar rcr

folder = r'D:\GitHub\Entrega_E3_CNR\Claro_Hacienda_Nieves\PESTHP20002020'
os.chdir(folder)
ruta_par = r'.\master20002020sce.par'
par = pd.read_csv(ruta_par, delim_whitespace=True, skiprows = 1, names= ['par','val', '1', '0'], index_col = 0)

#subir rcr
par_change = [x for x in par.index if x.find('rcr') == 0]
par_change_idx = par.loc[par_change, 'val'][par.loc[par_change, 'val'] == 1].index
par.loc[par_change_idx, 'val'] = 0.99
par.columns = ['single', 'point', '']
par.index.name = ''
par.to_csv(r'.\master20002020sce.par', sep = ' ')

#%%Río Colorado en Junta con Palos
# bajar rcr

folder = r'D:\GitHub\Entrega_E3_CNR\Colorado_Junta_Palos\PESTHP20002020'
os.chdir(folder)
ruta_par = r'.\master20002020sce.par'
par = pd.read_csv(ruta_par, delim_whitespace=True, skiprows = 1, names= ['par','val', '1', '0'], index_col = 0)

#subir rcr
par_change = [x for x in par.index if x.find('rcs') == 0]
par_change_idx = par.loc[par_change, 'val'][par.loc[par_change, 'val'] == 1].index
par.loc[par_change_idx, 'val'] = 0.95
par.columns = ['single', 'point', '']
par.index.name = ''
par.to_csv(r'.\master20002020sce.par', sep = ' ')

#%% Río Maipo en El Manzano

folder = r'D:\GitHub\Entrega_E3_CNR\Maipo_en_el_Manzano\PESTHP20002020'
os.chdir(folder)
ruta_par = r'.\master20002020sce.par'
par = pd.read_csv(ruta_par, delim_whitespace=True, skiprows = 1, names= ['par','val', '1', '0'], index_col = 0)

#subir rcr
par_change = [x for x in par.index if x.find('rcs') == 0]
par_change_idx = par.loc[par_change, 'val'][par.loc[par_change, 'val'] == 1].index
par.loc[par_change_idx, 'val'] = 0.98
par.columns = ['single', 'point', '']
par.index.name = ''
par.to_csv(r'.\master20002020sce.par', sep = ' ')

#%% Río Mapocho en Los Almendros

folder = r'D:\GitHub\Entrega_E3_CNR\Mapocho_Los_Almendros\PESTHP20002020'
os.chdir(folder)
ruta_par = r'.\master20002020sce.par'
par = pd.read_csv(ruta_par, delim_whitespace=True, skiprows = 1, names= ['par','val', '1', '0'], index_col = 0)

#subir rcr
par_change = [x for x in par.index if x.find('rcs') == 0]
par_change_idx = par.loc[par_change, 'val'][par.loc[par_change, 'val'] == 1].index
par.loc[par_change_idx, 'val'] = 0.99000000
par.columns = ['single', 'point', '']
par.index.name = ''
par.to_csv(r'.\master20002020sce.par', sep = ' ')