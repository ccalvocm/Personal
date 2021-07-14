# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:38:15 2021

@author: Carlos
"""

import pandas as pd

ruta_par = r'D:\GitHub\Entrega_E3_CNR\Cachapoal_Puente_Termas\PEST_HP\PESTHP20002020\master20002020sce.par'
par = pd.read_csv(ruta_par, delim_whitespace=True, skiprows = 1, names= ['par','val', '1', '0'], index_col = 0)


