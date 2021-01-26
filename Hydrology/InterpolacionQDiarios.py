# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:10:57 2021

@author: ccalvo
"""

import pandas as pd
import numpy as np


ruta = r'C:\Users\ccalvo\Documents\GitHub\M-SRM_2020\Calibracion\Validar\Q_Copiapo.xls.xlsx'
Q = pd.read_excel(ruta, names = ['dia','Q'])

caudales = np.interp(range(367),Q['dia'],Q['Q'])/1000
    