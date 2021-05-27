# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:57:12 2021

@author: Carlos
"""

import pandas as pd

pp_cen = pd.read_excel(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\datosCEN\Pp\Precipitaciones_CEN.xlsx',skiprows = 1, 
                       parse_dates = True, index_col = 10,sheet_name = 'Datos')

pp_CEN = pd.DataFrame([],index = pd.date_range('1979-01-01','2021-12-31',freq = '1d'), columns = ['CIPRESES', 'COLB?ÜN', 'MELADO', 'PEHUENCHE', 'RAPEL'])

for col in ['CIPRESES', 'COLB?ÜN', 'MELADO', 'PEHUENCHE', 'RAPEL']:
    pp_CEN.loc[pp_cen[col].index[pp_cen[col].index.notnull()],col] = pp_cen.loc[pp_cen[col].index[pp_cen[col].index.notnull()],col]

pp_CEN.to_csv(r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC\Etapa 1 y 2\datos\datosCEN\Pp\pp_CEN_completo.csv')
