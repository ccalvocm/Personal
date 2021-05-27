# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 09:49:48 2021

@author: Carlos
"""

import pandas as pd

ruta_OD = r'E:\CIREN\OneDrive - ciren.cl\Of hidrica\AOHIA_ZC'
ruta = ruta_OD + r'\Etapa 1 y 2\datos\Datos CF\Datos_BNA_EstacionesDGA\BNAT_CaudalDiario.txt'

q_EAQ = pd.read_csv(ruta, header = None, index_col = 0, sep = ';')
q_EAQ = q_EAQ.loc['06043001-2']
q_EAQ.index = pd.to_datetime(q_EAQ[2], dayfirst = True)
q_EAQ = q_EAQ.loc[q_EAQ.index >= '1979-01-01']
q_EAQ[5] = 1
q_EAQ_mon = q_EAQ[3].resample('MS').mean()
q_EAQ_flags = q_EAQ[5].resample('MS').sum()

writer = pd.ExcelWriter(ruta_OD+r'\Etapa 1 y 2\datos\EsteroAlhueQuilamuta_flags'+'.xlsx', engine='xlsxwriter')
        
# Write each dataframe to a different worksheet.
q_EAQ_mon.to_excel(writer, sheet_name= 'data')
q_EAQ_flags.to_excel(writer, sheet_name= 'info data')

# Close the Pandas Excel writer and output the Excel file.
writer.save()