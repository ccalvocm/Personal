# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 08:26:25 2021

@author: ccalvo
"""

import pandas as pd

def main():
    
    ruta_OD = r'C:\Users\ccalvo\OneDrive - ciren.cl\Of hidrica'
    Q_Maipo = pd.read_csv(ruta_OD+r'\AOHIA_ZC\Etapa 1 y 2\datos\Q_relleno_MLR_Maipo_1979-2019_relleno.csv', index_col = 0, parse_dates = True)
    Q_Maipo_mm = Q_Maipo.resample('MS').mean()
    Q_Rapel = pd.read_csv(ruta_OD+r'\AOHIA_ZC\Etapa 1 y 2\datos\Q_relleno_MLR_Rapel_1979-2020_outlier_in_correction_median.csv', index_col = 0, parse_dates = True)
    Q_Rapel_mm = Q_Rapel.resample('MS').mean()
    Q_Maule = pd.read_csv(ruta_OD+r'\AOHIA_ZC\Etapa 1 y 2\datos\Q_relleno_MLR_Maule_1979-2020_outlier_in_correction_median.csv', index_col = 0, parse_dates = True)
    Q_Maule_mm = Q_Maule.resample('MS').mean()
    print('Fin del promedio')


if __name__ == '__main__':
    main()