# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: CCCM

"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    plt.close("all")
    
    #Rutas 
    ruta_trabajo = r'D:.\FloPy0'
    os.chdir(ruta_trabajo)   
    ruta_lst = r'.\Cuenca_NN.lst'
    
    #Inicializar
    error = pd.DataFrame([[0,0,0,0]], columns = ['Time step','Stress period','Error volumen','Error flujo'])
    contador = 0
    
    # Abrir archivos
    lst = open(ruta_lst)
            
    for i, line in enumerate(lst):
        if 'VOLUMETRIC BUDGET FOR ENTIRE MODEL AT END OF TIME STEP' in line:
            ts = line[60:61]
            sp = line[-3:-1]
    
        if 'PERCENT DISCREPANCY =' in line:
            error.loc[contador] = [ts, sp, line[32:41], line[-6:-1]]
            contador += 1
    
    error = error.astype(float)
    error.plot()
    
    lst.close()
    
if __name__ == '__main__': main()
