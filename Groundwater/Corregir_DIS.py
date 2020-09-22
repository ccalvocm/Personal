# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:25:22 2020

@author: ccc
"""

import flopy as fp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


mf_dir = r'C:\transitorioCalibracionMF2005'


def main():
    
    plt.close("all")   
    ml = fp.modflow.Modflow.load(mf_dir+"\modelCalR2.nam", model_ws='', verbose=False,
                                   check=False, exe_name="mf2005", version = "mf2005"   )
    
    
    ml.dis.check(verbose=True, level=12)
    
    model_top = ml.dis.top.array
    model_bottom = ml.dis.getbotm()
    
    espesor_l1 = model_top - model_bottom[0,:,:]
    espesor_l2 = model_bottom[0,:,:]-model_bottom[1,:,:]
    plt.imshow(espesor_l2)
    print(np.min(espesor_l1)) #El error está en las celdas inactivas
    print(np.min(espesor_l2))
    YI, XI = np.where(espesor_l2 == 0.0)

    X = np.arange(0,espesor_l1.shape[1],1)
    Y = np.arange(0,espesor_l1.shape[0],1)
    X,Y = np.meshgrid(X,Y)
    values = model_bottom[0,:,:].flatten()
    
    Z0 = griddata((X.ravel(),Y.ravel()), values, (XI, YI))+.1
    model_bottom[0,YI,XI] = Z0
    
    dis = fp.modflow.ModflowDis(ml, 4,nrow = 418 ,ncol = 153,top=model_top,botm=model_bottom,itmuni=1, unitnumber = 29 )

    ml.write_input()

    #Check
    
    mfl = fp.modflow.Modflow.load(r"C:\Users\Carlos\modelCalR2.nam", model_ws=mf_dir, verbose=False,
                                   check=False, exe_name="mf2005", version = "mf2005"   )
    
    model_top = ml.dis.top.array
    model_bottom = ml.dis.getbotm()
    espesor_l2 = model_bottom[0,:,:]-model_bottom[1,:,:]

    YI, XI = np.where(espesor_l2 == 0.0)
    
    if len(XI) == 0 and len(YI) == 0:
        print('Elevaciones arregladas')


if __name__ == '__main__':
    main()