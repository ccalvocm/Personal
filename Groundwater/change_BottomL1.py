# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:25:22 2020

@author: ccc
"""

import flopy
import matplotlib.pyplot as plt
from flopy.export.vtk import export_package



def main():
    
    mf_dir = r'C:\Users\fcidm\Downloads\transitorioCalibracionMF2005'
    
    m = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(m)

if __name__ == "__main__":
    main()



