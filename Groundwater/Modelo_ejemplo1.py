# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:56:52 2020

@author: Carlos
"""


import flopy
import os
import sys
import flopy.utils.binaryfile as bf
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from flopy.export import vtk


#%%

workspace = r'E:\WEAP\Weap Areas\Estero_derecho_v5_utm_ope_v1_SA_con_nodos_inyeccion_variacion_fb_v1\MODFLOW\QAQC modelo'
os.chdir(workspace)
try:
    os.mkdir(workspace+'\\model_output_test')
except:
    print('directorio ya existe')
    
model_output = os.path.join(workspace, 'model_output_test')

#%% load model for examples
nam_file = "hum_e_der_v1.nam"
# model_ws = os.path.join(".",")
ml = flopy.modflow.Modflow.load(nam_file, model_ws=workspace, check=True)

ml.export(model_output, fmt='vtk',vtk_grid_type ='UnstructuredGrid', point_scalars=True)
ml.dis.export(model_output, fmt='vtk', vtk_grid_type ='UnstructuredGrid', point_scalars=True)
ml.lpf.export(model_output, fmt='vtk', vtk_grid_type ='UnstructuredGrid')
