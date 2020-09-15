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

workspace = r'E:\WEAP\Weap Areas\Tutorial_ET_v1\MODFLOW_2005_ET\backup'
os.chdir(workspace)
try:
    os.mkdir(workspace+'\\model_output_test')
except:
    print('directorio ya existe')
    
model_output = os.path.join(workspace, 'model_output_test')

#%% load model for examples
nam_file = "MODFLOW_2005_ET.nam"
# model_ws = os.path.join(".",")
ml = flopy.modflow.Modflow.load(nam_file, model_ws=workspace, check=True)

# bcf = flopy.modflow.ModflowBcf(ml)

# ml.export(model_output, fmt='vtk', binary=True,vtk_grid_type ='UnstructuredGrid', point_scalars=True)
ml.dis.export(model_output, fmt='vtk',vtk_grid_type ='UnstructuredGrid')
ml.lpf.export(model_output, fmt='vtk',vtk_grid_type ='UnstructuredGrid')
# bcf.export(model_output, fmt='vtk', point_scalars=True)
ml.rch.export(model_output, fmt='vtk',vtk_grid_type ='UnstructuredGrid')
