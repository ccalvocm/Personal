# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:39:24 2020

@author: fcidm
"""

import os, re, sys, hataripy
import numpy as np
#hataripy is installed in E:\Software\Anaconda3\lib\site-packages\hataripy
modPath = 'C:\\gwv7\\models\\transitorioCalibracion\\'
modName = 'modelCalR2'
exeName = r'C:\gwv7\MF2k.exe'  
mfModel = hataripy.modflow.Modflow.load(modName+'.nam', model_ws=modPath, exe_name="mf2k", version = "mf2k" )
# get a list of the model packages
mfModel.get_package_list()

#Define objects that will be represented on the VTKs and add them to geometry object
# read heads from the model output
headArray = hataripy.utils.binaryfile.HeadFile(modPath+modName+'.hds').get_data()
# get information about the drain cells
drnCells = mfModel.drn.stress_period_data[0]
# add the arrays to the vtkObject
vtkObject = hataripy.export.vtk.Vtk3D(mfModel,modPath,verbose=True)
vtkObject.add_array('head',headArray)
vtkObject.add_array('drn',drnCells)
#Create the VTKs for model output, boundary conditions and water table
vtkObject.modelMesh('modelMesh.vtu',smooth=True,cellvalues=['head'])
vtkObject.modelMesh('modelDrn.vtu',smooth=True,cellvalues=['drn'],boundary='drn',avoidpoint=True)
vtkObject.waterTable('waterTable.vtu',smooth=True)