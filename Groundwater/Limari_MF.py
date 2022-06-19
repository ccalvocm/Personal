# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:33:47 2022

@author: Carlos
"""

import os
import sys
import gsflow
import flopy
import platform
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from gsflow import GsflowModel
from gsflow.output import PrmsDiscretization, PrmsPlot
from flopy.plot import PlotMapView
import geopandas

# modelo Limari
path=r'E:\WEAP\Weap Areas\Limari_PEGRH_20\MODFLOW\Limari_mm_3sp.nam'
ml = flopy.modflow.Modflow.load(path)
dis=ml.dis
top = ml.dis.top.array
ibound=ml.bas6.ibound.array
top[ibound[0]==0]=np.nan
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(0,dis.ncol), np.arange(0,dis.nrow))  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X[:,1:], Y[:,1:], top[:,1:])
plt.show()
plt.xlabel('dX')
plt.ylabel('dY')
ax = hf.gca(projection='3d')
ax.set_zlabel('Elevación (m.s.n.m.)')
ax.set_title('Top Layer 1')

# plot celdas activas
plt.figure()
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound()

# export vtk
model_hk_dir = os.path.join('.', "ActiveCells")
ml.bas6.ibound.export(
    model_hk_dir, smooth=True, fmt="vtk", name="ActiveCells", point_scalars=True
)
ml.riv.export(
    model_hk_dir, smooth=True, fmt="vtk", name="Rivers", point_scalars=True
)