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


# funciones
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([0, z_middle + .33*plot_radius])
    
# modelo Limari
plt.close('all')
path=r'E:\WEAP\Weap Areas\Limari_PEGRH_20\MODFLOW\Limari_mm_3sp.nam'
ml = flopy.modflow.Modflow.load(path)
# setear las coordenadas de la grilla
xll=239000
yll=6659200
ml.modelgrid.set_coord_info(xll, yll, 0)
dis=ml.dis
top = ml.dis.top.array
ibound=ml.bas6.ibound.array
top[ibound[0]==0]=-999
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(xll,xll+dis.ncol*300,300), np.arange(yll,yll+dis.nrow*300,300))  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, top)
plt.show()
plt.xlabel('Este')
plt.ylabel('Norte')
ax = hf.gca(projection='3d')
ax.set_zlabel('Elevación (m.s.n.m.)')
ax.set_title('Top Layer 1')
set_axes_equal(ha)

#%% plot celdas activas
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

#%% check drains
ml.drn.plot(inactive='false')
# ml.drn.plot()
