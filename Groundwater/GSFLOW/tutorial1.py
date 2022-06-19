# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 20:13:56 2022

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

def main():
    # crear un objeto modelo gsflow
    os.chdir(r'D:\GitHub\pygsflow\examples')
    model_ws = os.path.join("data", "sagehen", "gsflow")
    control_file = os.path.join(model_ws, "saghen_new_cont.control")
    gsf = gsflow.GsflowModel.load_from_file(control_file)
    
    exe_name = os.path.join("..", "bin", "gsflow.exe")
    if platform.system().lower() == "windows":
        exe_name += ".exe"
    
    #load & run
    gsf = gsflow.GsflowModel.load_from_file(control_file, gsflow_exe=exe_name)
    gsf.run_model(silent=True)
    
    # outputs
    stat_var = gsf.prms.get_StatVar()
    
    # load csv
    csv_name = os.path.join(model_ws, "saghen_new_csv_output.csv")
    df = pd.read_csv(csv_name,index_col=0,parse_dates=True)
    
    # read heads
    head_name = os.path.join(model_ws, "saghen_new.hds")
    head = flopy.utils.HeadFile(head_name)
    
    cbc_name = os.path.join(model_ws, "saghen_new.cbc")
    cbc = flopy.utils.CellBudgetFile(cbc_name)
    
    # crear un nuevo objeto modelo gsflow en blanco
    control = gsflow.ControlFile(records_list=[])
    gsf = gsflow.GsflowModel(control)
    
    # cargar modelo
    model_ws = os.path.join("data", "sagehen", "gsflow")
    control_file = os.path.join(model_ws, "saghen_new_cont.control")
    gsf = gsflow.GsflowModel.load_from_file(control_file)
    
    # control file
    control = gsf.control
    csv_out = control.get_values("csv_output_file")
    
    # setear parametros
    csv_out = "gsflow_example.csv"
    control.set_values("csv_output_file", [csv_out,])
    
    # remover parámetros
    control.remove_record("csv_output_file")
    
    # prms
    model_ws = os.path.join("data", "sagehen", "gsflow")
    control_file = os.path.join(model_ws, "saghen_new_cont.control")
    gsf = gsflow.GsflowModel.load_from_file(control_file)
    
    # parámetros prms
    params = gsf.prms.parameters
    param_names = params.parameters_list
    
    # obtener valores de los parámetros
    ssr2gw = params.get_values("ssr2gw_rate")
    
    # modificar parámetros y sobrescribirlos
    ssr2gw = params.get_values("ssr2gw_rate")
    ssr2gw *= 0.8
    params.set_values("ssr2gw_rate", ssr2gw)
    
    # remover parámetros
    params.remove_record("ssr2gw_rate")
    
    # agregar mas parametros
    nhru = gsf.mf.nrow * gsf.mf.ncol
    ssr2gw = np.random.random(nhru)
    params.add_record("ssr2gw_rate",
                      ssr2gw,
                      dimensions=[["nhru", nhru]],
                      datatype=3)
    
    # la parte de modflow
    model_ws = os.path.join("..", "data", "sagehen", "gsflow")
    control_file = os.path.join(model_ws, "saghen_new_cont.control")
    gsf = gsflow.GsflowModel.load_from_file(control_file)
    
    # llamar al objeto modelo MODFLOW
    ml = gsf.mf
    
    # llamar a los paquetes del objeto modelo MODFLOW
    dis = gsf.mf.dis
    
    # model top
    top = ml.dis.top.array
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(0,dis.ncol), np.arange(0,dis.nrow))  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X[:,1:], Y[:,1:], top[:,1:])
    plt.show()
    plt.xlabel('dX')
    plt.ylabel('dY')
    ax = hf.gca(projection='3d')
    ax.set_zlabel('Elevación (m)')
    
    # obtener los stress periods y crear paquete well
    spd = {i: [[0, 30, 30, -150.],] for i in range(ml.nper)}
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=spd)
    ml.write_input()
    
    # plotting
    workspace = os.path.join(".", "data", "sagehen", "gsflow")
    control = "saghen_new_cont.control"
    
    gsf = GsflowModel.load_from_file(os.path.join(workspace, control))
    # leer MODFLOW y prms
    mf = gsf.mf
    # dis de MODFLOW
    prms_dis = PrmsDiscretization.load_from_flopy(mf)
    prms_dis.plot_discretization()
    
    # leer dis de shape
    workspace = os.path.join(".", "data", "sagehen", "shapefiles")
    shp = "hru_params.shp"
    
    prms_dis = PrmsDiscretization.load_from_shapefile(os.path.join(workspace, shp))
    prms_dis.plot_discretization()
    
    # remember, prms_dis was loaded from our shapefile
    plot = PrmsPlot(prms_dis)
    PrmsPlot
    
    plot = PrmsPlot(prms_dis)
    
    fig = plt.figure(figsize=(8, 6))
    
    # let's grab a parameter to plot
    ssr2gw = gsf.prms.parameters.get_record("ssr2gw_rate")
    
    # mask out 0 value and set the colormap to viridis
    ax = plot.plot_parameter(ssr2gw, masked_values=[0], cmap="viridis")
    plt.colorbar(ax)
    
    # plotear ajuste de lluvia
    rain_adj = gsf.prms.parameters.get_record("rain_adj")
    
    # mask out 0 value and set the colormap to jet
    ax = plot.plot_parameter(rain_adj, masked_values=[0], cmap="jet")
    
    # contour de parametros
    plot = PrmsPlot(prms_dis)
    
    fig = plt.figure(figsize=(8, 6))
    
    # let's grab a parameter to plot
    ssr2gw = gsf.prms.parameters.get_record("ssr2gw_rate")
    
    # set the colormap to viridis
    ax = plot.contour_parameter(ssr2gw, cmap="viridis")
    plt.colorbar(ax)
    
    # contour en varias dimensiones
    rain_adj = gsf.prms.parameters.get_record("rain_adj")
    
    # mask out 0 value and set the colormap to jet
    ax = plot.contour_parameter(rain_adj, masked_values=[0], cmap="jet")
    
    # plot array
    array = np.random.rand(prms_dis.nhru) * 10
    
    fig = plt.figure(figsize=(8, 6))
    plot = PrmsPlot(prms_dis)
    ax = plot.plot_array(array)
    plt.colorbar(ax)
    
    # contour de array
    fig = plt.figure(figsize=(8, 6))
    ax = plot.contour_array(array)
    plt.colorbar(ax)
    
    # plot modflow con propiedad PrmsPlot de Prms
    fig = plt.figure(figsize=(12, 12))
    
    # get the gsflow modflow object and pass it to flopy's ModelMap
    ml = gsf.mf
    ml.modelgrid.set_coord_info(xoff=prms_dis.extent[0], yoff=prms_dis.extent[2])
    m_map = PlotMapView(model=ml)
    m_map.plot_ibound()
    # get the current working matplotlib axes object
    flopy_ax = plt.gca()
    
    # let's grab a prms parameter to plot
    ssr2gw = gsf.prms.parameters.get_record("ssr2gw_rate")
    plot = PrmsPlot(prms_dis)
    # let's pass the flopy_ax to contour_parameter()
    ax = plot.contour_parameter(ssr2gw, ax=flopy_ax, masked_values=[0], cmap="viridis", alpha=0.5)
    plt.title("PRMS ssr2gw_rate contour with MODFLOW IBOUND array and CHD cells")
    plt.colorbar(ax, shrink=0.75)
    
    #%% trabajando con archivos vectoriales
    # define our model workspace and spatial information
    ws = os.path.abspath(os.path.dirname(sys.argv[0]))
    model_ws = os.path.join("data", "sagehen", "gsflow")
    
    xll = 438979.0
    yll = 3793007.75
    angrot = 4
    
    # load the model
    gsf = gsflow.GsflowModel.load_from_file(os.path.join(model_ws, 'saghen_new_cont.control'))
    
    # set model grid coordinate info
    gsf.mf.modelgrid.set_coord_info(xll, yll, 4)
    
    rch = gsf.mf.sfr.reach_data
    
    # set up an array of iseg numbers to visualize connectivity
    arr = np.zeros((1, gsf.mf.nrow, gsf.mf.ncol), dtype='int')
    for rec in rch:
        arr[0, rec.i, rec.j] = rec.iseg
    
    # use flopy to plot the SFR segments
    fig = plt.figure(figsize=(12, 12))
    pmv = flopy.plot.PlotMapView(model=gsf.mf)
    collection = pmv.plot_array(arr, masked_values=[0,], cmap='plasma')
    pmv.plot_inactive()
    pmv.plot_grid(lw=0.25)
    plt.colorbar(collection, shrink=0.75)
    
    #%% crear MODSIM
    # creating a MODSIM instance is as easy as
    modsim = gsflow.modsim.Modsim(gsf)
    shp_file = os.path.join(ws, "data", "temp", "sagehen_vectors.shp")
    modsim.write_modsim_shapefile(shp_file)
    vectors=geopandas.read_file(shp_file)
    
    # use flopy to plot the SFR segments
    fig,ax = plt.subplots(figsize=(10, 10))
    pmv = flopy.plot.PlotMapView(model=gsf.mf)
    # collection = pmv.plot_array(arr, masked_values=[0,], cmap='plasma')
    pmv.plot_inactive()
    pmv.plot_grid(lw=0.25)
    pmv.plot_bc(package=gsf.mf.sfr, color='gainsboro')
    vectors.plot(ax = ax)
    
    plt.title("Stream vector plot")
    
if __name__ == '__main__':
    main()