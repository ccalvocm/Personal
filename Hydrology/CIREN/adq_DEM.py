#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:48:49 2021

@author: faarrosp
"""

import geopandas as gpd
import contextily as ctx
import os
import rasterio as rio
from matplotlib import pyplot as plt
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
import rasterio.mask
from rasterio.plot import show
from rasterio.enums import Resampling
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas.plotting import table

