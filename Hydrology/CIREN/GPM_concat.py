#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:47:00 2021

@author: faarrosp
"""

import os

# Script to concatenate in time the GPM files


# change directory to where GPM files are located
os.chdir('/home/faarrosp/Insync/farrospide@ciren.cl/OneDrive Biz - Shared/' + \
         'AOHIA_ZC/Etapa 1 y 2/GIS/GPM')

# get all the GPM rasters
list_gpm_ncfiles = [x for x in os.listdir() if (x.endswith('.nc4') and x.startswith('3B'))]
list_gpm_ncfiles.sort()

# concatenate using cdo
rasters = ' '.join(list_gpm_ncfiles)
command = ' '.join(['cdo cat', rasters, './concat/GPM_concat.nc'])
os.system(command)

 
