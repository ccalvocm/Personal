#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:00:10 2021

@author: faarrosp
"""
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import plotly.graph_objects as go
from matplotlib import pyplot as plt
init_notebook_mode()


fig = go.Figure(go.Sankey(
    arrangement = "snap",
    node = {
        "label": ["Maipo en Las Melosas", #0
                  "Maipo en San Alfonso", #1
                  "Maipo en El Manzano", #2
                  "Maipo en La Obra", #3
                  "Embalse el Yeso", #4
                  "Volcan en Queltehues", #5
                  "Colorado antes junta Olivares", #6
                  "Olivares antes junta Colorado", #7
                  "Colorado en desembocadura", #8
                  "Canal de Pirque", #9
                  "Canal San Carlos" #10
                  ],
        # "x": [0.1, 0.3, 0.5, 0.7],
        # "y": [0.5, 0.5, 0.5, 0.5],
        'pad':20},  # 10 Pixels
    link = {
        "source": [0, 0, 1, 2, 4, 5, 6, 7, 8, 2, 2],
        "target": [1, 0, 2, 3, 1, 1, 8, 8, 2, 9, 10],
        "value":  [2,1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}))
plot(fig)  