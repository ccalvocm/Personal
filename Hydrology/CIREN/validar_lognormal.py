# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:09:07 2021

@author: Carlos
"""
# =========
# Librerias
# =========
from scipy.stats import lognorm

# =========
# Funciones
# =========

def lognorm_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale

def main():
    mode = 123
    
    stddev = 99
    
    sigma, scale = lognorm_params(mode, stddev)
    
    sample = lognorm.rvs(sigma, 0, scale, size=1000000)

if __name__ == '__main__':
    main()