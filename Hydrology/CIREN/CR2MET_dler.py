# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:20:06 2021

@author: Carlos
"""

import requests #librería para hacer consultas https
import re
import sys


def download_url(url, save_path, headers, year):
    session = requests.Session()
    response = session.get(url, headers={'User-Agent': 'Mozilla/5.0'})

    with open(save_path+"\\TCR2MET"+str(year)+".zip","wb") as zipf:
         for chunk in response.iter_content():
             zipf.write(chunk)
    zipf.close()
            
def bajarCR2(year, directorio,prod_dl):
    headers  = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    for i in range(1,50):
        print(i)
        r = requests.get('http://www.cr2.cl/datos-productos-grillados/?cp_cr2met='+str(i), headers = headers, stream = True)
        sc = r.text
        if 'http://www.cr2.cl/download/'+prod_link+str(year) in sc:
            link = 'http://www.cr2.cl/download/'+prod_link+re.search('http://www.cr2.cl/download/'+prod_link+'(.+?)\';', sc).group(1)
            download_url(link, directorio, headers, year)
            sys.exit('Descarga completa')
            
def main():
    dicc_prod = {'pp' : 'cr2met_v2-0_pr_day_1979_', 'tmin' : 'cr2met_v2-0_tmin_day-_1979_', 'tmax' : 'cr2met_v2-0_tmax_day_1979_'}
    prod = input('Especifique el producto por favor')
    prod_link = dicc_prod[prod]
    yr = input('Especifique el último año por favor')
    directorio = input('Especifique la carpeta de descarga')
    bajarCR2(yr,directorio,prod_link)

if __name__ == '__main__':
    main()
