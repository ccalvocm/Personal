# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:09:11 2021

@author: Carlos
"""

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding
from bibtexparser.customization import convert_to_unicode


with open('AOHZC2.bib') as bibtex_file:
    parser = BibTexParser()
    parser.customization = homogenize_latex_encoding
    parser.customization = convert_to_unicode
    bib_database = bibtexparser.load(bibtex_file)
    print(bib_database)
