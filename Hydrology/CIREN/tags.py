# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:13:28 2021

@author: farrospide
"""
import itertools
import string

valid_chars = set(string.ascii_lowercase + string.digits) - set('lio01')
unique_id_generator = itertools.combinations(valid_chars, 8)
# Probably would want to persist the used values by using some sort of database/file
# instead of this
used = set()


tags = []

for x in range(1000000):
    

    generated = "".join(next(unique_id_generator))
    while generated in used:
        generated = "".join(next(unique_id_generator))
    
    # Once an unused value has been found, add it to used list (or some sort of database where you can keep track)
    used.add(generated)
    tags.append(generated)