#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 5/31/2021
The function for smoothing func.gii by calling external wb_command

Author: dzhi
'''
import os


def smooth(file=None, base_surf=None, outfile_name='smoothed_func.gii', kernel=12):
    if file is None or base_surf is None:
        raise TypeError("Must give a file to smooth or the base surf.gii!")
    else:
        command = "wb_command -metric-smoothing \
        %s %s %d %s -fix-zeros" % (base_surf, file, kernel, outfile_name)
        os.system(command)
