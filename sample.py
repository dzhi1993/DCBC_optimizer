#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Sep 30 15:11:29 2020
Example of how to use DCBC function to evaluation expected cortical parcellations

Author: Da Zhi
'''

from eval_DCBC import DCBC
from plotting import plot_wb_curve
import nibabel as nb
import numpy as np


if __name__ == "__main__":
    print('Start evaluating DCBC sample code ...')

    mat = nb.load('parcellations/Power2011.32k.L.label.gii')
    parcels = [x.data for x in mat.darrays]
    parcels = np.reshape(parcels, (len(parcels[0]),))

    # Create a DCBC evaluation object of the desired evaluation parameters(left hemisphere)
    myDCBC = DCBC(hems='L', maxDist=35, binWidth=2.5, dist_file="distanceMatrix/distSphere_sp.mat")

    # Do the valuation on the parcellation
    T = myDCBC.evaluate(parcels)

    plot_wb_curve(T, path='data', hems='all')
    print('Done')


