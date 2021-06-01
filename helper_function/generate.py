#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 5/30/2021
The helper function of generating the random functional map of given size with,
or without the boundaries.

Author: dzhi
'''
import numpy as np
import nibabel as nb

import os


def makeFuncGifti(data, anatomicalStruct='CortexLeft', columnNames=[]):
    [N, Q] = [data.shape[0], data.shape[1]]
    # Make columnNames if empty
    if not columnNames:
        for i in range(Q):
            columnNames.append("col_{:02d}".format(i + 1))

    C = nb.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomicalStruct,
        'encoding': 'XML_BASE64_GZIP'})

    E = nb.gifti.gifti.GiftiLabel()
    E.key = 0
    E.label = '???'
    E.red = 1.0
    E.green = 1.0
    E.blue = 1.0
    E.alpha = 0.0

    D = list()
    for i in range(Q):
        d = nb.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_NONE',
            datatype='NIFTI_TYPE_FLOAT32',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': columnNames[i]})
        )
        D.append(d)

    S = nb.gifti.GiftiImage(meta=C, darrays=D)
    S.labeltable.labels.append(E)

    return S


def generate_random_map(mean=0, kernel=0.5, num_nodes=32492, num_con=34, boundaries=False,
                        parcel=None, outfile_name=None, output=True, num_iter=10):
    '''

    :param mean: the mean of random values
    :param kernel: the sigma of random values
    :param num_nodes: the number of vertices or locations
    :param num_con: the number of conditions
    :param boundaries: if True, the return will be a random functional map
                       with true boundaries. Otherwise, the return will be
                       a simple 2d random functional map
    :param parcel_file: The given true boundaries of the random functional map.
                        Only when the boundaries=True, parcel_file is not None.
    :param outfile_name: The name of output file
    :param output: If true, the function outputs a file

    :return: the expected random functional map
    '''
    random = np.random.normal(mean, kernel, size=(num_nodes, num_con))

    if not boundaries:  # no simulated functional boundaries needed
        if output:
            if outfile_name:
                func_out = makeFuncGifti(data=random)
                nb.save(func_out, outfile_name)
                return random
            else:
                raise TypeError("Must give a name of the output func.gii file!")
        else:
            return random
    else:
        # with estimate functional boundaries added by a given parcellation
        if parcel is None:
            raise TypeError("Must give a parcellation to generate functional boundaries!")
        else:
            for ite in range(num_iter):
                for i in range(num_con):
                    for k in np.unique(parcel):
                        this_mean = random[parcel == k, i].mean()
                        random[parcel == k, i] += this_mean

            if output:
                if outfile_name:
                    func_out = makeFuncGifti(data=random)
                    nb.save(func_out, outfile_name)
                else:
                    raise TypeError("Must give a name of the output func.gii file!")

            return random


