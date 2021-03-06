#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 4/30/2021

The optimization algorithm to fine-tune functional boundaries on the cortical surface

Author: dzhi
'''

import numpy as np
import nibabel as nb
from helper_function import compute_similarity, smoothing, generate

from eval_DCBC import DCBC, compute_var_cov
from scipy.io import savemat, loadmat
import scipy.io as spio
import scipy as sp
from scipy.spatial import SphericalVoronoi


def load_labelgii(file=None):
    '''
    Helper function that load the given label gifti file

    :param file:  the file path of label gifti
    :return: the loaded labels, shape: [N, ]
    '''
    if file is not None:
        mat = nb.load(file)
        parcels = [x.data for x in mat.darrays]
        parcels = np.reshape(parcels, (len(parcels[0]),))
    else:
        raise TypeError("Must give the path of input label.gii file!")

    return parcels


def load_funcgii(file=None):
    '''
    Helper function that load the given func gifti file

    :param file:  the file path of func gifti
    :return: the loaded function profile, shape: [N, num_conditions]
    '''
    if file is not None:
        mat = nb.load(file)
        wbeta_data = [x.data for x in mat.darrays]
        wbeta = np.reshape(wbeta_data, (len(wbeta_data), len(wbeta_data[0])))
        data = wbeta.transpose()
    else:
        raise TypeError("Must give the path of input func.gii file!")

    return data


def load_surfgii(file=None):
    '''
    Helper function that load the given surf gifti file

    :param file:  the file path of surf gifti
    :return: the loaded surf data, vertices and faces
    '''
    if file is not None:
        mat = nb.load(file)
        wbeta_data = [x.data for x in mat.darrays]
        vertices = wbeta_data[0]
        faces = wbeta_data[1]
    else:
        raise TypeError("Must give the path of input func.gii file!")

    return vertices, faces


def check_boundaries(this_label, neighbours):
    '''
    Boolean function that check whether the given vertex is a boundary node

    :param this_label:  the label of given node
    :param neighbours:  the labes of all neighbouring nodes

    :return: True - if the given node is at boundaires
             False - Otherwise
    '''
    result = np.all(neighbours == this_label)
    return not result


def split_neighbours(parcels,this_node,neighours):
    within = []
    between = []

    for v in neighours:
        if parcels[v] == parcels[this_node]:
            within.append(v)
        else:
            between.append(v)

    return np.asarray(within), np.asarray(between)


def compute_neighbouring(file):
    '''
    compute the neighbouring matrix of cortical mash

    :param file: the cortical mash file (e.g surf.gii)
    :return:     the vertices neighbouring matrix, shape = [N,N],
                 N is the number of vertices

    '''
    mat = nb.load(file)
    surf = [x.data for x in mat.darrays]

    surf_vertices = surf[0]
    surf_faces = surf[1]

    neigh = np.zeros([surf_vertices.shape[0], surf_vertices.shape[0]])
    for idx in range(surf_faces.shape[0]):
        # the surf faces usually shape of [N,3] because a typical mesh is a triangle
        # so this part can be simpled by just combination of the three nodes.
        connected = surf_faces[idx]
        combination = np.array(np.meshgrid(connected, connected)).T.reshape(-1, 2)
        for i in range(combination.shape[0]):
            neigh[combination[i][0]][combination[i][1]] = 1

    np.fill_diagonal(neigh, 0)
    return neigh


def vertex_dcbc(parcels, neigh_matrix, connectivity, dist=None,
                kernel=60, bin=10, func=None):
    boundaries = []
    new_parcel = np.copy(parcels)
    cor = np.corrcoef(func)
    numBins = int(np.floor(kernel / bin))

    # Find the vertices index at the boundaries
    for v in range(parcels.shape[0]):
        neighbours = np.where(neigh_matrix[v] == 1)
        if check_boundaries(parcels[v], parcels[neighbours]):
            boundaries = np.append(boundaries, v)

    boundaries = boundaries.astype(np.int)

    for v in boundaries:  # main loop of iterating all vertices at boundaries
        connecting = np.where(neigh_matrix[v] == 1)  # Check all neighbouring vertices
        neighbours = np.where(dist[v] <= kernel)
        v_idx = np.where(neighbours[0] == v)

        # Making subset of dist, parcels, and func matrix within current radius
        this_dist = dist[:,neighbours[0]]
        this_dist = this_dist[neighbours[0],:]
        this_parcels = parcels[neighbours]
        this_func = func[neighbours]



        T = compute_DCBC(maxDist=kernel, binWidth=bin, dist=this_dist,
                         parcellation=this_parcels, func=this_func)

        curr_dcbc = T['DCBC']

        # Only check for the boundaries vertices
        within_connecting, between_connecting = split_neighbours(parcels, v, np.asarray(connecting).flatten())

        for par in np.unique(parcels[between_connecting]):
            # Change this vertex's label to each of the connecting parcel
            changed_parcels = np.copy(this_parcels)
            changed_parcels[v_idx] = par
            this_T = compute_DCBC(maxDist=kernel, binWidth=bin, dist=this_dist,
                         parcellation=changed_parcels, func=this_func)
            this_dcbc = this_T['DCBC']
            if this_dcbc > curr_dcbc:
                new_parcel[v] = par
                curr_dcbc = this_dcbc

        # within_neighbours, between_neighbours = split_neighbours(parcels, v, np.asarray(neighbours).flatten())
        # if not np.size(within_connecting):
        #     # if this node has no neighbouring vertices with same label
        #     # parcels[v] = np.bincount(parcels[within_neighbours]).argmax()
        #     max_r = -2
        # else:
        #     max_r = np.mean(connectivity[v][within_neighbours])  # presume the max correlation is within R
        #
        # for par in np.unique(parcels[between_connecting]):
        #     this_nb = np.extract(np.equal(parcels[between_neighbours], par), between_neighbours)
        #     this_r_between = np.mean(connectivity[v][this_nb])
        #     if this_r_between > max_r:
        #         new_parcel[v] = par
        #         max_r = this_r_between

    return new_parcel


def compute_DCBC(maxDist=35, binWidth=1, parcellation=np.empty([]),
                 func=None, dist=None, weighting=True):
    """
    Constructor of DCBC class
    :param hems:        Hemisphere to test. 'L' - left hemisphere; 'R' - right hemisphere; 'all' - both hemispheres
    :param maxDist:     The maximum distance for vertices pairs
    :param binWidth:    The spatial binning width in mm, default 1 mm
    :param parcellation:
    :param dist_file:   The path of distance metric of vertices pairs, for example Dijkstra's distance, GOD distance
                        Euclidean distance. Dijkstra's distance as default
    :param weighting:   Boolean value. True - add weighting scheme to DCBC (default)
                                       False - no weighting scheme to DCBC
    """

    numBins = int(np.floor(maxDist / binWidth))

    cov, var = compute_var_cov(func)
    cor = np.corrcoef(func)

    # remove the nan value and medial wall from dist file
    row, col, distance = sp.sparse.find(dist)

    # making parcellation matrix without medial wall and nan value
    par = parcellation
    num_within, num_between, corr_within, corr_between = [], [], [], []
    for i in range(numBins):
        inBin = np.where((distance > i * binWidth) & (distance <= (i + 1) * binWidth))[0]

        # lookup the row/col index of within and between vertices
        within = np.where((par[row[inBin]] == par[col[inBin]]) == True)[0]
        between = np.where((par[row[inBin]] == par[col[inBin]]) == False)[0]

        # retrieve and append the number of vertices for within/between in current bin
        num_within = np.append(num_within, within.shape[0])
        num_between = np.append(num_between, between.shape[0])

        # Compute and append averaged within- and between-parcel correlations in current bin
        # this_corr_within = np.nanmean(cov[row[inBin[within]], col[inBin[within]]]) / np.nanmean(var[row[inBin[within]], col[inBin[within]]])
        # this_corr_between = np.nanmean(cov[row[inBin[between]], col[inBin[between]]]) / np.nanmean(var[row[inBin[between]], col[inBin[between]]])

        this_corr_within = np.nanmean(cor[row[inBin[within]], col[inBin[within]]])
        this_corr_between = np.nanmean(cor[row[inBin[between]], col[inBin[between]]])

        corr_within = np.append(corr_within, this_corr_within)
        corr_between = np.append(corr_between, this_corr_between)

        del inBin

    if weighting:
        weight = 1/(1/num_within + 1/num_between)
        weight = weight / np.sum(weight)
        DCBC = np.nansum(np.multiply((corr_within - corr_between), weight))
    else:
        DCBC = np.nansum(corr_within - corr_between)
        weight = np.nan

    D = {
        "binWidth": binWidth,
        "maxDist": maxDist,
        "num_within": num_within,
        "num_between": num_between,
        "corr_within": corr_within,
        "corr_between": corr_between,
        "weight": weight,
        "DCBC": DCBC
    }

    return D


if __name__ == "__main__":
    print('Start DCBC optimizing on simulation ...')

    surf = 'simulation/Sphere.6k.L.surf.gii'
    dist = spio.loadmat("simulation/distSphere_6k.mat")
    dist = dist['distGOD']
    random_func_file = 'simulation/test_boundaries.func.gii'
    smoothed_func_file = 'simulation/test_boundaries_smoothed.func.gii'
    # the destination boundaries
    boundary = load_labelgii('simulation/test_Icosahedron-42.6k.L.label.gii')

    # Making functional map with estimate boundaries and smoothing
    estimate_func = generate.generate_random_map(num_nodes=boundary.shape[0], num_con=34,
                                                 outfile_name=random_func_file, parcel=boundary,
                                                 boundaries=True, num_iter=4)
    smoothing.smooth(file=random_func_file, base_surf=surf,
                     outfile_name=smoothed_func_file, kernel=5)
    estimate_func_smoothed = load_funcgii(smoothed_func_file)

    # Start optimizing from random Icosahedron to destination parcellation
    neigh = compute_neighbouring(surf)
    r = compute_similarity.compute_similarity(files=smoothed_func_file,
                                              type='pearson', dense=False)

    rotations = loadmat('simulation/Icos_rotation.mat')['icos']
    dcbc = np.empty((4,))
    for i in range(10):
        parcels = rotations[:, i]
        # parcels = load_labelgii('simulation/test_Icosahedron-42.6k.L.label.gii')
        original = compute_DCBC(maxDist=60, binWidth=10, dist=dist,
                                parcellation=parcels, func=estimate_func)

        global_dcbc = original['DCBC']
        fine_tune_parcels = np.empty((parcels.shape[0],))
        for k in range(15):  # Main loop
            print("Fine-tuning the DCBC optimization, iter #", k)
            parcels = vertex_dcbc(parcels=parcels, neigh_matrix=neigh, connectivity=r,
                                  dist=dist, kernel=60, bin=10, func=estimate_func)

            if (k+1) % 5 == 0:
                # Create a DCBC evaluation object of the desired evaluation parameters(left hemisphere)
                T = compute_DCBC(maxDist=60, binWidth=10, dist=dist, parcellation=parcels, func=estimate_func)
                global_dcbc = np.hstack((global_dcbc, T['DCBC']))
                fine_tune_parcels = np.vstack((fine_tune_parcels, parcels))

        dcbc = np.vstack((dcbc, global_dcbc))

        fine_tune_parcels = np.delete(fine_tune_parcels, 0, axis=0)
        fine_tune_parcels = fine_tune_parcels.transpose()
        outfile_name = "parcel_fineTune_iter_%d.mat" % i
        pdic = {"parcels": fine_tune_parcels}
        savemat(outfile_name, pdic)

    dcbc = np.delete(dcbc, 0, axis=0)
    mdic = {"dcbc": dcbc}
    savemat("dcbc_finTune_10_random.mat", mdic)
    print("END..")

