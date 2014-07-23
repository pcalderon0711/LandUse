# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 23:49:28 2014

@author: Pio Calderon
"""
import os
import extractregions
import delaunayconstruction
    
pwd = os.getcwd()
imgpath = pwd + '\\img_test\\'

single_convex = "single_convex.png"
single_concave = "single_concave.png"
single_hole = "single_hole.png"
single_holes = "single_holes.png"
single_concave_complex1 = "single_concave_complex1.png"
single_concave_complex2 = "single_concave_complex2.png"
single_concave_complex3 = "single_concave_complex3.png"

sample_single_convex = "sample_single_convex.png"
sample_single_concave = "sample_single_concave.png"
sample_single_hole = "sample_single_hole.png"
sample_single_holes = "sample_single_holes.png"
sample_single_concave_complex1 = "sample_single_concave_complex1.png"
sample_single_concave_complex2 = "sample_single_concave_complex2.png"
sample_single_concave_complex3 = "sample_single_concave_complex3.png"

multi_convex = "multi_convex.png"
multi_concave = "multi_concave.png"
multi_hole = "multi_hole.png"
multi_holes = "multi_holes.png"
multi_concave_complex1 = "multi_concave_complex1.png"
multi_concave_complex2 = "multi_concave_complex2.png"
multi_concave_complex3 = "multi_concave_complex3.png"

sample_multi_convex = "sample_multi_convex.png"
sample_multi_concave = "sample_multi_concave.png"
sample_multi_hole = "sample_multi_hole.png"
sample_multi_holes = "sample_multi_holes.png"
sample_multi_concave_complex1 = "sample_multi_concave_complex1.png"
sample_multi_concave_complex2 = "sample_multi_concave_complex2.png"
sample_multi_concave_complex3 = "sample_multi_concave_complex3.png"

single_paths = [(single_convex, sample_single_convex), 
                (single_concave, sample_single_concave),
                (single_hole, sample_single_hole),
                (single_holes, sample_single_holes),
                (single_concave_complex1, sample_single_concave_complex1),
                (single_concave_complex2, sample_single_concave_complex2),
                (single_concave_complex3, sample_single_concave_complex3)]

multi_paths = [(multi_convex, sample_multi_convex), 
                (multi_concave, sample_multi_concave),
                (multi_hole, sample_multi_hole),
                (multi_holes, sample_multi_holes),
                (multi_concave_complex1, sample_multi_concave_complex1),
                (multi_concave_complex2, sample_multi_concave_complex2),
                (multi_concave_complex3, sample_multi_concave_complex3)]

paths = single_paths + multi_paths

dist = "uniform"
scale = 0.2

numType=3
run=0
numSample=100

for path, samplepath in paths:
    path = imgpath + path
    print path
    extractregions.randomly_sample(path, 1000)
    samplepath = imgpath + samplepath  
    print samplepath
    regions, interior_indices, dimensions = extractregions.extract_regions(path)
    sampleList = extractregions.extract_samples(samplepath, regions) 
    landUse = delaunayconstruction.return_land_use(sampleList, numType)
    networks = delaunayconstruction.construct_delaunay_networks(sampleList, landUse, regions, interior_indices)  
    delaunayconstruction.plot_regions(networks, path, dimensions, numSample, run, 0, dist, scale)