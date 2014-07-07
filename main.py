# -*- coding: utf-8 -*-
"""
Created on Sun Jan 05 00:32:07 2014

@author: Pio Calderon
"""

from osgeo import gdal
from osgeo import ogr
from rasterprocessing import do_probability_sampling
from delaunayconstruction import return_land_use, return_sample_list, construct_delaunay_networks, plot_regions
from axelrodmodel import run_model

from imageprocessing import plot_cosine_similarity
import matplotlib.pyplot as plt

#vector = ogr.Open(r'C:\Users\Pio Calderon\Desktop\mmshp\Regions_REGION__Metropolitan Manila.shp')
#raster = gdal.Open(r'C:\Users\Pio Calderon\Desktop\philpop\popmap10adj_MM.tif')

#sampleList = do_probability_sampling(vector, raster, 1000)
#sampleList=[(np.random.random(), np.random.random()) for i in range(50)]

"""
with open('sampleList.csv', 'r') as sl:
    a=sl.readlines()
for index, line in enumerate(a):
    a[index] = line.strip().split(',')
    if index > 0:
        for index2, entry in enumerate(a[index]):
            a[index][index2] = float(entry)
sampleList = a[1:]
"""

path = "c.png"
for numSample in [10]:
    for numType in [3]:
        for numStep in [5000]:
            for dist in ["uniform"]:
                run_model(path, numSample, numType, numStep, dist, 0.2, 10, stepDef="iteration")

#from PIL import Image
#import numpy as np
#
#print 'first part'
#for run in [1]:#range(5):
#    imgPath = 'a.png'
#    #imgPath = r'map_sample1000_square_normal_scale0.2_step000000_run0{0}.png'.format(run)
#    img=Image.open(imgPath)
#    arr=np.array(img)
#    ratio, similarityList = plot_cosine_similarity(arr,imgPath)
#    plt.plot(ratio, similarityList)
#    print "done: run "#, run
#plt.xlabel('frame length / image dimension')
#plt.ylabel('average cosine similarity')
#plt.savefig('cosinesimilarityrunaverage_{0}.png'.format('a'))#(r'map_sample1000_square_uniform_scale0.2_step000000'))
#plt.close()
#print 'second part'
#
#for run in [1]:#range(5):
#    imgPath = 'b.png'
#    #imgPath = r'map_sample1000_square_normal_scale0.2_step005000_run0{0}.png'.format(run)
#    img=Image.open(imgPath)
#    arr=np.array(img)
#    ratio, similarityList = plot_cosine_similarity(arr,imgPath)
#    plt.plot(ratio, similarityList)
#    print "done: run "#, run
#plt.xlabel('frame length / image dimension')
#plt.ylabel('average cosine similarity')
#plt.savefig('cosinesimilarityrunaverage_{0}.png'.format('b'))#r'map_sample1000_square_uniform_scale0.2_step005000'))
#plt.close()

#for run in [1]:#range(5):
#    imgPath = 'blank.png'
#    #imgPath = r'map_sample1000_square_normal_scale0.2_step005000_run0{0}.png'.format(run)
#    img=Image.open(imgPath)
#    arr=np.array(img)
#    ratio, similarityList = plot_cosine_similarity(arr,imgPath)
#    plt.plot(ratio, similarityList)
#    print "done: run "#, run
#plt.xlabel('frame length / image dimension')
#plt.ylabel('average cosine similarity')
#plt.savefig('cosinesimilarityrunaverage_{0}.png'.format('blank'))#r'map_sample1000_square_uniform_scale0.2_step005000'))
#plt.close()