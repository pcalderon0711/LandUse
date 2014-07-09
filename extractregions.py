# -*- coding: utf-8 -*-
"""
Created on Sun Jul 06 22:39:01 2014

@author: Pio Calderon
"""

import numpy as np
import Image
from skimage.measure import find_contours
from shapely.geometry import Polygon, Point

def extract_regions(filepath):
    """
    Extracts the habitable regions from filepath and returns the (x,y)
    coords in a list.
    """   
    img = Image.open(filepath)
    img_as_array = np.array(img.convert('L').rotate(180)\
        .transpose(Image.FLIP_LEFT_RIGHT))
    
    # x and y are inverted
    inv_regions = find_contours(img_as_array, 0.5)
    
    # container for corrected (x, y)'s
    regions = []
    
    # loop to reverse each (x, y)
    for region in inv_regions:        
        current = []        
        for y, x in region:            
            current.append([x, y])          
        regions.append(current)
      
    return regions, img.size
    
def extract_samples(samplepath, regions):
    """
    Extract samples from samplepath and group them (in lists) according
    to their region of membership.
    """    
    sample_img = Image.open(samplepath)
    sample_as_array = np.array(sample_img.convert('L').rotate(180)\
        .transpose(Image.FLIP_LEFT_RIGHT))

    region_polygons = []
    for region in regions:
        region_polygons.append(Polygon(region))
    
    samples = np.nonzero(sample_as_array)
    samples = zip(samples[1], samples[0])
    
#    samples = [(x, y) for x in xrange(int(sample_as_array.shape[0])) \
#        for y in xrange(int(sample_as_array.shape[1])) if \
#        sample_as_array[x, y] != 0]

    sorted_samples = [[]] * len(regions)
    
    i = 0
    print len(samples)
    for sample in samples:
        #print i
        i+=1
        for index, region_polygon in enumerate(region_polygons):
            if region_polygon.contains(Point(sample)):
                sorted_samples[index].append(sample)
                break
    
    return sorted_samples
    
