# -*- coding: utf-8 -*-
"""
Created on Sun Jul 06 22:39:01 2014

@author: Pio Calderon
"""

import numpy as np
import Image
from skimage.measure import find_contours

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