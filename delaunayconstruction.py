# -*- coding: utf-8 -*-
"""
DELAUNAY TRIANGULATION

Created on Sat Jan 04 15:00:10 2014

@author: Pio Calderon
"""

from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from descartes.patch import PolygonPatch
from extractregions import extract_regions

def return_sample_list(num, regions, dist, scale):
    """
    Return a list containing the position (x,y) of the samples.
    Output is a list of lists: the ith list contains samples for
    the ith region.
    """
    sample_list = []
    for index, region in enumerate(regions):
        min_x = min([region[i][0] for i in xrange(len(region))])
        max_x = max([region[i][0] for i in xrange(len(region))])
        min_y = min([region[i][1] for i in xrange(len(region))])
        max_y = max([region[i][1] for i in xrange(len(region))])
        poly = Polygon(region)
        region_samples = []
        while len(region_samples) != num:
            if dist == "normal":            
                candidate = (np.random.normal(loc = (min_x + max_x)/2, scale = scale), \
                np.random.normal(loc = (min_y + max_y)/2, scale = scale))            
            if dist == "uniform":           
                candidate = (np.random.uniform(min_x, max_x),np.random.uniform(min_y, max_y))                        
            if poly.contains(Point(candidate)):            
                region_samples.append(candidate)
        sample_list.append(region_samples)
 
    print "-done sampling-"
    return sample_list    

#def return_sample_list(total, shape, dist, scale):
#    """
#    Return a list containing the position (x,y) of the samples.
#    """
#    
#    if shape == "circle":
#        p = Point(0.5,0.5)
#        shape = p.buffer(.5)
#    elif shape == "square":
#        shape = Polygon([(0., 0.), (0., 1.), (1., 1.), (1., 0.)])
#        
#    sampleList = []
#    while len(sampleList) != total:
#        if dist == "normal":            
#            candidate = (np.random.normal(loc = 0.5, scale = scale), \
#            np.random.normal(loc = 0.5, scale = scale))            
#        if dist == "uniform":           
#            candidate = (np.random.uniform(),np.random.uniform())                        
#        if shape.contains(Point(candidate)):            
#            sampleList.append(candidate)
# 
#    print "-done sampling-"
#    return sampleList

def return_land_use_of_region(region_samples, numType, mode=0):
    """
    Return the landuse type of the samples in a single region
    in the format (type1,type2,type3).
    """
    totalUse = {}
    for index1, sample in enumerate(region_samples):   
        sampleUse = []             
        radius = np.sqrt((sample[0]-0.5)**2 + (sample[1]-0.5)**2)

        for index in range(numType): 
            if mode == 0:              
                sampleUse.append(np.random.uniform()) # RUN 01                    
            elif mode == 1:
                if index == 0:
                    sampleUse.append(5*np.random.uniform()) # RUN 02
                else:
                    sampleUse.append(np.random.uniform())
            elif mode == 2:
                if index == 0 or index == 1:
                    sampleUse.append(5*np.random.uniform()) # RUN 03
                else:
                    sampleUse.append(np.random.uniform())
            elif mode == 3:
                if index == 0:
                    if np.sqrt((sample[0]-0.5)**2 + (sample[1]-0.5)**2) < 0.2: # RUN 04
                        sampleUse.append(5*np.random.uniform())
                    else:
                        sampleUse.append(0)
                else:
                    sampleUse.append(np.random.uniform())
            elif mode == 4:
                if index == 0:
                    if radius < 0.167: # RUN 05
                        sampleUse.append(10*np.random.uniform())
                    else:
                        sampleUse.append(np.random.uniform())
                if index == 2:
                    if 0.167 <= radius < 0.334:
                        sampleUse.append(10*np.random.uniform())
                    else:
                        sampleUse.append(np.random.uniform())
                if index == 1:
                    if 0.334 <= radius:
                        sampleUse.append(10*np.random.uniform())
                    else:
                        sampleUse.append(np.random.uniform())
            elif mode == 5:
                if index == 0:
                    if sample[0] < 0.333: # RUN 06
                        sampleUse.append(10*np.random.uniform())
                    else:
                        sampleUse.append(np.random.uniform())
                if index == 1:
                    if 0.333 <= sample[0] < 0.666:
                        sampleUse.append(10*np.random.uniform())
                    else:
                        sampleUse.append(np.random.uniform())
                if index == 2:
                    if 0.666 <= sample[0]:
                        sampleUse.append(10*np.random.uniform())
                    else:
                        sampleUse.append(np.random.uniform())
            
        total = sum(sampleUse)
        sampleUse = [1.0*sampleUse[i]/total for i in xrange(len(sampleUse))]
        totalUse[index1] = sampleUse
        
    return totalUse
    
def return_land_use(sample_list, mode):
    land_use = []
    for region_samples in sample_list:
        land_use.append(return_land_use_of_region(region_samples, mode))   
    return land_use

#def return_land_use(sampleList, numType, occupationProbability):
#    """
#    Return the landuse type of the samples in the format (type1,type2,type3).
#    """
#    
#    totalUse = {}
#    for index1, sample in enumerate(sampleList):       
#        if np.random.uniform() > occupationProbability:
#      #      totalUse.append([0,0,0,1]) # unoccupied
#            continue               
# 
#        sampleUse = []             
#        radius = np.sqrt((sample[0]-0.5)**2 + (sample[1]-0.5)**2)
#
#        for index in range(numType):               
#        #    if index == numType - 1:
#        #        sampleUse.append(0)
#        #        continue            
#            sampleUse.append(np.random.uniform()) # RUN 01                    
#            
##            if index == 0:
##                sampleUse.append(5*np.random.uniform()) # RUN 02
##            else:
##                sampleUse.append(np.random.uniform())
##            """
##            """
##            if index == 0 or index == 1:
##                sampleUse.append(5*np.random.uniform()) # RUN 03
##            else:
##                sampleUse.append(np.random.uniform())
##            """
##            """
##            if index == 0:
##                if np.sqrt((sample[0]-0.5)**2 + (sample[1]-0.5)**2) < 0.2: # RUN 04
##                    sampleUse.append(5*np.random.uniform())
##                else:
##                    sampleUse.append(0)
##            else:
##                sampleUse.append(np.random.uniform())
##            """
##            """
##            if index == 0:
##                if radius < 0.167: # RUN 05
##                    sampleUse.append(10*np.random.uniform())
##                else:
##                    sampleUse.append(np.random.uniform())
##            if index == 2:
##                if 0.167 <= radius < 0.334:
##                    sampleUse.append(10*np.random.uniform())
##                else:
##                    sampleUse.append(np.random.uniform())
##            if index == 1:
##                if 0.334 <= radius:
##                    sampleUse.append(10*np.random.uniform())
##                else:
##                    sampleUse.append(np.random.uniform())
##            """
##            """
##            if index == 0:
##                if sample[0] < 0.333: # RUN 06
##                    sampleUse.append(10*np.random.uniform())
##                else:
##                    sampleUse.append(np.random.uniform())
##            if index == 1:
##                if 0.333 <= sample[0] < 0.666:
##                    sampleUse.append(10*np.random.uniform())
##                else:
##                    sampleUse.append(np.random.uniform())
##            if index == 2:
##                if 0.666 <= sample[0]:
##                    sampleUse.append(10*np.random.uniform())
##                else:
##                    sampleUse.append(np.random.uniform())
##            
#        total = sum(sampleUse)
#        sampleUse = [1.0*sampleUse[i]/total for i in xrange(len(sampleUse))]
#        totalUse[index1] = sampleUse
#        
#    return totalUse

# http://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci
def find_neighbors(pindex, triangulation):

    neighbors = []
    for triangle in triangulation.vertices:
        if pindex in triangle:
            neighbors.extend([triangle[i] for i in range(len(triangle)) if \
                triangle[i] != pindex])

    return list(set(neighbors)) #index of points in sampleList
    
def return_neighbor_list(region_samples):
    tri = Delaunay(np.array(region_samples))
    neighbor_list_of_region = [find_neighbors(point, tri) for point in range(len(\
        region_samples))]

    return neighbor_list_of_region   
            
def construct_delaunay_network_of_region(region, region_samples, region_land_use):
    """
    Construct the delaunay network of a region from region samples and land use.
    """
    region_boundary = Polygon(region)
    totalArea = region_boundary.area
    G = nx.Graph()
    region_samples = np.asarray(region_samples)
    
    multiCount = 0    
    multiLookUp = []    
    
    # construct voronoi object
    vor = Voronoi(region_samples) 
    # return neighborlist for each node in network
    neighborList = return_neighbor_list(region_samples) 

# FROM https://gist.github.com/pv/8036995#file-colorized_voronoi-py-L48 
    
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    radius = 100  

    # Construct a map containing all ridges for a given point
    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))    

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append([p1,vertices])
            continue
        
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
 
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
 
            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
 
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
 
        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
 
        # finish
        new_regions.append([p1, new_region.tolist()])
    
    # iterate over each voronoi region
    for sampleIndex, simplex in new_regions:     
        if sampleIndex in region_land_use.keys():
            simplex = np.asarray(simplex)            
            # check if region contains vertex at infinity
            if np.all(simplex >= 0):                 
                # container of voronoi vertices per region
                pointList = []                
                for component in simplex:
                    pointList.append((new_vertices[component][0], new_vertices\
                        [component][1]))

                if len(pointList) > 2:

                    polygon = Polygon(pointList)
                    polygon = region_boundary.intersection(polygon)
                    if polygon.geom_type == 'MultiPolygon':
                        for index, p in enumerate(polygon):
                            area = p.area
                            pointList = list(p.exterior.coords)
                            coordSample = p.centroid
                            if index == 0:
                                network_index = sampleIndex
                            else:
                                network_index = len(region_samples) + multiCount
                                multiCount += 1
                            G.add_node(network_index, position = coordSample, area = area, density = 1.0 * area / totalArea,
                                       landUse = region_land_use[sampleIndex], pointList = pointList, polygon = polygon)
                            multiLookUp.append(sampleIndex)
                                       
                    else:
                        area = polygon.area
                        #sampleIndex = list(lookup).index(index)
                        coordSample = region_samples[sampleIndex]
                        G.add_node(sampleIndex, position = coordSample, area = area, density = 1.0 * area / totalArea, landUse = region_land_use[sampleIndex]\
                            , pointList = pointList, polygon = polygon)  
    
    minimum = len(region_samples)
    for index in xrange(minimum, len(G.nodes())):
        neighbors = []
        starting_from_zero = index - minimum
        focusPolygon = G.node[index]['polygon']
        for nindex in neighborList[starting_from_zero]:
            neighborPolygon = G.node[nindex]['polygon']
            if focusPolygon.touches(neighborPolygon):
                neighbors.append(nindex)
        neighborList.append(neighbors)
        
    for index in G.nodes():
        
        for nindex in range(len(list(neighborList[index]))):

            if index != neighborList[index][nindex] and neighborList[index]\
                [nindex] in G.nodes():

                    G.add_edge(index, neighborList[index][nindex])

    print "-done constructing network-"
    return G
    
#def construct_delaunay_network(sampleList, landUse, shape):
#    """
#    Construct the delaunay network from the samplelist and landuse.
#    """
#
#    if shape == "circle":
#
#        p = Point(0.5,0.5)
#        shape = p.buffer(.5)
#
#    elif shape == "square":
#
#        shape = Polygon([(0., 0.), (0., 1.), (1., 1.), (1., 0.)])   
#        
#    totalArea = shape.area
#    G = nx.Graph()
#    sampleList = np.asarray(sampleList)
#    # construct voronoi object
#    vor = Voronoi(sampleList) 
#    # return neighborlist for each node in network
#    neighborList = return_neighbor_list(sampleList) 
#
## FROM https://gist.github.com/pv/8036995#file-colorized_voronoi-py-L48 
#    
#    new_regions = []
#    new_vertices = vor.vertices.tolist()
#    center = vor.points.mean(axis=0)
#    radius = 100  
#
#    # Construct a map containing all ridges for a given point
#    all_ridges = {}
#
#    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
#
#        all_ridges.setdefault(p1, []).append((p2, v1, v2))
#        all_ridges.setdefault(p2, []).append((p1, v1, v2))    
#
#    for p1, region in enumerate(vor.point_region):
#
#        vertices = vor.regions[region]
#
#        if all([v >= 0 for v in vertices]):
#
#            # finite region
#            new_regions.append([p1,vertices])
#            continue
#
#        # reconstruct a non-finite region
#        ridges = all_ridges[p1]
#        new_region = [v for v in vertices if v >= 0]
# 
#        for p2, v1, v2 in ridges:
#
#            if v2 < 0:
#
#                v1, v2 = v2, v1
#
#            if v1 >= 0:
#
#                # finite ridge: already in the region
#                continue
# 
#            # Compute the missing endpoint of an infinite ridge
#            t = vor.points[p2] - vor.points[p1] # tangent
#            t /= np.linalg.norm(t)
#            n = np.array([-t[1], t[0]])  # normal
# 
#            midpoint = vor.points[[p1, p2]].mean(axis=0)
#            direction = np.sign(np.dot(midpoint - center, n)) * n
#            far_point = vor.vertices[v2] + direction * radius
#            new_region.append(len(new_vertices))
#            new_vertices.append(far_point.tolist())
# 
#        # sort region counterclockwise
#        vs = np.asarray([new_vertices[v] for v in new_region])
#        c = vs.mean(axis=0)
#        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
#        new_region = np.array(new_region)[np.argsort(angles)]
# 
#        # finish
#        new_regions.append([p1, new_region.tolist()])
#    
#    # iterate over each voronoi region
#    for sampleIndex, simplex in new_regions: 
#    
#        if sampleIndex in landUse.keys():
#
#            simplex = np.asarray(simplex)
#            
#            # check if region contains vertex at infinity
#            if np.all(simplex >= 0): 
#                
#                # container of voronoi vertices per region
#                pointList = []
#                
#                for component in simplex:
#
#                    pointList.append((new_vertices[component][0], new_vertices\
#                        [component][1]))
#
#                if len(pointList) > 2:
#
#                    polygon = Polygon(pointList)
#                    polygon = shape.intersection(polygon)
#                    area = polygon.area
#                    #sampleIndex = list(lookup).index(index)
#                    corrSample = sampleList[sampleIndex]
#                    G.add_node(sampleIndex, position = corrSample, area =\
#                        1.0 * area / totalArea, landUse = landUse[sampleIndex]\
#                        , pointList = pointList, polygon = polygon)    
#        
#    for index in G.nodes():
#        
#        for nindex in range(len(list(neighborList[index]))):
#
#            if index != neighborList[index][nindex] and neighborList[index]\
#                [nindex] in G.nodes():
#
#                    G.add_edge(index, neighborList[index][nindex])
#
#    print "-done constructing network-"
#    return G

def construct_delaunay_networks(sample_list, land_use, regions):
    networks = []
    for index in xrange(len(regions)):
        region = regions[index]
        region_samples = sample_list[index]
        region_land_use = land_use[index]
        network = construct_delaunay_network_of_region(region, region_samples, region_land_use)
        networks.append(network)
    return networks
    
def plot_region(network, fig, ax):
    """
    Plots a region, colored by phenotype.
    """

    color_dict = {
        0 : 'red',
        1 : 'green',
        2 : 'blue',
        3 : 'black',
        4 : 'blue',
        5 : 'brown'
    }
    
    pointList = nx.get_node_attributes(network, 'pointList')
    pointList = pointList.values()

    for index in network.nodes():
        if list(network.node[index]['landUse']).count(max(network.node[index]\
            ['landUse'])) == 1:
            maxIndex = network.node[index]['landUse'].index(max(network.node[\
            index]['landUse']))
        else:
            withMaxLandUse = []
            for index2, value in enumerate(network.node[index]['landUse']):
                if value == max(network.node[index]['landUse']):
                    withMaxLandUse.append(index2)
            maxIndex = np.random.choice(withMaxLandUse)

        polygon = network.node[index]['polygon']
        try:
            patch = PolygonPatch(polygon, fc = color_dict[maxIndex], ec = "none",\
            aa=False)
            ax.add_patch(patch)
        except AssertionError:
            for p in polygon:
                patch = PolygonPatch(p, fc = color_dict[maxIndex], ec = "none",\
                aa=False)
                ax.add_patch(patch)

def plot_regions(networks, dimensions, num, run, iteration, dist, scale):
    fig = plt.figure(figsize = (10, 10), facecolor = "black")
    ax = fig.add_subplot(111)#fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0,dimensions[0])
    ax.set_ylim(0,dimensions[1])    
    for network in networks:
        plot_region(network, fig, ax)
    #ax.add_patch(PolygonPatch( Polygon([(0., 0.), (0., 1.), (1., 1.), (1., 0.)]), fc = 'none'))
    ax.set_aspect('equal')
    plt.legend()
    plt.axis('off')
    plt.savefig('graph_sample{0}_run{1}_step{2}_{3}_std{4}.png'.\
        format(num, str(run).zfill(2), str(iteration).zfill(6), dist, scale), \
        bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    plt.close()

#def plot_regions_shaded(numSample, network, shape, dist, scale, name, run, op):
#    """
#    Plots the map, colored by phenotype.
#    """
#
#    color_dict = {
#        0 : 'red',
#        1 : 'green',
#        2 : 'blue',
#        3 : 'black',
#        4 : 'blue',
#        5 : 'brown'
#    }
#    
#    fig = plt.figure(figsize = (10, 10))
#    ax = fig.add_axes([0, 0, 1, 1])
#    pointList = nx.get_node_attributes(network, 'pointList')
#    pointList = pointList.values()
#
#    for index in network.nodes():
#
#        if list(network.node[index]['landUse']).count(max(network.node[index]\
#            ['landUse'])) == 1:
#
#            maxIndex = network.node[index]['landUse'].index(max(network.node[\
#            index]['landUse']))
#
#        else:
#
#            withMaxLandUse = []
#
#            for index2, value in enumerate(network.node[index]['landUse']):
#
#                if value == max(network.node[index]['landUse']):
#
#                    withMaxLandUse.append(index2)
#
#            maxIndex = np.random.choice(withMaxLandUse)
#
#        polygon = network.node[index]['polygon']
#        patch = PolygonPatch(polygon, fc = color_dict[maxIndex], ec = "none",\
#            aa=False)
#        ax.add_patch(patch)
#        
#    #ax.add_patch(PolygonPatch( Polygon([(0., 0.), (0., 1.), (1., 1.), (1., 0.)]), fc = 'none'))
#    ax.set_aspect('equal')
#    ax.set_adjustable('box-forced')
#    plt.legend()
#    plt.axis('off')
#    plt.savefig('map_sample{0}_{1}_{2}_scale{3}_step{4}_run{5}_op{6}.png'.\
#        format(numSample, shape, dist, scale, str(name).zfill(6), \
#        str(run).zfill(2), op), bbox_inches='tight', pad_inches=0)
#    plt.close()
            
if __name__ == "__main__":
    path = "c.png"
    dist = "uniform"
    scale = 0.2
    numType=3
    run=0
    numSample=100
    regions, dimensions = extract_regions(path)
    sampleList = return_sample_list(numSample, regions, dist, scale)
    landUse = return_land_use(sampleList, numType)
    networks = construct_delaunay_networks(sampleList, landUse, regions)  
    plot_regions(networks, dimensions, numSample, run, 0, dist, scale)

