# -*- coding: utf-8 -*-
"""
DELAUNAY TRIANGULATION

Created on Sat Jan 04 15:00:10 2014

@author: Pio Calderon
"""

from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import networkx as nx
import extractregions
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from descartes.patch import PolygonPatch

#countez = 0

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
    for region_samples in sample_list.values():
        land_use.append(return_land_use_of_region(region_samples, mode))   
    return land_use

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
            
def construct_delaunay_network_of_region(region, region_samples, region_land_use, interior_polygons):
    """
    Construct the delaunay network of a region from region samples and land use.
    """

    region_boundary = Polygon(region)

#    fig = plt.figure(figsize = (10, 10), facecolor = "black")
#    ax = fig.add_subplot(111)#fig.add_axes([0, 0, 1, 1])
#    ax.set_xlim(0,dimensions[0])
#    ax.set_ylim(0,dimensions[1])    
#    patch = PolygonPatch(region_boundary, fc = "red", ec = "green",\
#                aa=False)
#    ax.add_patch(patch)
#    plt.plot()

    
    for interior in interior_polygons:
        if interior.within(region_boundary):
            region_boundary = region_boundary.difference(interior)
    
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
    radius = 1000  

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

def construct_delaunay_networks(sample_list, land_use, regions, interior_indices):

    networks = []
    interior_polygons = []
        
    for index in interior_indices:
        interior_polygons.append(Polygon(regions[index]))
    for index in [i for i in xrange(len(regions)) if i not in interior_indices]:
        region = regions[index]
        region_samples = sample_list[index]
        region_land_use = land_use[index]
        try:
            network = construct_delaunay_network_of_region(region, region_samples, region_land_use, interior_polygons)
            networks.append(network)
        except:
            pass
    return networks
    
def plot_region(network, fig, ax):
    """
    Plots a region, colored by phenotype.
    """

#    global countez

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
            patch = PolygonPatch(polygon, fc = color_dict[maxIndex], ec = "black",\
            aa=False)
            ax.add_patch(patch)
            
#            plt.savefig(str(countez).zfill(2) + ".png")
#            countez += 1
            
        except AssertionError:
            for p in polygon:
                patch = PolygonPatch(p, fc = color_dict[maxIndex], ec = "black",\
                aa=False)
                ax.add_patch(patch)

def plot_regions(networks, regionpath, dimensions, num, run, iteration, dist, scale):
    fig = plt.figure(figsize = (10, 10), facecolor = "black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0,dimensions[0])
    ax.set_ylim(0,dimensions[1])    
    for network in networks:
        plot_region(network, fig, ax)

    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig('{0}_sample{1}_run{2}_step{3}_{4}_std{5}.png'.\
        format(regionpath[:-4], num, str(run).zfill(2), str(iteration).zfill(6), dist, scale), \
        bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    
    regionpath = "visayas.png"
    samplepath = "samples.png"
    dist = "uniform"
    scale = 0.2
    numType=3
    run=0
    numSample=100
    regions, interior_indices, dimensions = extractregions.extract_regions(regionpath)
    sampleList = extractregions.extract_samples(samplepath, regions)
#    sampleList = return_sample_list(numSample, regions, dist, scale)
    landUse = return_land_use(sampleList, numType)
    networks = construct_delaunay_networks(sampleList, landUse, regions, interior_indices)  
    plot_regions(networks, regionpath, dimensions, numSample, run, 0, dist, scale)
