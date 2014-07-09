# -*- coding: utf-8 -*-
"""
Created on Sun Dec 01 16:28:42 2013

@author: Pio Calderon
"""

import networkx as nx
import matplotlib.pyplot as plt
from delaunayconstruction import *
from extractregions import extract_regions

def run_model(path, numSample, numType, numStep, dist='uniform',\
            scale=0.0, numRun=1, stepDef = "step"):
    """
    Run the model.
    """
    
    regions, dimensions = extract_regions(path)
    
    runAreaDensity = []
    runAreaPhenotype = []
 
    if stepDef == "step":
        numIteration = numStep * numSample
    elif stepDef == "iteration":
        numIteration = numStep
        
    run = 0
    sampleList = return_sample_list(numSample, regions, dist, scale)
    landUse = return_land_use(sampleList, numType)

    while run < numRun:
        networks = construct_delaunay_networks(sampleList, landUse, regions) 
        #return_max(wmap, shape, dist, scale, 0, run)
        
        plot_regions(networks, dimensions, numSample, run, 0, dist, scale)
        
        areaDensity = []
        areaPhenotype = []
        
        areaDensity.append(get_land_area_per_type(networks, landUse))
        areaPhenotype.append(get_max_density_area(networks, landUse))
        
        iteration = 0
        while iteration < numIteration:
            
#            if iteration % numSample == 0:
#                for i in xrange(wmap.number_of_nodes()):
#                    do_transition_rules(wmap) #alternate per step
            region = np.random.choice(len(regions))
            agent = np.random.choice(landUse[region].keys())
                    
            do_transition_rules(networks[region], agent) #alternate per iteration
            do_axelrod(networks[region], agent, landUse[region])
            
            if stepDef == "step":  
                if iteration % numSample == 0:      
                    areaDensity.append(get_land_area_per_type(networks, \
                        landUse))
                    areaPhenotype.append(get_max_density_area(networks, \
                        landUse))
                    
            elif stepDef == "iteration":
                
                areaDensity.append(get_land_area_per_type(networks, landUse))
                areaPhenotype.append(get_max_density_area(networks, landUse))

            if iteration % 1000 == 0:

                print "Run: ", run, "; Step: ", iteration

            iteration += 1

#        return_max(wmap, shape, dist, scale, numStep, run)
        plot_regions(networks, numSample, run, iteration, dist, scale)
        plot_area_density(areaDensity, numSample, dist, scale, run)
        plot_area_phenotype(areaPhenotype, numSample, dist. scale, run)
        
        runAreaDensity.append(areaDensity)
        runAreaPhenotype.append(areaPhenotype)
       
        run += 1

    D_aDensityAverage, U_aDensityAverage, I_aDensityAverage = np.zeros(\
        numStep+1), np.zeros(numStep+1), np.zeros(numStep+1)
    D_aDensityStdErr, U_aDensityStdErr, I_aDensityStdErr = np.zeros(\
        numStep+1), np.zeros(numStep+1), np.zeros(numStep+1)
    D_aPhenotypeAverage, U_aPhenotypeAverage, I_aPhenotypeAverage = \
        np.zeros(numStep+1), np.zeros(numStep+1), np.zeros(numStep+1)
    D_aPhenotypeStdErr, U_aPhenotypeStdErr, I_aPhenotypeStdErr = \
        np.zeros(numStep+1), np.zeros(numStep+1), np.zeros(numStep+1)
    
    for step in xrange(numStep+1):

        adD, adU, adI = zip(*zip(*runAreaDensity)[step])
        apD, apU, apI = zip(*zip(*runAreaPhenotype)[step])
        
        D_aDensityAverage[step] = np.mean(adD)
        D_aPhenotypeAverage[step] = np.mean(apD)
        
        U_aDensityAverage[step] = np.mean(adU)
        U_aPhenotypeAverage[step] = np.mean(apU)
        
        I_aDensityAverage[step] = np.mean(adI)
        I_aPhenotypeAverage[step] = np.mean(apI)    

        D_aDensityStdErr[step] = np.std(adD)/np.sqrt(numRun)
        D_aPhenotypeStdErr[step] = np.std(apD)/np.sqrt(numRun)

        U_aDensityStdErr[step] = np.std(adU)/np.sqrt(numRun)
        U_aPhenotypeStdErr[step] = np.std(apU)/np.sqrt(numRun)

        I_aDensityStdErr[step] = np.std(adI)/np.sqrt(numRun)
        I_aPhenotypeStdErr[step] = np.std(apI)/np.sqrt(numRun)

    g=open('averageareadensity_sample{0}_{1}_scale{2}.csv'.format(\
        numSample,dist,scale),'w')
    g.write('step,d,u,i,stdd,stdu,stdi\n')

    for index in xrange(numStep+1):

        g.write('{0},{1},{2},{3}\n'.format(index,D_aDensityAverage[index]\
            ,U_aDensityAverage[index], I_aDensityAverage[index], \
            D_aDensityStdErr[index], U_aDensityStdErr[index], \
            I_aDensityStdErr[index]))
    g.close()

    g=open('averageareaphenotype_sample{0}_{1}_scale{2}.csv'.format(\
        numSample,dist,scale),'w')
    g.write('step,d,u,i,stdd,stdu,stdi\n')

    for index in xrange(numStep+1):

        g.write('{0},{1},{2},{3}\n'.format(index,D_aPhenotypeAverage[index]\
            ,U_aPhenotypeAverage[index], I_aPhenotypeAverage[index], \
            D_aPhenotypeStdErr[index], U_aPhenotypeStdErr[index], \
            I_aPhenotypeStdErr[index]))

    g.close()

    plt.errorbar(range(numStep+1), D_aDensityAverage, D_aDensityStdErr, \
        label = "D", color = "red")
    plt.errorbar(range(numStep+1), U_aDensityAverage, U_aDensityStdErr, \
        label = "U", color = "green")
    plt.errorbar(range(numStep+1), I_aDensityAverage, I_aDensityStdErr, \
        label = "I", color = "blue")
    plt.legend()
    plt.xlabel('time step')
    plt.ylabel('density')  
    plt.xlim(0,numStep)
    plt.ylim(-0.01,1.01)
    plt.savefig('averageareadensity{0}_{1}_scale{2}.png'.format(\
        numSample,dist,scale),bbox_inches=\
        'tight')    
    plt.close()  

    plt.errorbar(range(numStep+1), D_aPhenotypeAverage, D_aPhenotypeStdErr,\
        label = "D", color = "red")
    plt.errorbar(range(numStep+1), U_aPhenotypeAverage, U_aPhenotypeStdErr,\
        label = "U", color = "green")
    plt.errorbar(range(numStep+1), I_aPhenotypeAverage, I_aPhenotypeStdErr,\
        label = "I", color = "blue")
    plt.legend()
    plt.xlabel('time step')
    plt.ylabel('density')  
    plt.xlim(0,numStep)
    plt.ylim(-0.01,1.01)
    plt.savefig('averagephenotypedensity{0}_{1}_scale{2}.png'.format\
        (numSample,dist,scale),bbox_inches='tight')    
    plt.close()  


#def run_model(numSample, numType, numStep, shape, dist = "uniform",\
#            scale = 0.0, numRun = 1, occupationProbability = 0, \
#             stepDef = "step"):
#    """
#    Run the model.
#    """
#    
#    runAreaDensity = []
#    runAreaPhenotype = []
# 
#    if stepDef == "step":
#
#        numIteration = numStep * numSample
#
#    elif stepDef == "iteration":
#
#        numIteration = numStep
#        
#    run = 0
#    
#    while run < numRun:
#        
#        sampleList = return_sample_list(numSample, shape, dist, scale)
#        landUse = return_land_use(sampleList, numType, \
#            occupationProbability)
#        wmap = construct_delaunay_network(sampleList, landUse, shape)  
#        #return_max(wmap, shape, dist, scale, 0, run)
#        
#        plot_regions_shaded(numSample, wmap, shape, dist, scale, 0, \
#            run, occupationProbability)
#        
#        areaDensity = []
#        areaPhenotype = []
#        
#        areaDensity.append(get_land_area_per_type(wmap, landUse))
#        areaPhenotype.append(get_max_density_area(wmap, landUse))
#        
#        iteration = 0
#        while iteration < numIteration:
#            
##            if iteration % numSample == 0:
##                for i in xrange(wmap.number_of_nodes()):
##                    do_transition_rules(wmap) #alternate per step
#            
#            agent = np.random.choice(landUse.keys())
#                    
#            do_transition_rules(wmap, agent) #alternate per iteration
#            do_axelrod(wmap, agent, landUse)
#            
#            if stepDef == "step":
#                
#                if iteration % numSample == 0:
#                    
#                    areaDensity.append(get_land_area_per_type(wmap, \
#                        landUse))
#                    areaPhenotype.append(get_max_density_area(wmap, \
#                        landUse))
#                    
#            elif stepDef == "iteration":
#                
#                areaDensity.append(get_land_area_per_type(wmap, landUse))
#                areaPhenotype.append(get_max_density_area(wmap, landUse))
#
#            if iteration % 1000 == 0:
#
#                print "Run: ", run, "; Step: ", iteration
#
#            iteration += 1
#                    
##        return_max(wmap, shape, dist, scale, numStep, run)
#        plot_regions_shaded(numSample, wmap, shape, dist, scale, numStep,\
#            run, occupationProbability)
#        plot_area_density(numSample,wmap, areaDensity, shape, dist, scale,\
#            run, occupationProbability)
#        plot_area_phenotype(numSample,wmap, areaPhenotype, shape, dist, \
#            scale, run, occupationProbability)
#        
#        runAreaDensity.append(areaDensity)
#        runAreaPhenotype.append(areaPhenotype)
#       
#        run += 1
#
#    D_aDensityAverage, U_aDensityAverage, I_aDensityAverage = np.zeros(\
#        numStep+1), np.zeros(numStep+1), np.zeros(numStep+1)
#    D_aDensityStdErr, U_aDensityStdErr, I_aDensityStdErr = np.zeros(\
#        numStep+1), np.zeros(numStep+1), np.zeros(numStep+1)
#    D_aPhenotypeAverage, U_aPhenotypeAverage, I_aPhenotypeAverage = \
#        np.zeros(numStep+1), np.zeros(numStep+1), np.zeros(numStep+1)
#    D_aPhenotypeStdErr, U_aPhenotypeStdErr, I_aPhenotypeStdErr = \
#        np.zeros(numStep+1), np.zeros(numStep+1), np.zeros(numStep+1)
#    
#    for step in xrange(numStep+1):
#
#        adD, adU, adI = zip(*zip(*runAreaDensity)[step])
#        apD, apU, apI = zip(*zip(*runAreaPhenotype)[step])
#        
#        D_aDensityAverage[step] = np.mean(adD)
#        D_aPhenotypeAverage[step] = np.mean(apD)
#        
#        U_aDensityAverage[step] = np.mean(adU)
#        U_aPhenotypeAverage[step] = np.mean(apU)
#        
#        I_aDensityAverage[step] = np.mean(adI)
#        I_aPhenotypeAverage[step] = np.mean(apI)    
#
#        D_aDensityStdErr[step] = np.std(adD)/np.sqrt(numRun)
#        D_aPhenotypeStdErr[step] = np.std(apD)/np.sqrt(numRun)
#
#        U_aDensityStdErr[step] = np.std(adU)/np.sqrt(numRun)
#        U_aPhenotypeStdErr[step] = np.std(apU)/np.sqrt(numRun)
#
#        I_aDensityStdErr[step] = np.std(adI)/np.sqrt(numRun)
#        I_aPhenotypeStdErr[step] = np.std(apI)/np.sqrt(numRun)
#
#    g=open('averageareadensity_sample{0}_{1}_{2}_scale{3}.csv'.format(\
#        numSample,shape,dist,scale),'w')
#    g.write('step,d,u,i,stdd,stdu,stdi\n')
#
#    for index in xrange(numStep+1):
#
#        g.write('{0},{1},{2},{3}\n'.format(index,D_aDensityAverage[index]\
#            ,U_aDensityAverage[index], I_aDensityAverage[index], \
#            D_aDensityStdErr[index], U_aDensityStdErr[index], \
#            I_aDensityStdErr[index]))
#    g.close()
#
#    g=open('averageareaphenotype_sample{0}_{1}_{2}_scale{3}.csv'.format(\
#        numSample,shape,dist,scale),'w')
#    g.write('step,d,u,i,stdd,stdu,stdi\n')
#
#    for index in xrange(numStep+1):
#
#        g.write('{0},{1},{2},{3}\n'.format(index,D_aPhenotypeAverage[index]\
#            ,U_aPhenotypeAverage[index], I_aPhenotypeAverage[index], \
#            D_aPhenotypeStdErr[index], U_aPhenotypeStdErr[index], \
#            I_aPhenotypeStdErr[index]))
#
#    g.close()
#
#    plt.errorbar(range(numStep+1), D_aDensityAverage, D_aDensityStdErr, \
#        label = "D", color = "red")
#    plt.errorbar(range(numStep+1), U_aDensityAverage, U_aDensityStdErr, \
#        label = "U", color = "green")
#    plt.errorbar(range(numStep+1), I_aDensityAverage, I_aDensityStdErr, \
#        label = "I", color = "blue")
#    plt.legend()
#    plt.xlabel('time step')
#    plt.ylabel('normalized land area')  
#    plt.xlim(0,numStep)
#    plt.ylim(-0.01,1.01)
#    plt.savefig('averageareadensity{0}_{1}_{2}_scale{3}_op{4}.png'.format(\
#        numSample,shape,dist,scale,occupationProbability),bbox_inches=\
#        'tight')    
#    plt.close()  
#
#    plt.errorbar(range(numStep+1), D_aPhenotypeAverage, D_aPhenotypeStdErr,\
#        label = "D", color = "red")
#    plt.errorbar(range(numStep+1), U_aPhenotypeAverage, U_aPhenotypeStdErr,\
#        label = "U", color = "green")
#    plt.errorbar(range(numStep+1), I_aPhenotypeAverage, I_aPhenotypeStdErr,\
#        label = "I", color = "blue")
#    plt.legend()
#    plt.xlabel('time step')
#    plt.ylabel('normalized land area')  
#    plt.xlim(0,numStep)
#    plt.ylim(-0.01,1.01)
#    plt.savefig('averagephenotypedensity{0}_{1}_{2}_scale{3}_op{4}.png'.format\
#        (numSample,shape,dist,scale,occupationProbability),bbox_inches='tight')    
#    plt.close()  

def do_axelrod(network, agent, landUse):  
    """
    Perform the Axelrod step on the specified agent.
    """

    #agent = np.random.choice(network.nodes())
    numType = len(network.node[agent]['landUse'])
    
    if len(network.neighbors(agent)) == 0:

        return
    
    neighbor = np.random.choice(landUse.keys())
    
    while neighbor == agent:

        neighbor = np.random.choice(landUse.keys())
    
    #chosenType = np.random.choice(range(numType)) RANDOM PICKING TO
    #interactionProbability = sum([network.node[agent]['landUse'][i] * \
    #     network.node[neighbor]['landUse'][i] for i in xrange(numType)])
    #ABOVE AY UNG DIRECT DOT PRODUCT
    #interactionProbability = (1.0/numType)*sum([1.0*min(network.node[agent]\
    #    ['landUse'][i], network.node[neighbor]['landUse'][i])/max(network.node[agent]\
    #    ['landUse'][i], network.node[neighbor]['landUse'][i]) for i in xrange(numType)])
    #ABOVE AY ANG MIN/MAX RULE
    
    normActivated = np.sqrt(sum([component**2 for component in \
        network.node[agent]['landUse']]))
    normNeighbor = np.sqrt(sum([component**2 for component in \
        network.node[neighbor]['landUse']]))
    interactionProbability = (1./normActivated) * (1./normNeighbor) * \
        sum([network.node[agent]['landUse'][i] * network.node[neighbor]\
            ['landUse'][i] for i in xrange(numType)])
            
    if np.random.uniform(0,1) < interactionProbability:
        
        if network.node[neighbor]['landUse'].count(max(network.node[neighbor]\
            ['landUse'])) == 1:

            chosenType = network.node[neighbor]['landUse'].index(max(\
                network.node[neighbor]['landUse']))
                
        else:

            withMaxLandUse = []

            for index, value in enumerate(network.node[neighbor]['landUse']):

                if value == max(network.node[neighbor]['landUse']):

                    withMaxLandUse.append(index)

            chosenType = np.random.choice(withMaxLandUse)

        network.node[agent]['landUse'][chosenType] = network.node[neighbor]\
            ['landUse'][chosenType]
        norm = 1 - network.node[agent]['landUse'][chosenType]
        reducedSum = sum([network.node[agent]['landUse'][index] for index in \
                range(0, chosenType) + range(chosenType + 1, numType)])

        for index in range(0, chosenType) + range(chosenType + 1, numType):

            if reducedSum != 0:
                network.node[agent]['landUse'][index] = norm * \
                    (network.node[agent]['landUse'][index] / reducedSum)

            else:

                continue  

def do_transition_rules(network, agent):
    """
    Perform the local transition step between the agent and a randomly selected
    neighbor.
    """

    #agent = np.random.choice(network.nodes())
    agentLandUse = network.node[agent]['landUse']
    selected = []

    randomNumber1 = np.random.uniform()
    
     # d selected
    if randomNumber1 < agentLandUse[0]:

        selected.append(0)

    # u selected        
    elif agentLandUse[0] <= randomNumber1 < agentLandUse[0] + agentLandUse[1]:

        selected.append(1)
    
    # i selected    
    elif agentLandUse[0] + agentLandUse[1] <= randomNumber1 < 1:

        selected.append(2)        

    notSelected = [i for i in xrange(3) if i not in selected]
    normalization = sum(agentLandUse[i] for i in notSelected)
    randomNumber2 = np.random.uniform()

    if agentLandUse[notSelected[0]] == 0:

        if agentLandUse[notSelected[1]] == 0:

            return

        else:

            selected.append(notSelected[1])

    elif agentLandUse[notSelected[1]] == 0:

        selected.append(notSelected[0])

    else: 

        if randomNumber2 < agentLandUse[notSelected[0]] / normalization:

            selected.append(notSelected[0])

        else:

            selected.append(notSelected[1])

    if 0 in selected and 1 in selected:

        agentLandUse[0] = agentLandUse[0] + agentLandUse[1] * (agentLandUse[0])\
            / (agentLandUse[0] + agentLandUse[1])
        agentLandUse[1] = agentLandUse[1] - agentLandUse[1] * (agentLandUse[0])\
            / (agentLandUse[0] + agentLandUse[1])
#        agentLandUse[0] = 1. - agentLandUse[2]
#        agentLandUse[1] = 0

    elif 1 in selected and 2 in selected:

        agentLandUse[2] = agentLandUse[2] + agentLandUse[1] * (agentLandUse[2])\
            / (agentLandUse[2] + agentLandUse[1])
        agentLandUse[1] = agentLandUse[1] - agentLandUse[1] * (agentLandUse[2])\
            / (agentLandUse[2] + agentLandUse[1])
#        agentLandUse[1] = 0
#        agentLandUse[2] = 1. - agentLandUse[0]

    elif 0 in selected and 2 in selected:

        if agentLandUse[0] > agentLandUse[2]:

            agentLandUse[0] = agentLandUse[0] + agentLandUse[2] * \
                (agentLandUse[0])/(agentLandUse[2]+agentLandUse[0])
            agentLandUse[2] = agentLandUse[2] - agentLandUse[2] * \
                (agentLandUse[0])/(agentLandUse[2]+agentLandUse[0])

        else:

            return

#            agentLandUse[1] = 1. - agentLandUse[0]
#            agentLandUse[2] = 0

def get_land_area_per_type(networks, landUse):
    """
    Return the density of each land use type.
    """
    numType = 3
    total = np.array([0]*numType, dtype='f')
    region_area = np.array([0]*len(networks), dtype='f')
    for network_index, network in enumerate(networks):
        region_area[network_index] = sum(nx.get_node_attributes(network,'area'))
        # region_land_use = nx.get_node_attributes(network,'landUse')
        density = nx.get_node_attributes(network,'density')
        region_total = []
        for landType in xrange(numType):
            region_density = np.sum([np.array(landUse[i][landType]) * density[i] for \
                i in landUse[network_index].keys()])
            region_total.append(region_density)
        total += np.array(region_total) * (region_area)

    return list((total / sum(total)) / sum(region_area))
    
#def get_land_area_per_type(network, landUse):
#    """
#    Return the density of each land use type.
#    """
#
#    landUse1 = nx.get_node_attributes(network,'landUse')
#    area = nx.get_node_attributes(network,'area')
#    numType = len(landUse1[landUse.keys()[1]])
#    total = []
#
#    for landType in xrange(numType):
#
#        totalArea = np.sum([np.array(landUse[i][landType]) * area[i] for \
#            i in landUse.keys()])
#        total.append(totalArea)
#
#    return list(np.array(total)/sum(total))

def get_max_density_area(networks, landUse):
    """
    Def: phenotype of a parcel - landuse type with highest area in the parcel
    Return the area of each landuse type such that the landuse
    is considered the phenotype.
    """

    numType = 3
    density = np.array([0]*numType, dtype="f")  
    region_area = np.array([0]*len(networks), dtype="f")  
    for network_index, network in enumerate(networks):
        temp_density = np.array([0]*numType, dtype="f")  
        region_area[network_index] = sum(nx.get_node_attributes(network,'area'))
        for index in landUse[network_index].keys():
            if list(network.node[index]['landUse']).count(max(network.node[index]\
                ['landUse'])) == 1:
                maxIndex = list(network.node[index]['landUse']).index(max(\
                    network.node[index]['landUse']))
                temp_density[maxIndex] += network.node[index]['density']
            else:
                withMaxLandUse = []
                for index2, value in enumerate(network.node[index]['landUse']):
                    if value == max(n.node[index]['landUse']):
                        withMaxLandUse.append(index)
    
                # this involves double-counting
                for maxIndex in withMaxLandUse:            
                    temp_density[maxIndex] += network.node[index]['density']            
        density += temp_density * region_area
    return density/(sum(density) * sum(region_area))

def plot_area_density(density, numSample, dist, scale, run):
    """
    Plot the density of each landuse type.
    Save a csv file of the data in the plot.
    """

    a,b,c = zip(*density)    
    plt.plot(range(len(a)), a, color = 'red', label = 'D')
    plt.plot(range(len(b)), b, color = 'green', label = 'U')
    plt.plot(range(len(c)), c, color = 'blue', label = 'I')
    plt.legend()
    plt.xlabel('time step')
    plt.ylabel('density')
    plt.xlim(0, len(a))
    plt.ylim(-0.01, 1.01)
    plt.savefig('density_sample{0}_run{1}_dist{2}_scale{3}.png'.format(\
        numSample, str(run).zfill(2), dist, scale), bbox_inches=\
        'tight')    
    plt.close()    

    g=open('density_sample{0}_run{1}_dist{2}_scale{3}.csv'.format(\
        numSample, str(run).zfill(2), dist, scale),'w')
    g.write('step,d,u,i\n')

    for index in xrange(len(density)):

        g.write('{0},{1},{2},{3}\n'.format(index, a[index], b[index], c[index]))

    g.close()

#def plot_area_density(numSample, network, area, shape, dist, scale, run, op):
#    """
#    Plot the density of each landuse type.
#    Save a csv file of the data in the plot.
#    """
#
#    a,b,c = zip(*area)    
#    plt.plot(range(len(a)), a, color = 'red', label = 'D')
#    plt.plot(range(len(b)), b, color = 'green', label = 'U')
#    plt.plot(range(len(c)), c, color = 'blue', label = 'I')
#    plt.legend()
#    plt.xlabel('time step')
#    plt.ylabel('normalized land area')
#    plt.xlim(0, len(a))
#    plt.ylim(-0.01, 1.01)
#    plt.savefig('area_density{0}_{1}_{2}_scale{3}_run{4}_op{5}.png'.format(\
#        numSample, shape, dist, scale, str(run).zfill(2), op), bbox_inches=\
#        'tight')    
#    plt.close()    
#
#    g=open('area_density_sample{0}_{1}_{2}_scale{3}_run{4}_op{5}.csv'.format(\
#        numSample, shape, dist, scale, str(run).zfill(2), op),'w')
#    g.write('step,d,u,i\n')
#
#    for index in xrange(len(area)):
#
#        g.write('{0},{1},{2},{3}\n'.format(index, a[index], b[index], c[index]))
#
#    g.close()

def plot_area_phenotype(density, numSample, dist, scale, run):
    """
    Plot the phenotype of the map.
    Save a csv file of the data in the plot.
    """

    a,b,c = zip(*density)    
    plt.plot(range(len(a)), a, color = 'red', label = 'D')
    plt.plot(range(len(b)), b, color = 'green', label = 'U')
    plt.plot(range(len(c)), c, color = 'blue', label = 'I')
    plt.legend()
    plt.xlabel('time step')
    plt.ylabel('density')
    plt.xlim(0,len(a))
    plt.ylim(-0.01,1.01)
    plt.savefig('phenotype_sample{0}_run{1}_{2}_scale{3}.png'.format(\
        numSample, str(run).zfill(2), dist, scale), \
        bbox_inches='tight')    
    plt.close()   

    g=open('phenotype_sample{0}_run{1}_{2}_scale{3}.csv'.format(\
        numSample, str(run).zfill(2), dist, scale),'w')
    g.write('step,d,u,i\n')

    for index in xrange(len(density)):

        g.write('{0},{1},{2},{3}\n'.format(index, a[index], b[index], c[index]))

    g.close()

#def plot_area_phenotype(numSample, network, area, shape, dist, scale, run,op):
#    """
#    Plot the phenotype of the map.
#    Save a csv file of the data in the plot.
#    """
#
#    a,b,c = zip(*area)    
#    plt.plot(range(len(a)), a, color = 'red', label = 'D')
#    plt.plot(range(len(b)), b, color = 'green', label = 'U')
#    plt.plot(range(len(c)), c, color = 'blue', label = 'I')
#    plt.legend()
#    plt.xlabel('time step')
#    plt.ylabel('normalized land area')
#    plt.xlim(0,len(a))
#    plt.ylim(-0.01,1.01)
#    plt.savefig('area_phenotype{0}_{1}_{2}_scale{3}_run{4}_op{5}.png'.format(\
#        numSample, shape, dist, scale, str(run).zfill(2), op), \
#        bbox_inches='tight')    
#    plt.close()   
#
#    g=open('area_phenotype_sample{0}_{1}_{2}_scale{3}_run{4}_op{5}.csv'.format(\
#        numSample, shape, dist, scale, str(run).zfill(2), op),'w')
#    g.write('step,d,u,i\n')
#
#    for index in xrange(len(area)):
#
#        g.write('{0},{1},{2},{3}\n'.format(index, a[index], b[index], c[index]))
#
#    g.close()

#def plot_as_network(wmap, name): # as a network
#    color = []
#    for i in wmap.nodes():
#        color.append(20*list(wmap.node[i]['landUse']).index(max(wmap.node[i]\
#            ['landUse'])))
#    positions = return_positions(wmap)
#    nx.draw_networkx_edges(wmap, pos = positions, edge_color = 'gray', width = 0.8)
#    nx.draw_networkx_nodes(wmap, pos = positions, node_color = color, \
#        linewidths = 0, node_size=50)
#    plt.axis('off')
#    plt.tight_layout()
#    plt.savefig('{0}.png'.format(str(name).zfill(6)), bbox_inches='tight')
#    plt.close()
#
#
#def return_positions(wmap):
#    positions = [wmap.node[i]['position'] for i in range(len(wmap.nodes()))]
#    return positions
#   
#
#def do_transition_rules(network):
#    agent = np.random.choice(network.nodes())
#    if list(network.node[agent]['landUse']).count(max(network.node[agent]['landUse'])) == 1:
#        agentPhenotype = list(network.node[agent]['landUse']).index(max(network.node[agent]['landUse']))
#    else:
#        withMaxLandUse = []
#        for index, value in enumerate(network.node[agent]['landUse']):
#            if value == max(network.node[agent]['landUse']):
#                withMaxLandUse.append(index)
#        agentPhenotype = np.random.choice(withMaxLandUse)       
#    agentPhenotype = list(network.node[agent]['landUse']).index(max(network.node[agent]['landUse']))
#    neighborhoodPhenotype = []
#    for neighbor in network.neighbors(agent):
#        if list(network.node[neighbor]['landUse']).count(max(network.node[neighbor]['landUse'])) == 1:
#            neighborhoodPhenotype.append([neighbor,list(network.node[agent]['landUse']).index(max(network.node[agent]['landUse']))])
#        else:
#            withMaxLandUse = []
#            for index, value in enumerate(network.node[neighbor]['landUse']):
#                if value == max(network.node[neighbor]['landUse']):
#                    withMaxLandUse.append(index)
#            neighborhoodPhenotype.append([neighbor,np.random.choice(withMaxLandUse)])
#    phenotypeCount = [[neighborhoodPhenotype[1] for i in xrange(len(neighborhoodPhenotype))].count(i) for i in [0,1,2]]        
#    maxPhenotype = phenotypeCount.index(max(phenotypeCount))
#    neighborsWithMaxPhenotype = [neighborhoodPhenotype[0] for i in xrange(len(neighborhoodPhenotype)) if neighborhoodPhenotype[1] == maxPhenotype]
#    
#    encroachmentProbability = 1.0*max(phenotypeCount)/len(network.neighbors(agent))
#    
#    if agentPhenotype == maxPhenotype:
#        pass
#    
#    if agentPhenotype == 0 and maxPhenotype == 1 and np.random.uniform() < encroachmentProbability:
#        neighbor = np.random.choice(neighborsWithMaxPhenotype)
#        network.node[neighbor]['landUse'][0] = 1        
#        network.node[neighbor]['landUse'][1] = 0
#        network.node[neighbor]['landUse'][2] = 0
#                      
#    if agentPhenotype == 0 and maxPhenotype == 2 and np.random.uniform() < encroachmentProbability:
#        neighbor = np.random.choice(neighborsWithMaxPhenotype)
#        network.node[neighbor]['landUse'][0] = 0        
#        network.node[neighbor]['landUse'][1] = 0
#        network.node[neighbor]['landUse'][2] = 1
#    
#    if agentPhenotype == 1 and maxPhenotype == 0 and np.random.uniform() < encroachmentProbability:
#        network.node[neighbor]['landUse'][0] = 1
#        network.node[neighbor]['landUse'][1] = 0
#        network.node[neighbor]['landUse'][2] = 0        
#    
#    if agentPhenotype == 1 and maxPhenotype == 2 and np.random.uniform() < encroachmentProbability:
#        network.node[neighbor]['landUse'][0] = 0
#        network.node[neighbor]['landUse'][1] = 0
#        network.node[neighbor]['landUse'][2] = 1        
#
#    if agentPhenotype == 2 and maxPhenotype == 0 and np.random.uniform() < encroachmentProbability:
#        network.node[neighbor]['landUse'][0] = 0
#        network.node[neighbor]['landUse'][1] = 1
#        network.node[neighbor]['landUse'][2] = 0        
#    
#    if agentPhenotype == 2 and maxPhenotype == 1 and np.random.uniform() < encroachmentProbability:
#        network.node[neighbor]['landUse'][0] = 0
#        network.node[neighbor]['landUse'][1] = 0
#        network.node[neighbor]['landUse'][2] = 1      

#def return_max(network,shape, dist, scale, step, run):
#    g=open('maxlist_sample{0}_{1}_{2}_scale{3}_step{4}_run{5}.csv'.format(nx.number_of_nodes(network),shape,dist,scale,str(step).zfill(6), str(run).zfill(2)),'w')
#    g.write('index,max\n')
#    for index in xrange(network.number_of_nodes()):
#        maximum = list(network.node[index]['landUse']).index(max(network.node[index]['landUse']))
#        g.write('{0},{1}\n'.format(index,maximum))
#    g.close()
#        not working have to change 
#
#def get_average_land_use_density(network, landUse):
#    landUse1 = nx.get_node_attributes(network,'landUse')
#    numType = len(landUse1[landUse.keys()[1]])
#    average = []
#    for landType in xrange(numType):
#        averageValue = np.mean([landUse[i][landType] for i in landUse.keys()])
#        average.append(averageValue)
#    return average    
#def get_max_density_count(network, landUse):
#    maxList = []    
#    for index in landUse.keys():
#        if list(network.node[index]['landUse']).count(max(network.node[index]['landUse'])) == 1:
#            maxList.append(list(network.node[index]['landUse']).index(max(network.node[index]['landUse'])))
#        else:
#            withMaxLandUse = []
#            for index2, value in enumerate(network.node[index]['landUse']):
#                if value == max(n.node[index]['landUse']):
#                    withMaxLandUse.append(index)
#            maxList.append(np.random.choice(withMaxLandUse))   
#    landUse1 = nx.get_node_attributes(network,'landUse')
#    numType = len(landUse1[landUse.keys()[1]])
#    return 1.0* np.array([maxList.count(i) for i in range(numType)])/network.number_of_nodes()
#
#def plot_density(numSample, network,densityList, shape, dist, scale, run):
#    a,b,c = zip(*densityList)    
#    plt.plot(range(len(a)), a, color = 'red', label = 'D')
#    plt.plot(range(len(b)), b, color = 'green', label = 'U')
#    plt.plot(range(len(c)), c, color = 'blue', label = 'I')
#    plt.legend()
#    plt.xlabel('time step')
#    plt.ylabel('type density')  
#    plt.xlim(0,len(a))
#    plt.ylim(-0.01,1.01)
#    plt.savefig('density{0}_{1}_{2}_scale{3}_run{4}.png'.format(numSample,shape,dist,scale,str(run).zfill(2)),bbox_inches='tight')    
#    plt.close()      
#
#    g=open('density_sample{0}_{1}_{2}_scale{3}_run{4}.csv'.format(numSample,shape,dist,scale,str(run).zfill(2)),'w')
#    g.write('step,d,u,i\n')
#    for index in xrange(len(densityList)):
#        g.write('{0},{1},{2},{3}\n'.format(index,a[index],b[index],c[index]))
#    g.close()
#
#def plot_phenotype(numSample,network, phenotype, shape, dist, scale, run):
#    a,b,c = zip(*phenotype)
#    plt.plot(range(len(a)), a, color = 'red', label = 'D')
#    plt.plot(range(len(b)), b, color = 'green', label = 'U')
#    plt.plot(range(len(c)), c, color = 'blue', label = 'I')
#    plt.legend()
#    plt.xlabel('time step')
#    plt.ylabel('fraction of regions')
#    plt.xlim(0,len(a))
#    plt.ylim(-0.01,1.01)
#    plt.savefig('phenotype{0}_{1}_{2}_scale{3}_run{4}.png'.format(numSample,shape,dist,scale,str(run).zfill(2)),bbox_inches='tight')    
#    plt.close()        
#    g=open('phenotype{0}_{1}_{2}_scale{3}_run{4}.csv'.format(numSample,shape,dist,scale,str(run).zfill(2)),'w')
#    g.write('step,d,u,i\n')
#    for index in xrange(len(a)):
#        g.write('{0},{1},{2},{3}\n'.format(index,a[index],b[index],c[index]))
#    g.close()    
#
#def plot_phase_portrait(numSample,network,densityList, shape, dist, scale, run):
#    a,b,c = zip(*densityList)
#    plt.plot(a,b, 'ko-', ms = 2)
#    plt.annotate('t={0}'.format(0),(a[0], b[0]), xytext=(0.2, 0.2), textcoords='figure fraction',arrowprops=dict(facecolor='blue', shrink=0.05))
#    plt.annotate('t={0}'.format(len(a)-1),(a[-1], b[-1]), xytext=(0.7, 0.8), textcoords='figure fraction',arrowprops=dict(facecolor='blue', shrink=0.05))
#    plt.grid()
#    plt.xlabel('average D density')
#    plt.ylabel('average U density')
#    plt.xlim(min(a)+0.1*min(a), max(a)+0.1*min(a))
#    plt.ylim(min(b)+0.1*min(a), max(b)+0.1*min(a))
#    plt.axis('equal')
#    plt.savefig('phaseportrait_sample{0}_{1}_{2}_scale{3}_run{4}.png'.format(numSample,shape,dist,scale,str(run).zfill(2)))    
#    plt.close()    
#
#def plot_phenotype_portrait(numSample,network,phenotype, shape, dist, scale, run):
#    a,b,c = zip(*phenotype)
#    plt.plot(a,b, 'ko-', ms = 2)
#    plt.annotate('t={0}'.format(0),(a[0], b[0]), xytext=(0.2, 0.2), textcoords='figure fraction',arrowprops=dict(facecolor='blue', shrink=0.05))
#    plt.annotate('t={0}'.format(len(a)-1),(a[-1], b[-1]), xytext=(0.7, 0.8), textcoords='figure fraction',arrowprops=dict(facecolor='blue', shrink=0.05))
#    plt.grid()
#    plt.xlabel('fraction of regions with phenotype D')
#    plt.ylabel('fraction of regions with phenotype U')
#    plt.xlim(min(a)+0.1*min(a), max(a)+0.1*min(a))
#    plt.ylim(min(b)+0.1*min(a), max(b)+0.1*min(a))
#    plt.axis('equal')
#    plt.savefig('phenotypeportrait_sample{0}_{1}_{2}_scale{3}_run{4}.png'.format(numSample,shape,dist,scale,str(run).zfill(2)))    
#    plt.close()
    
if __name__ == "__main__":

    pass