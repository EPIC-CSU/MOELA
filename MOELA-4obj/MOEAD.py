from collections import defaultdict
from heapq import *
from copy import deepcopy
import random
import math
import numpy as np
import datetime
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import timeit
import csv
import hvwfg
import time
from Environment import *
import sys

NUM_IPS = 64
HOP_TIME = 1
ROUTER_ENERGY_PER_PORT = 1e-3 * 2.25 * 4.57
LINK_ENERGY = 1e-3 * 7.44 * 4.57
Z_LINK_ENERGY = 1e-3 * 0.22 * 128  ##TSV energy per bit 0.2pJ/bit
global_min_weight = 0.0001

####################################### chebyshev calculation ##########################
def chebyshev(solution, weights, min_weight=global_min_weight):
    a = max([max(weights[i], min_weight) * (solution[i]) for i in range(len(solution))])
    return a

def random_mesh(): ##create SWNoC not mesh##
    links = numpy.zeros(shape=(64, 64))
    record = [0] * 144
    for i in range(64):
        b = random.randint(0, 63)
        while links[i][b] == 1 or i == b:
            b = random.randint(0, 63)
        links[i][b] = 1
        links[b][i] = 1
        record[i] = record[i] + 1
        record[b] = record[b] + 1
    for i in range(80):
        a = random.randint(0, 63)
        b = random.randint(0, 63)
        while a == b or links[a][b] == 1:
            b = random.randint(0, 63)
        links[a][b] = 1
        links[b][a] = 1
        record[a] = record[a] + 1
        record[b] = record[b] + 1
    return deepcopy(links)


####################################### parents from neighbourhood or population ##########################
def get_mating_indices(index, neighborhoods, popsize):

    if random.uniform(0.0, 1.0) <= 0.9:
        return neighborhoods[index]
    else:
        return list(range(popsize))

####################################### PMX crossover ##########################
def PMX(pa1, pa2):
    p1 = deepcopy(pa1)
    p2 = deepcopy(pa2)
        
    n = len(p1)
    o1 = [None]*n
    o2 = [None]*n
                
    # select cutting points
    cp1 = random.randrange(n)
    cp2 = random.randrange(n)
                
    if n > 1:
        while cp1 == cp2:
            cp2 = random.randrange(n)
                
    if cp1 > cp2:
        cp1, cp2 = cp2, cp1
                    
    # exchange between the cutting points, setting up replacement arrays
    replacement1 = {}
    replacement2 = {}
            
    for i in range(cp1, cp2+1):
        o1[i] = p2[i]
        o2[i] = p1[i]
        replacement1[p2[i]] = p1[i]
        replacement2[p1[i]] = p2[i]
                
        # fill in remaining slots with replacements
    for i in range(n):
        if i < cp1 or i > cp2:
            n1 = p1[i]
            n2 = p2[i]
                        
            while n1 in replacement1:
                n1 = replacement1[n1]
                            
            while n2 in replacement2:
                n2 = replacement2[n2]
                            
            o1[i] = n1
            o2[i] = n2
                        
    result1 = o1
    result2 = o2   
                
    return [result1, result2]

####################################### Update the population ##########################
def _update_solution(n1, l1, obj, p, nodes, links, archive, vector):
    c = 0
    mating_indices = deepcopy(p)
    random.shuffle(mating_indices)
    updated = False

    for i in mating_indices:
        candidate = archive[i]
        weight = vector[i]
        replace = False
        fit_new = chebyshev(obj, weight)
        fit_old = chebyshev(candidate, weight)
        if fit_new < fit_old:
            replace = True

        if replace:
            nodes[i] = deepcopy(n1)
            links[i] = deepcopy(l1)
            archive[i] = deepcopy(obj)
            c = c + 1
            updated = True

        if c >= 1:
            break
    return nodes, links, archive, updated


def main(app, max_time, seed_number, init_random):
    start_time = time.time()
    with open('weights/sirui_weight5.csv', newline='') as f:
        reader = csv.reader(f)
        weight = list(reader)
    for i in range(len(weight)):
        weight[i] = [float(j) for j in weight[i]]
    with open('weights/sirui_weight5_n.csv', newline='') as f:
        reader = csv.reader(f)
        neighbor = list(reader)
    for i in range(len(neighbor)):
        neighbor[i] = [int(j) for j in neighbor[i]]
    popsize = len(weight)
    ori_link = create_mesh()
    ori_nodes = create_nodes()
    nodes = []
    links = []
    archive = []
    phv = []
    elas = []
    trained = []
    node_record = []
    link_record = []
    num_local = []
    nobj = 3 #sirui
    # ref = np.array([1.0, 1.0, 10.0])
    # ref = np.array([0.022, 0.19, 12.2]) #sirui
    for i in range(popsize):
        node = deepcopy(ori_nodes)
        if i >9:
            while 1:
                random.shuffle(node)
                if not mc_placement_violation_check(node):
                    break
        nodes.append(node)
        links.append(ori_link)
    traffic,injection=load_traffic(app) #load benchmark
    
    
    mesh_mean, mesh_dev, mesh_lat, mesh_energy, mesh_temperature = calc_params(ori_nodes,ori_link,traffic) #depend on num_obj
    mesh = [mesh_mean, mesh_dev, mesh_lat, mesh_energy]
    ref = np.array(calc_params(ori_nodes, ori_link, traffic))[0:4]
    print("initial design:", ref)
    ref = ref
    print("reference:", ref)

    max_objs = deepcopy(mesh)
    # max_objs[4] = 10000
    min_link = create_all2all()
    min_node = deepcopy(ori_nodes)
    min_objs = calc_params(min_node, min_link, traffic, 'all2all')

    min_objs = (min_objs[0], 0.0, min_objs[2], 0.0)
    print("min objectives:", min_objs)
    print("max objectives:", max_objs)
    scaler = MinMaxScaler((0, 100))
    print(scaler.fit([min_objs, max_objs]))
    minmax_mesh = scaler.transform([(mesh_mean, mesh_dev, mesh_lat, mesh_energy)])[0].tolist()
    print('scalered mesh:',minmax_mesh)

    a = time.time()
    #####randomly generate population
    for i in range(popsize):
        m, d, la, energy, temperature = calc_params(nodes[i], links[i], traffic)
        # t = calc_temperature(nodes[i])
        # temp = [m, d, la]
        minmax_temp = scaler.transform([(m, d, la, energy)])[0].tolist()
        # temp = [m, d] #sirui
        archive.append(minmax_temp)
    vector = weight
    # phv.append(hvwfg.hv(np.array(archive), ref))
    unscaler_archive = scaler.inverse_transform(archive)
    # print(unscaler_archive)
    phv.append(hvwfg.wfg(unscaler_archive, ref))
    node_record.append(nodes)
    link_record.append(links)
    elas.append(time.time() - a)
    print("initialization time:", time.time() - start_time)
    # for h in range(num_iteration):
    h = 0
    while elas[-1] < max_time:
        # start_time = time.time()
        indices = []
        indices.extend(list(range(popsize)))
        random.shuffle(indices)
        count = 0
        for k in indices:
            #####get parents from neighbourhood or population
            mating_indices = get_mating_indices(k, neighbor, popsize)
            mating = deepcopy(mating_indices)
            mating.remove(k)
            #####choose parents
            next1 = random.choice(mating)
            next2 = random.choice(mating)
            while (next1 == next2):
                next2 = random.choice(mating)
            node1 = nodes[next1]
            node2 = nodes[next2]
            #####crossover and mutation
            n1, n2 = PMX(node1, node2)
            l1 = perturb(node1, links[next1], 1)[1]
            l2 = perturb(node2, links[next2], 1)[1]
            n = [n1, n2, n1, n2]
            l = [l1, l2, l2, l1]
            count = 0
            #####objectives calculation
            for i in range(4):
                m, d, la, energy, temperature = calc_params(n[i], l[i], traffic)
                minmax_temp = scaler.transform([(m, d, la, energy)])[0].tolist()#could be wrong here
                nodes, links, archive, updated = _update_solution(n[i], l[i],  minmax_temp, mating_indices, nodes, links, archive, vector)
                # if updated == True:
                #     count = count + 1
                # if count == 2:
                #     break
        # print("MOEAD Time:", time.time() - start_time)
        #####recoed results
        if (h % 1 == 0):
            unscaler_archive = scaler.inverse_transform(archive)
            phv.append(hvwfg.wfg(unscaler_archive, ref))
            elas.append(time.time() - a)
            link_record.append(deepcopy(links))
            node_record.append(deepcopy(nodes))
        save_data_to_files(method, seed_number, app, h ,scaler.inverse_transform(archive).tolist(), phv[-1], elas[-1], link_record[-1], node_record[-1], init_random)
        if (h % 10 == 0):
            save_design_to_files(method, seed_number, app, h ,scaler.inverse_transform(archive).tolist(), phv[-1], elas[-1], link_record[-1], node_record[-1], init_random)
        h+=1
    save_design_to_files(method, seed_number, app, h-1 ,scaler.inverse_transform(archive).tolist(), phv[-1], elas[-1], link_record[-1], node_record[-1], init_random)
    return 

if __name__ == '__main__':
    method = 'sirui_MOEAD'
    # app = 'nw'
    init_random = 10
    seed_number = int(sys.argv[-1])
    app = sys.argv[-2]
    # num_iteration = 1000
    # max_time = 300
    max_time = 180000
    # max_time = 30000
    random.seed(seed_number)
    initilize_files(method, seed_number, app, init_random)
    main(app, max_time, seed_number, init_random)
