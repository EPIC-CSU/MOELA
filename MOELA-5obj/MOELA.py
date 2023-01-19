from audioop import reverse
from collections import defaultdict
from heapq import *
from copy import deepcopy
from multiprocessing import connection
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
# from decom import chebyshev, pbi, pbi1
import multiprocessing
import pickle
from sklearn.cluster import KMeans 
import statistics
from Environment import *
import sys
# from g_weight import sort_weights

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

    for i in mating_indices: #(from 0 to 50)
        candidate = archive[i]
        weight = vector[i]
        replace = False
        fit_new = chebyshev(obj, weight) #the most promising optimization direction
        fit_old = chebyshev(candidate, weight) #the most promising optimization direction

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

def update_population(n1, l1, obj, nodes, links, archive, index):

    nodes[index] = deepcopy(n1)
    links[index] = deepcopy(l1)
    archive[index] = deepcopy(obj)

    return nodes, links, archive
####################################### For MOELA-P multi processing ##########################
def local_fit1(obj, weight):
    temp = []
    for i in range(len(obj)):
        temp.append((obj[i]) * max(weight[i], global_min_weight)) #(difference * weight)
    fit = sum(temp)
    return fit 

def simplify_node(node):
    simple_placement = []
    for k in range(0,len(node)):
        if node[k]<8: #CPU
            simple_placement.append(1)
        elif node[k]>7 and node[k]<24: #MC
            simple_placement.append(2)
        elif node[k]>23: #GPU
            simple_placement.append(3)
    
    return simple_placement

def flatten_connection(links):
    simple_connection = links.flatten().tolist()
    return simple_connection

####################################### Local searches ##########################
def local2(node, link, weight, obj, traffic, scaler):
    
    obj_best = deepcopy(obj)
    n_best = deepcopy(node)
    l_best = deepcopy(link)

    old_fit = local_fit1(obj, weight)
    fit_traj = [old_fit]
    placement_traj = [node]
    connection_traj = [link]
    minmax_obj_traj = [obj]

    update = False
    count = 0
    coun = 0
    n = 0
    while count < 15 and n < 15:
        coun = coun + 1
        count = count + 1
        nn, nl = perturb(n_best,l_best,0)
        m, d, la, energy, temperature = calc_params(nn, nl, traffic)
        # temp = [m, d, la]#sirui
        minmax_temp = scaler.transform([(m, d, la, energy, temperature)])[0].tolist()
        new_fit = local_fit1(minmax_temp, weight)

        placement_traj.append(nn)
        connection_traj.append(nl)
        minmax_obj_traj.append(minmax_temp)
        fit_traj.append(new_fit)

        if new_fit < old_fit:
            update = True
            obj_best = deepcopy(minmax_temp)
            old_fit = deepcopy(new_fit)
            n_best = deepcopy(nn)
            l_best = deepcopy(nl)
            count = 0
            n = n + 1
    
    best_fit = local_fit1(obj_best, weight)

    return update, obj_best, n_best, l_best, fit_traj, minmax_obj_traj, placement_traj, connection_traj, coun, best_fit


####################################### Randomly choose start points for early iteration######
def get_tolo(weight_group):
    tolo = []
    for i in range(len(weight_group)):
        tolo += random.choices(weight_group[i], k=2)
    return tolo

def random3(list, neighbor):
    neighbor_list = []
    for i in list:
        neighbor_list += neighbor[i]
        neighbor_list.append(i)
    weight_list = []
    for i in range(50):
        if i not in neighbor_list:
            weight_list.append(i)
    if len(weight_list) < 3:
        random_list = weight_list
    else:
        random_list = random.choices(weight_list, k=3)
    return random_list


####################################### For MOELA-P multi processing ##########################
def get_tolo1(regr, archive, nodes, links, vector, neighbor, weight_group):
    tolo = []
    prediction = []
    for i in range(len(weight_group)):
        current_pareto_front = []
        for index in weight_group[i]:
            single_node_objs = archive[index]
            single_node_weight = vector[index]
            single_node_placement = simplify_node(nodes[index])
            single_node_connection = flatten_connection(links[index])
            current_pareto_front.append(deepcopy(single_node_objs + single_node_weight + single_node_placement + single_node_connection))
        pred = regr.predict(current_pareto_front)
        pre = pred.tolist()
        sorted_pre = sorted(range(len(pre)), key=lambda i: pre[i])
        index = sorted_pre[0]
        prediction.append(pre[index])
        tolo.append(weight_group[i][index])
    return tolo, prediction

def diversity_guarantee(sorted_pre, vector, threshold=0.5):
    top3_tolo = []
    object_1_guarantee = False
    object_2_guarantee = False
    object_3_guarantee = False
    for i in sorted_pre:
        if not object_1_guarantee:
            if vector[i][0] > threshold:
                object_1_guarantee = True
                top3_tolo.append(i)
        if not object_2_guarantee:
            if vector[i][1] > threshold:
                object_2_guarantee = True
                top3_tolo.append(i)
        if not object_3_guarantee:
            if vector[i][2] > threshold:
                object_3_guarantee = True
                top3_tolo.append(i)
    
    # if not object_1_guarantee:
    #     # print("weight diversity guarantee engage for dimension 1")
    #     for i in sorted_pre:
    #         if vector[i][0] > threshold:
    #             tolo.append(i)
    #             break
    # if not object_2_guarantee:
    #     # print("weight diversity guarantee engage for dimension 2")
    #     for i in sorted_pre:
    #         if vector[i][1] > threshold:
    #             tolo.append(i)
    #             break
    # if not object_3_guarantee:
    #     # print("weight diversity guarantee engage for dimension 3")
    #     for i in sorted_pre:
    #         if vector[i][2] > threshold:
    #             tolo.append(i)
    #             break
    return top3_tolo

####################################### For MOELA-P multi processing ##########################
def multi_local(func,inp):
    p = multiprocessing.Pool(processes=(len(inp)))
    data = p.starmap(func, [i for i in inp])
    p.close()
    return data

def main(app, max_time, seed_number, init_random):
    start_time = time.time()
    regr = RandomForestRegressor(100)
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
    
    kmeans = KMeans(n_clusters=6, random_state=0)
    cluster_list = kmeans.fit_predict(weight)
    # print(cluster_list)
    weight_group_0 = list(np.where(cluster_list == 0)[0])
    weight_group_1 = list(np.where(cluster_list == 1)[0])
    weight_group_2 = list(np.where(cluster_list == 2)[0])
    weight_group_3 = list(np.where(cluster_list == 3)[0])
    weight_group_4 = list(np.where(cluster_list == 4)[0])
    weight_group_5 = list(np.where(cluster_list == 5)[0])
    weight_group = [weight_group_0, weight_group_1, weight_group_2, weight_group_3, weight_group_4, weight_group_5]
    popsize = len(weight)
    # popsize = 50
    ori_link = create_mesh()
    ori_nodes = create_nodes()
    nodes = []
    links = []
    archive = []
    phv = []
    elas = []
    train = []
    train_f = []
    record = []
    link_record = []
    node_record = []
    tolo_record = []
    p_record = []
    error_record = []
    search_record = []
    align_record = []
    # std_record = []
    traj_error_record = []
    # nobj = 3#sirui
    # w1 = 0.3 # all could be 0.25
    # w2 = 0.3
    # w3 = 0.4
    regr = RandomForestRegressor(100)
    
    #####randomly generate population
    for i in range(popsize):
        node = deepcopy(ori_nodes)
        if i >9:
            while 1:
                random.shuffle(node)
                if not mc_placement_violation_check(node):
                    break
        nodes.append(node)
        links.append(ori_link)
    traffic,injection=load_traffic(app) #load traffic
    vector = weight
    #####the objectives of mesh
    mesh_mean, mesh_dev, mesh_lat, mesh_energy, mesh_temperature = calc_params(ori_nodes,ori_link,traffic) #depend on num_obj
    mesh = [mesh_mean, mesh_dev, mesh_lat, mesh_energy, mesh_temperature]
    ref = np.array(calc_params(ori_nodes, ori_link, traffic))
    print("initial design:", ref)
    ref = ref
    print("reference:", ref)

    max_objs = deepcopy(mesh)
    max_objs[4] = 10000 #the ori_nodes is one good design because all gpus are close to heat sink, here we assume all gpus as worst case
    min_link = create_all2all()
    min_node = deepcopy(ori_nodes)
    min_objs = calc_params(min_node, min_link, traffic, 'all2all')

    min_objs = (min_objs[0], 0.0, min_objs[2], 0.0, 0.0)
    print("min objectives:", min_objs)
    print("max objectives:", max_objs)
    scaler = MinMaxScaler((0, 100))
    print(scaler.fit([min_objs, max_objs]))
    minmax_mesh = scaler.transform([(mesh_mean, mesh_dev, mesh_lat, mesh_energy, mesh_temperature)])[0].tolist()
    print('scalered mesh:',minmax_mesh)

    a = time.time()
    #####calculate  population
    for i in range(popsize):
        m, d, la, energy, temperature = calc_params(nodes[i], links[i], traffic) #depend on num_obj
        # t = calc_temperature(nodes[i])
        # print('minmax:', scaler.transform([(m,d,la)])[0].tolist())
        # temp = [m, d, la]#sirui
        minmax_temp = scaler.transform([(m, d, la, energy, temperature)])[0].tolist()
        # temp = [m, d]
        # print(temp)
        archive.append(minmax_temp) #archive is the objectice value of poplulation
    
    # phv.append(hvwfg.hv(np.array(archive), ref))
    unscaler_archive = scaler.inverse_transform(archive)
    phv.append(hvwfg.wfg(unscaler_archive, ref))
    link_record.append(deepcopy(links))
    node_record.append(deepcopy(nodes))
    p_record.append(deepcopy(archive))
    prediction_flag = False
    elas.append(time.time() - a)
    print("initialization time:", time.time() - start_time)
    # for h in range(num_iteration):
    h = 0
    while elas[-1] < max_time:
        # start_time = time.time()
        indices = []
        indices.extend(list(range(popsize)))
        # print("indices:", indices)
        random.shuffle(indices)
        #####get local searches start points
        # prediction_time = time.time()
        if h <= init_random: #random 10 start point
            tolo = get_tolo(weight_group)
            error_record.append(0)
            traj_error_record.append(0)
            # std_record.append(0)
        else: 
            tolo, prediction = get_tolo1(regr, archive, nodes, links, vector, neighbor, weight_group)
            prediction_flag = True
        tolo_record.append(tolo)
            # print(h, "iterations weight:", [vector[i] for i in tolo])
        # print(str(h)+" prediction time:", time.time() - prediction_time)
        count = 0
        #####local searches part
        test = []
        traj_input = []
        traj_test_f = []
        # local_search_time = time.time()
        for i in tolo:
            # print(archive)
            update, obj_best, n_best, l_best, fit_traj, minmax_obj_traj, placement_traj, connection_traj, coun, best_fit = local2(nodes[i], links[i],
             vector[i], archive[i], traffic, scaler)
            test.append(best_fit)
            count = count + coun
            if update == True:
                # nodes, links, archive, updated = _update_solution(n_best, l_best, obj_best, indices, nodes, links, archive, vector)       
                nodes, links, archive = update_population(n_best, l_best, obj_best, nodes, links, archive, i)
            for index in range(0, len(minmax_obj_traj)):
                single_node_objs = minmax_obj_traj[index]
                single_node_weight = vector[i]
                single_node_fit = fit_traj[index]
                single_node_placement = simplify_node(placement_traj[index])
                single_node_connection = flatten_connection(connection_traj[index])
                traj_input.append(deepcopy(single_node_objs + single_node_weight + single_node_placement + single_node_connection))
                train.append(deepcopy(single_node_objs + single_node_weight + single_node_placement + single_node_connection))
                train_f.append(deepcopy(best_fit))
                traj_test_f.append(best_fit)
        # print(str(h)+" local search time:", time.time() - local_search_time)
            # if count>max(200, 400*math.pow(0.9, h)):
            #     break
        search_record.append(count)
        # validation_time = time.time()
        if prediction_flag:
            traj_prediction = regr.predict(traj_input)
            # std_traj_test_f = statistics.pstdev(traj_prediction)
            # std_record.append(std_traj_test_f)
            traj_error = np.mean(np.absolute(np.subtract(traj_prediction,traj_test_f)/traj_test_f))*100
            traj_error_record.append(traj_error)
        # print(str(h)+" validationl time:", time.time() - validation_time)
            
        ####Evolutionary algorithm part
        # for i in range(len(weight_group)):
        #     for j in range(len(weight_group)):
        #         mating_indices = weight_group[i] + weight_group[j]
        #         next1 = random.choice(weight_group[i])
        #         next2 = random.choice(weight_group[j])
        #         node1 = nodes[next1]
        #         node2 = nodes[next2]
        #         n1, n2 = PMX(node1, node2)
        #         l1 = perturb(node1, links[next1], 1)[1]
        #         l2 = perturb(node2, links[next2], 1)[1]
        #         n = [n1, n2, n1, n2]
        #         l = [l1, l2, l2, l1]
        #         for z in range(4):
        #             m, d, la = calc_params(n[z], l[z], traffic)
        #             minmax_temp = scaler.transform([(m,d,la)])[0].tolist()
        #             nodes, links, archive, updated = _update_solution(n[z], l[z],  minmax_temp, mating_indices, nodes, links, archive, vector)
        # MOEAD_time = time.time()
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
                minmax_temp = scaler.transform([(m, d, la, energy, temperature)])[0].tolist()
                nodes, links, archive, updated = _update_solution(n[i], l[i],  minmax_temp, mating_indices, nodes, links, archive, vector)

        # print(str(h)+" MOEAD time:", time.time() - MOEAD_time)
        # align_count = 50
        # for i in range(50):
        #     the_obj = archive[i]
        #     the_weight = weight[i]
        #     the_value = local_fit1(the_obj,the_weight)
        #     for one_obj in archive:
        #         if local_fit1(one_obj,the_weight) < the_value:
        #             align_count-=1
        #             break
        # align_record.append(align_count)
        # record.append(deepcopy(archive))
        # print(np.array(archive).shape)
        # print(np.shape(ref))
        #####recoed results
        if (h % 1 == 0):
            unscaler_archive = scaler.inverse_transform(archive)
            phv.append(hvwfg.wfg(unscaler_archive, ref))
            elas.append(time.time() - a)
            link_record.append(deepcopy(links))
            node_record.append(deepcopy(nodes))
            p_record.append(deepcopy(archive))
            # print(h, "iterations phv:", format(phv[-1], '.5f'), "train_error:", format(error_record[-1], '.5f'), 
            # "search:", search_record[-1], "align:", align_record[-1], "test_error:", format(traj_error_record[-1], '.5f'), "time:", format(elas[-1], '.1f'))
        if (h >= init_random and h % 1 == 0):
            regr.fit(train, train_f)
            prediction = regr.predict(train)
            error = np.mean(np.absolute(np.subtract(prediction,train_f)/train_f))*100
            error_record.append(error)
        # print(str(h)+" traning time:", time.time() - training_time)

        # saving_time = time.time()
        # if len(train_f)>10000:
        #     train_f = train_f[-10000:]
        #     train = train[-10000:]
        save_data_to_files(method, seed_number, app, h ,scaler.inverse_transform(archive).tolist(), phv[-1], elas[-1], link_record[-1], node_record[-1], init_random)
        if (h % 10 == 0):
            save_design_to_files(method, seed_number, app, h ,scaler.inverse_transform(archive).tolist(), phv[-1], elas[-1], link_record[-1], node_record[-1], init_random)
        # if len(train_f)>10000:
        #     train_f = train_f[-10000:]
        #     train = train[-10000:]

        with open(method+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_errors_'+app+'5.csv', 'a+') as f:
            f.writelines('\n' + str(h) + ',' + str(error_record[-1]) + ',' + str(traj_error_record[-1]))
        h+=1
    save_design_to_files(method, seed_number, app, h-1 ,scaler.inverse_transform(archive).tolist(), phv[-1], elas[-1], link_record[-1], node_record[-1], init_random)
    return 

def create_all2all(): ##create SWNoC not mesh##
    links = np.zeros(shape=(64, 64))
    for i in range(64):
        for j in range(64):
            links[i][j] = 1

    return deepcopy(links)
#####must use this when running parallelism
if __name__ == '__main__':
    method = 'sirui_MOELA_inf'
    # app = 'bfs'
    init_random = 10
    seed_number = int(sys.argv[-1])
    app = sys.argv[-2]
    # num_iteration = 1000
    # max_time = 3000
    max_time = 180000
    # max_time = 30000
    random.seed(seed_number)
    initilize_files(method, seed_number, app, init_random)
    with open(method+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_errors_'+app+'5.csv', 'w') as f:
        f.write('\niteration,training_error,testing_error')

    main(app, max_time, seed_number, init_random)
