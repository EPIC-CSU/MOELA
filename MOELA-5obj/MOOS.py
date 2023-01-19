import copy
from distutils import archive_util
import random
import time
import math
from Environment import *
import numpy as np
import datetime
import sys
def lamda_initilize(objs):
    num_lamda = len(objs[0])
    parent_node =[]
    for i in range(num_lamda):
        parent_node.append([0, 1])
    return parent_node


def chebychev_select(local_pareto_node, lamda_node, reference_point_set, mode_num):

    s_objs_value_set = []
    for i in range(len(lamda_node)):
        center_lamda_value = (lamda_node[i][0] + lamda_node[i][1]) / 2
        scalarized_value = center_lamda_value * (local_pareto_node[i] - reference_point_set[i])
        s_objs_value_set.append(scalarized_value)
    the_chosen_obj_value = max(s_objs_value_set)  # what if two maximium exists? --does not matter
    the_chosen_obj_index = s_objs_value_set.index(max(s_objs_value_set)) # what if two maximium exists?
    if mode_num == 0:
        return the_chosen_obj_value
    if mode_num == 1:
        return the_chosen_obj_index


def starting_point_select(local_pareto, lamda_node, reference_point_set):
    scalarized_obj_value_set = []
    for node in local_pareto:
        scalarized_obj_value_set.append(chebychev_select(node, lamda_node, reference_point_set, 0))
    starting_point_index = scalarized_obj_value_set.index(min(scalarized_obj_value_set))  # what if two minimium exists?
    node = local_pareto[starting_point_index]
    scalarized_obj_index = chebychev_select(node, lamda_node, reference_point_set, 1)
    return starting_point_index, scalarized_obj_index


def child_node_generate(parent_node, scalarized_obj_index):
    left_node = copy.deepcopy(parent_node)
    center_node = copy.deepcopy(parent_node)
    right_node = copy.deepcopy(parent_node)
    min, max = parent_node[scalarized_obj_index]
    one_third = min + (max - min) / 3
    two_third = min + 2 * (max - min) / 3
    left_node[scalarized_obj_index] = [min, one_third]
    center_node[scalarized_obj_index] = [one_third, two_third]
    right_node[scalarized_obj_index] = [two_third, max]
    return [left_node, center_node, right_node]

def update_pareto(global_p, local_pareto_set, global_n, local_node_set, global_l, local_link_set):
    for i in range(len(local_pareto_set)):
        itera = len(global_p)
        while(itera != 0):
            if dominate(local_pareto_set[i], global_p[itera-1]):
                del global_p[itera-1]
                del global_n[itera-1]
                del global_l[itera-1]
            itera = itera -1
        global_p.append(local_pareto_set[i])
        global_n.append(local_node_set[i])
        global_l.append(local_link_set[i])
    return global_p, global_n, global_l

def main(app, max_time, seed_number, init_random):
    
    # initial pareto set

    timeo = time.time()
    phv_record = []
    time_record = []
    link_record = []
    node_record = []
    tree_record = []
    nodes = create_nodes()  # create mesh
    # print(nodes)
    # print(nodes)
    links = create_mesh()  # create mesh
    traffic, inj = load_traffic(app)

    ref = np.array(calc_params(nodes, links, traffic))
    print("initial design:", ref)
    ref = ref
    print("reference:", ref)
    print("maximum time (h):", max_time/3600)

    mesh_mean, mesh_dev, mesh_lat, mesh_energy, mesh_temperature = calc_params(nodes, links, traffic)
    # print(mesh_mean, mesh_dev, mesh_lat)
    obj = [mesh_mean, mesh_dev, mesh_lat, mesh_energy, mesh_temperature]#sirui
    # obj = [mesh_mean, mesh_dev]
    local_pareto_set, local_node_set, local_link_set, best_phv = local_search(nodes, links, obj, traffic, ref)
    time_record.append(time.time() - timeo)
    phv_record.append(best_phv)
    node_record.append(local_node_set)
    link_record.append(local_link_set)
    # initial tree model
    reference_point_set = []
    global_p = []   #global pareto
    global_n = []   #global node
    global_l = []   #global link
    global_p = global_p + local_pareto_set
    global_n = global_n + local_node_set
    global_l = global_l + local_link_set
    parent_lamda_node = lamda_initilize(local_pareto_set)
    # print(parent_lamda_node)
    center_lamda_value = [(lamda_node[0] + lamda_node[1]) / 2 for lamda_node in parent_lamda_node]
    # print(center_lamda_value)
    tree_record.append(center_lamda_value)
    three_node = child_node_generate(parent_lamda_node, 0)  # 0 can be replaced by a random number in index range
    for i in range(len(local_pareto_set[0])):
        reference_point_set.append(0)
        
    max_HPV = 0
    # for h in range(num_iteration):
    h = 0
    while time_record[-1] < max_time:
        
        max_node_index = 0
        split_obj_index = 0
        temp_global = copy.deepcopy(global_p)
        flag = False
        for i in range(len(three_node)):
            starting_point_index, scalarized_obj_index = starting_point_select(global_p, three_node[i],
                                                                               reference_point_set)
            start_node = global_n[starting_point_index]
            start_link = global_l[starting_point_index]
            mesh_mean, mesh_dev, mesh_lat, mesh_energy, mesh_temperature = calc_params(start_node, start_link, traffic)
            obj = [mesh_mean, mesh_dev, mesh_lat, mesh_energy, mesh_temperature]#sirui
            # obj = [mesh_mean, mesh_dev]
            local_pareto_set, local_node_set, local_link_set, best_phv = local_search(start_node, start_link, obj, traffic,
                                                                                      ref)
            global_p, global_n, global_l = update_pareto(global_p, local_pareto_set, global_n, local_node_set, global_l, 
                                                         local_link_set) #update global pareto
            temp = temp_global + local_pareto_set
            # print(temp,ref)
            phv = phv_calculator(temp, ref, ref)    #phv for glocbal pareto
            if phv > max_HPV:
                max_HPV = phv
                max_node_index = i
                split_obj_index = scalarized_obj_index
                flag = True
        max_HPV = phv_calculator(global_p, ref, ref)
        phv_record.append(max_HPV)
        
        node_record.append(global_n)
        link_record.append(global_l)
        if flag == True: #if update sucessfully
            parent_lamda_node = three_node[max_node_index]
            three_node = child_node_generate(parent_lamda_node, split_obj_index)
        else:
            n = random.randrange(len(ref))
            while n == split_obj_index:
                n = random.randrange(len(ref))
            split_obj_index = n
            three_node = child_node_generate(parent_lamda_node, split_obj_index)
        center_lamda_value = [(lamda_node[0] + lamda_node[1]) / 2 for lamda_node in parent_lamda_node]
        tree_record.append(center_lamda_value)
        # print("starting tree model iteration " + str(h), max_HPV)
        time_record.append(time.time() - timeo)
        # print(h, "iterations phv:", format(max_HPV, '.5f'), "time:", format(time_record[-1], '.5f'))
        save_data_to_files(method, seed_number, app, h, global_p, phv_record[-1], time_record[-1], link_record[-1], node_record[-1], init_random)
        if (h % 50 == 0):
            save_design_to_files(method, seed_number, app, h, global_p, phv_record[-1], time_record[-1], link_record[-1], node_record[-1], init_random)
        h+=1
    save_design_to_files(method, seed_number, app, h-1 ,global_p, phv_record[-1], time_record[-1], link_record[-1], node_record[-1], init_random)
    return 

    
if __name__ == '__main__':
    method = 'sirui_MOOS'
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
    main(app, max_time, seed_number, init_random)
        
