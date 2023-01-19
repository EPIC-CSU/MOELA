#################################### Import ###############################
from collections import defaultdict
from heapq import *
from copy import deepcopy
# from msilib.schema import Directory
from random import randint,random,seed
import math
import numpy
import random
from sklearn.ensemble import RandomForestRegressor
import timeit
import copy
import numpy as np
import datetime
import os
from csv import writer

################################ defn. Variables #############################

routerStages = 3
x_size = 8
y_size = 4
z_size = 2
num_links = 144
NUM_CORES = 128
NUM_IPS = 64
KMIN = 1
KMAXINITIAL = 7
KMAX = 8
LINK_ENERGY = 1e-3 * 7.44 * 4.57
ROUTER_ENERGY_PER_PORT = 1e-3 * 2.25 * 4.57
Z_LINK_ENERGY = 1e-3 * 0.22 * 128  ##TSV energy per bit 0.2pJ/bit
HOP_TIME = 1
LINK_ENERGY = 1e-3 * 7.44 * 4.57

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

def create_all2all(): ##create SWNoC not mesh##
    links = np.zeros(shape=(64, 64))
    for i in range(64):
        for j in range(64):
            links[i][j] = 1

    return deepcopy(links)
#################### Creating 64 nodes ##########################
def create_nodes():
    nodes = []
    # nodes_orig=[]
    mc = 8
    cpu = 0
    gpu = 24
    for i in range(0, 64):
        nodes.append(9999)
    for i in range(0, 4):
        nodes[i * 16 + 5] = cpu;
        nodes[i * 16 + 6] = cpu + 1;
        cpu = cpu + 2
        nodes[i * 16 + 0] = mc;
        nodes[i * 16 + 3] = mc + 1;
        nodes[i * 16 + 12] = mc + 2;
        nodes[i * 16 + 15] = mc + 3;
        mc = mc + 4
    for i in range(0, 64):
        if nodes[i] == 9999:
            nodes[i] = gpu;
            gpu = gpu + 1
            # nodes_orig.append(i)
    return deepcopy(nodes)

########################## Creating 4*4*4 mesh link connectivity #############################
def create_mesh(): ##create SWNoC not mesh##
    links = numpy.zeros(shape=(64, 64))
    for i in range(0, 64):
        for j in range(0, 64):
            zs = int(i / 16)
            ts = i % 16
            ys = int(ts / 4)
            xs = ts % 4
            zd = int(j / 16)
            td = j % 16
            yd = int(td / 4)
            xd = td % 4
            if (abs(zs - zd) == 1) and (xs == xd) and (ys == yd):  # z-links
                links[i][j] = 1
            elif (zs == zd) and (ys == yd) and (abs(xd - xs) == 1):  # x-links
                links[i][j] = 1
            elif (zs == zd) and (xs == xd) and (abs(yd - ys) == 1):  # y-links
                links[i][j] = 1

    return deepcopy(links)

################################### Load traffic ####################################


def load_traffic(app):
    with open('traffic/'+app+"_traffic.csv") as f:
        content = f.readlines()
    # also remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    traffic = []
    injection=[]
    for i in range(0, len(content)):
        temp = content[i].split()
        t = []
        injection1=0.0
        for j in range(0, len(temp)):
            t.append(float(temp[j])/200000000.0) # control traffic number unit, sirui
            injection1 = injection1+float(temp[j])
        traffic.append(t)
        injection.append(injection1)
        
    return traffic, injection

################################# Dijkstra shortest path algorithm ###############################


def dijkstra(edges, f, t):
    g = defaultdict(list)
    for l,r,c in edges:
        g[l].append((c,r))

    q, seen = [(0,f,())], set()
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 not in seen:
                    heappush(q, (cost+c, v2, path))

    return float("inf")


def make_edges(nodes,links,model = '3d'):
    edges=[]
    temp=[]
    if model == '3d':
        # print('3d model')
        for i in range(0, 64):
            for j in range(0, 64):
                if links[i][j] != 0:
                    temp.append(nodes[i])
                    temp.append(nodes[j])
                    zs=int(i/16)
                    ts=i%16
                    ys=int(ts/4)
                    xs=ts%4

                    zd = int(j / 16)
                    td = j % 16
                    yd = int(td / 4)
                    xd = td % 4
                    if xs==xd and ys==yd and abs(zs-zd)==1:
                        # temp.append(4)
                        temp.append(6)

                    elif zs==zd and (xs!=xd or ys!=yd):
                        # temp.append(3 + math.ceil((((xd-xs)**2)+((ys-yd)**2))**0.5))
                        temp.append(5 + math.ceil((((xd-xs)**2)+((ys-yd)**2))**0.5))
                    else:
                        print("link perturbation went wrong: ", i, " , ", j)
                    edges.append(temp)
                    temp = []
    if model == 'all2all':
        print('all2all model')
        for i in range(0, 64):
            for j in range(0, 64):
                if links[i][j] == 1:
                    temp.append(nodes[i])
                    temp.append(nodes[j])
                    temp.append(6)
                    edges.append(temp)
                    temp = []
    if model == '2d':
        print('2d model')
        for i in range(0, 64):
            for j in range(0, 64):
                if links[i][j] == 1:
                    temp.append(nodes[i])
                    temp.append(nodes[j])
                    xs=int(i/8)
                    ys=i%8
                    xd = int(j/8)
                    yd = j%8
                    if xs!=xd or ys!=yd:
                        temp.append(5+ math.ceil((((xd-xs)**2)+((ys-yd)**2))**0.5))
                    else:
                        print("2d mesh wrong: ", i, " , ", j)
                    edges.append(temp)
                    temp = []
    return edges


####################################### params calculation ##########################

def ports(links):
    ports = [0] * NUM_IPS
    for i in range(0, NUM_IPS):
        for j in range(0, NUM_IPS):
            if links[i][j] != 0:
                ports[i] += 1
    return ports


def calc_params(nodes, links, traffic, model='3d'):
    m = 0
    d = 0
    hop = 0
    path = 0
    lat = 0
    link_util = []
    r_ports = ports(links)
    max_h = 0
    energy = 0
    cpu_traffic = 0
    total_traffic = 0
    edges = make_edges(nodes, links, model)
    for i in range(0, 64):
        t = []
        for j in range(0, 64):
            t.append(0)
        link_util.append(t)
    # dev
    for i in range(0, 64):
        for j in range(0, 64):
            if nodes[i] == nodes[j]:
                continue
            # print(i, "  ", j)
            p = str(dijkstra(edges, nodes[i], nodes[j]))
            p_break = p.split(',')
            
            h_k = len(p_break) - 3
            if h_k>max_h:
                max_h = h_k
            hop += h_k
            path += 1

            if nodes[i] < 24 and nodes[j] < 24:
                cpu_traffic += traffic[nodes[i]][nodes[j]]
                lat += float(p_break[0].replace('(', '')) * traffic[nodes[i]][nodes[j]] * HOP_TIME

            total_traffic += traffic[nodes[i]][nodes[j]]

            #calculate energy   
            p2_break = p.split(',')
            intra_rp = 0
            inter_rp = 0
            for k in range(1, len(p2_break) - 2):
                node1 = int(p2_break[k][2:])
                node2 = int(p2_break[k + 1][2:])
                ind1 = nodes.index(node1)
                ind2 = nodes.index(node2)
                if links[ind1][ind2] < 1:
                    print('something is wrong..!!')
                intra_rp = r_ports[ind1] * ROUTER_ENERGY_PER_PORT
                if k == len(p2_break) - 2:
                    intra_rp = r_ports[ind2] * ROUTER_ENERGY_PER_PORT

                zs = int(ind1 / 16)
                ts = ind1 % 16
                ys = int(ts / 4)
                xs = ts % 4

                zd = int(ind2 / 16)
                td = ind2 % 16
                yd = int(td / 4)
                xd = td % 4
                if model == '3d':
                    if xs == xd and ys == yd and abs(zs - zd) == 1:
                        inter_rp = inter_rp + 1 * Z_LINK_ENERGY
                    elif (xs != xd or ys != yd):
                        dist = math.ceil((((xd - xs) ** 2) + ((ys - yd) ** 2) + ((zs - zd) ** 2)) ** 0.5)
                        inter_rp = inter_rp + dist * LINK_ENERGY
                    else:
                        print("link perturbation went wrong: ", ind1, " , ", ind2)
                if model == 'all2all':
                    if links[i][j] == 1:
                        inter_rp = inter_rp + 1 * LINK_ENERGY
            energy += ( intra_rp + inter_rp )* traffic[nodes[i]][nodes[j]]
            for k in range(1, len(p_break) - 2):
                node1 = int(p_break[k][2:])
                node2 = int(p_break[k + 1][2:])
                ind1 = nodes.index(node1)
                ind2 = nodes.index(node2)
                if links[ind1][ind2] != 1: #or links[ind1][ind2] != 3:
                    print('something is wrong..!!')
                link_util[ind1][ind2] = link_util[ind1][ind2] + traffic[nodes[i]][nodes[j]]

    for i in range(0, 64):
        for j in range(0, 64):
            m = m + link_util[i][j]
    # print("CPU-MC Traffic Portion:", cpu_traffic/total_traffic*100, "%")

    if model == '3d':
        m = m/144
    if model == '2d':
        m = m /112
    if model == 'all2all':
        # m=m/2016
        m=m/144
    for i in range(0, 64):
        for j in range(i, 64):
            if (links[i][j] != 1):
                continue
            d = d + (link_util[i][j] + link_util[j][i] - m)**2
    d = d**0.5
    temp = calc_temperature(nodes)
    return m, d, lat, energy, temp

def calc_temperature(nodes):
    layer0=[]
    layer1=[]
    layer2=[]
    layer3=[]
    r1=10
    r2=10
    r3=10
    r4=10
    for i in range(0,64):
        p=-1
        ### determine type of core
        if int(nodes[i]) == 0: #cpu0
            p=1.5
        elif int(nodes[i])<8: #the rest cpu
            p=1
        elif int(nodes[i])<24: #mc
            p=1.5
        else:
            p=6
        ### add power numbers
        if i<16:
            layer0.append(p)
        elif i<32:
            layer1.append(p)
        elif i<48:
            layer2.append(p)
        else:
            layer3.append(p)
    ### calculate temperature
    m=0
    for i in range(0,16):
        # temp=layer3[i]*(r1+r2+r3+r4)+layer2[i]*(r1+r2+r3)+layer1[i]*(r1+r2)+layer0[i]*r1 #bottom heat sink
        temp=layer0[i]*(r1+r2+r3+r4)+layer1[i]*(r1+r2+r3)+layer2[i]*(r1+r2)+layer3[i]*r1 #top heat sink
        m=m+int(temp)
    #print(m)
    return m

################################ PHV calculator #####################################
def dominate(p, q):
    k=0
    d = True
    while d and k < len(p):
        d = not (q[k] < p[k])
        k += 1
    return d


def slice(pl, k, ref):
    p = pl[0]
    pl = pl[1:]
    ql = []
    s = []
    while pl:
        ql = insert(p, k + 1, ql)
        p_prime = pl[0]
        s.append(((p_prime[k] - p[k]), ql))
        p = p_prime
        pl = pl[1:]
    ql = insert(p, k + 1, ql)
    s.append(((ref[k] - p[k]), ql))
    return s

def insert(p, k, pl):
    ql = []
    while pl and pl[0][k] < p[k]:
        ql.append(pl[0])
        pl = pl[1:]
    ql.append(p)
    while pl:
        if not dominates(p, pl[0], k):
            ql.append(pl[0])
        pl = pl[1:]
    return ql

def dominates(p, q, k=None):
    if k is None:
       k = len(p)
    d = True
    while d and k < len(p):
        d = not (q[k] < p[k])
        k += 1
    return d

def phv_calculator(archive, new_pt, ref):
    # print(archive, '\n',new_pt, '\n', ref)
    ps = deepcopy(archive)
    ps.append(deepcopy(new_pt))
#    ref = [5,5]
    n = min([len(p) for p in ps])
    pl = ps[:]
    pl.sort(key=lambda x: x[0], reverse=False)
    s = [(1, pl)]
    for k in range(n - 1):
        s_prime = []
        for x, ql in s:
            # print(ql, k, ref)
            for x_prime, ql_prime in slice(ql, k, ref):
                s_prime.append((x * x_prime, ql_prime))
        s = s_prime
    vol = 0
    for x, ql in s:
        vol = vol + x * (ref[n - 1] - ql[0][n - 1])
    return vol

############################### Make Perturbation #################################

def mc_placement_violation_check(node):
    mc_placement_violation_flag = False
    for i in [42,41,38,37,26,25,22,21]:
        if node[i]>7 and node[i]<24:
            mc_placement_violation_flag = True
            break
    return mc_placement_violation_flag
    
def swap(nodes):
    # nodes = deepcopy(node)
    L2_group = list(range(8,24))
    other_group = list(range(8))+list(range(24,64))
    surface_location_list = list(range(64))
    centra_area_list = [42,41,38,37,26,25,22,21]
    for element in centra_area_list:
        surface_location_list.remove(element)
    # rs = random.randint(0, 63)
    if random.random() < 0.75*0.9713:
        first_node = random.choice(other_group)
    else:
        first_node = random.choice(L2_group)
    rs = nodes.index(first_node)
    if nodes[rs] in L2_group:
        while 1:
            rd = random.choice(surface_location_list)
            if rd != rs:
                break
    else:
        if rs in surface_location_list:
            while 1:
                rd = random.choice(list(range(64)))
                if rd != rs:
                    break
        else:
            while 1:
                the_second_node = random.choice(other_group)
                if the_second_node != nodes[rs]:
                    rd = nodes.index(the_second_node)
                    break
    t1 = nodes[rs]
    nodes[rs] = nodes[rd]
    nodes[rd] = t1  # exchanged nodes
    return

def perturb(node, link, case):
    nodes = deepcopy(node)
    links = deepcopy(link)
    r1 = random.random()
    if case==0:
        threshold=0.5
    else:
        threshold=0.0
    if (r1 < threshold):  # exchange cores 0.6
        swap(nodes)
    else:  # change links
        while 1:
            rl1 = random.randint(0, 3)
            rs1 = random.randint(0, 15)
            rd1 = random.randint(0, 15)
            rl2 = random.randint(0, 3)
            rs2 = random.randint(0, 15)
            rd2 = random.randint(0, 15)
            if (rs1 == rd1) or (rs2 == rd2):
                continue
            l1 = links[rl1 * 16 + rs1][rl1 * 16 + rd1]  # remove
            l2 = links[rl2 * 16 + rs2][rl2 * 16 + rd2]  # add
            if (l1 == 0) or (l2 == 1):  # link absent/present
                continue
            # if links[rl2 * 16 + rs2].sum() == 7:
            if links[rl2 * 16 + rs2].sum()>= 8 or links[rl2 * 16 + rd2].sum() >= 8:
                continue
            # move links
            links[rl1 * 16 + rs1][rl1 * 16 + rd1] = 0
            links[rl1 * 16 + rd1][rl1 * 16 + rs1] = 0
            links[rl2 * 16 + rs2][rl2 * 16 + rd2] = 1
            links[rl2 * 16 + rd2][rl2 * 16 + rs2] = 1
            # check for islands
            edges_trial=[]
            temp=[]
            for i in range(0, 64):
                for j in range(0, 64):
                    temp.append(nodes[i])
                    temp.append(nodes[j])
                    if links[i][j] == 1:
                        temp.append(1)
                        # edges1.append(temp)
                    else:
                        temp.append(999)
                    edges_trial.append(temp)
                    # edges1.append(temp)
                    temp = []
            island=0
            for i in range(0, 64):
                p = str(dijkstra(edges_trial, nodes[rl1 * 16 + rs1], nodes[i]))
                q = str(dijkstra(edges_trial, nodes[rl1 * 16 + rd1], nodes[i]))
                p=p.split(',')
                q=q.split(',')
                cost_trial1=int(p[0][1:])
                cost_trial2=int(q[0][1:])
                if (cost_trial1>100 or cost_trial2>100):
                    island=1
                    break
            if (island==1): #reverse everything and restart
                links[rl1 * 16 + rs1][rl1 * 16 + rd1] = 1
                links[rl1 * 16 + rd1][rl1 * 16 + rs1] = 1
                links[rl2 * 16 + rs2][rl2 * 16 + rd2] = 0
                links[rl2 * 16 + rd2][rl2 * 16 + rs2] = 0
                continue
            break
    return nodes, links

################################# local search ##########################
def local_search(node, link, obj, traffic, ref):
    local_pareto = []
    local_pareto.append(obj)
    best_phv = phv_calculator(local_pareto, obj, ref)
    best_obj = deepcopy(obj)
    n_best = deepcopy(node)
    l_best = deepcopy(link)
    local_node = []
    local_link = []
    local_node.append(n_best)
    local_link.append(l_best)
    n=0
    while(n<15):
        n = n + 1
        nodes = []
        links = []
        objs = []
        phv = []
        for i in range(5):
            new_node, new_link = perturb(n_best, l_best, 0)
            mean, dev, lat, energy, temp = calc_params(new_node, new_link, traffic)
            new_pareto = [mean, dev, lat, energy]#sirui
            # new_pareto = [mean, dev]
            phv.append(phv_calculator(local_pareto, new_pareto, ref))
            nodes.append(new_node)
            links.append(new_link)
            objs.append(new_pareto)
        if max(phv) > best_phv:
            ind = phv.index(max(phv))
            best_phv = phv[ind]
            best_obj = objs[ind]
            n_best = nodes[ind]
            l_best = links[ind]
            itera = len(local_pareto)
            while(itera != 0):
                # print(best_obj, local_pareto[itera-1])
                if dominate(best_obj, local_pareto[itera-1]):
                    del local_pareto[itera-1]
                    del local_node[itera-1]
                    del local_link[itera-1]
                itera = itera -1
            local_pareto.append(best_obj)
            local_node.append(n_best)
            local_link.append(l_best)
        else:
            break
    return local_pareto, local_node, local_link, best_phv




def save_data_to_files(method, seed_number, app, h ,global_p, phv_record, time_record, link_record, node_record, init_random):

    with open(method+'/seed'+str(seed_number)+'_'+'init'+str(init_random)+'_'+method+'_iter_time_phv_'+app+'4.csv', 'a+') as f:
        f.writelines('\n' + str(h) + ',' + str(time_record) + ',' + str(phv_record))

    return

def save_design_to_files(method, seed_number, app, h ,global_p, phv_record, time_record, link_record, node_record, init_random):
    if not os.path.isdir(method+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_'+app+'4_iter'+str(h)+'/'):
        os.makedirs(method+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_'+app+'4_iter'+str(h)+'/')

    with open(method+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_'+app+'4_iter'+str(h)+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_'+app+'4_iter'+str(h)+'_ar.csv', 'w') as f:
        f.write('set,obj1,obj2,obj3,obj4,obj5')
        set_count = 0
        if len(global_p[0]) == 3:
            for each_set in global_p:
                f.write('\n'+str(set_count)+','+str(each_set[0])+','+str(each_set[1])+','+str(each_set[2]))
                set_count+=1
        elif len(global_p[0]) == 4:
            for each_set in global_p:
                f.write('\n'+str(set_count)+','+str(each_set[0])+','+str(each_set[1])+','+str(each_set[2])+','+str(each_set[3]))
                set_count+=1
        elif len(global_p[0]) == 5:
            for each_set in global_p:
                f.write('\n'+str(set_count)+','+str(each_set[0])+','+str(each_set[1])+','+str(each_set[2])+','+str(each_set[3])+','+str(each_set[4]))
                set_count+=1
        # f.write(str(global_p))

    set_count = 0
    for j in link_record:
        with open(method+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_'+app+'4_iter'+str(h)+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_'+app+'4_link_iter'+str(h)+'_set'+str(set_count)+'.csv', 'w') as f:
            np.savetxt(f, j, fmt='%i', delimiter=',')
        set_count+=1

    set_count = 0
    for z in node_record:
        with open(method+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_'+app+'4_iter'+str(h)+'/seed'+str(seed_number)+'_init'+str(init_random)+'_'+method+'_'+app+'4_node_iter'+str(h)+'_set'+str(set_count)+'.csv', 'w') as f:
            np.savetxt(f, z, fmt='%i', delimiter=',')
        set_count+=1

    return

def initilize_files(method, seed_number, app, init_random):
    if os.path.isdir(method+'/'):
        print('directory "' + method + '/"' + ' exists')
    else:
        print('directory "' + method + '/"' + ' does not exists')
        print('creating directory "' + method + '/"')
        os.mkdir(method+'/')
    current_time = datetime.datetime.now()
    with open(method+'/seed'+str(seed_number)+'_'+'init'+str(init_random)+'_'+method+'_iter_time_phv_'+app+'4.csv', 'w') as f:
        f.write(str(current_time))
        f.write('\niteration,time,phv')

    # with open(method+'/seed'+str(seed_number)+'_'+method+'_phv_'+app+'3.txt', 'w') as f:
    #     f.write(str(current_time))

    # with open(method+'/seed'+str(seed_number)+'_'+method+'_time_'+app+'3.txt', 'w') as f:
    #     f.write(str(current_time))

    # with open(method+'/seed'+str(seed_number)+'_'+method+'_link_'+app+'3.txt', 'w') as f:
    #     f.write(str(current_time))

    # with open(method+'/seed'+str(seed_number)+'_'+method+'_node_'+app+'3.txt', 'w') as f:
    #     f.write(str(current_time))

    return
