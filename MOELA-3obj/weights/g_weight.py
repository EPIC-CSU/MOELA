#################### generate weight vector and neighbourhood ##########################
from platypus import weights
import csv
import math
import functools

def sort_weights(base, weights):
    """Returns the index of weights nearest to the base weight."""

    def compare(weight1, weight2):
        dist1 = math.sqrt(sum([math.pow(base[i] - weight1[1][i], 2.0) for i in range(len(base))]))
        dist2 = math.sqrt(sum([math.pow(base[i] - weight2[1][i], 2.0) for i in range(len(base))]))

        if dist1 < dist2:
            return -1
        elif dist1 > dist2:
            return 1
        else:
            return 0

    sorted_weights = sorted(enumerate(weights), key=functools.cmp_to_key(compare))
    return [i[0] for i in sorted_weights]

w2 = weights.random_weights(5,50)
neighborhoods = []
for i in range(50):
    sorted_weights = sort_weights(w2[i], w2)
    neighborhoods.append(sorted_weights[:15]) #(15 is the size of neighbor )


with open('sirui_weight5.csv', 'w', newline='') as file: # ==4 objs and 100 population
    writer = csv.writer(file)
    writer.writerows(w2)

with open('sirui_weight5_n.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(neighborhoods)
    
# with open('weight4.csv', newline='') as f:
#      reader = csv.reader(f)
#      data = list(reader)
# for i in range(len(data)):
#     data[i] = [float(j) for j in data[i]]

# with open('weight4_n.csv', newline='') as f:
#      reader = csv.reader(f)
#      datan = list(reader)
# for i in range(len(datan)):
#     datan[i] = [int(j) for j in datan[i]]