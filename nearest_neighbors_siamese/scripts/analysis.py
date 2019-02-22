import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt

ground_dir = 'ground.txt'
pred_dir = 'predictions.npy'
n_index = 80

ground = open(ground_dir)
ground = ground.readlines()
ground = ground[0].split(' ')
ground.pop()
ground_arr = [[], [], []]

for i in range(len(ground)):
    rem = i % 3
    ground_arr[rem] += [ground[i]]
print(len(ground_arr[0]))
pred_arr = np.load(pred_dir)
pred_arr = pred_arr.tolist()
print(len(pred_arr))
ground_arr[0], ground_arr[1], ground_arr[2] = ground_arr[0][:len(pred_arr)], ground_arr[1][:len(pred_arr)], ground_arr[2][:len(pred_arr)]
conf_zero = []
conf_one = []

curr = ground_arr[1][0]
comp = [[curr, []]]
largest = [None, -1]
for i in range(len(pred_arr)):
    if ground_arr[1][i] == curr:
        comp[-1][1].append([ground_arr[2][i], pred_arr[i]])
        if pred_arr[i] > largest[1]:
            largest = [ground_arr[2][i], pred_arr[i]]
    else:
        f = open('analysis.txt', 'a')
        f.write('{} - {}, {}\n'.format(curr, largest[0], largest[1]))
        f.close()
        curr = ground_arr[1][i]
        largest = [ground_arr[2][i], pred_arr[i]]
        comp.append([curr, [[ground_arr[2][i], pred_arr[i]]]])

f = open('analysis.txt', 'a')
f.write('{} - {}, {}\n'.format(curr, largest[0], largest[1]))
f.close()

f = open('list.txt', 'a')
for i in range(len(comp)):
    f.write('------{}-------\n'.format(comp[i][0]))
    for n in range(len(comp[i][1])):
        f.write('{}\n'.format(comp[i][1][n]))
    f.write('\n\n')
f.close()
