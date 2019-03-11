import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
#sklearn.metrics.precision_recall_curve
#np.asarray

ground_dir = 'ground.txt'

ground = open(ground_dir)
ground = ground.readlines()
ground = ground[0].split(' ')
ground.pop()
ground_arr = [[], [], [], []]


for i in range(len(ground)):
    rem = i % 4
    ground_arr[rem] += [ground[i]]

curr_obj = None
curr_max = 0
curr_pair = None
errors = 0
total = 0
nones = 0
label = []
confidence = []
for i in range(len(ground_arr[0])):
    if curr_obj != ground_arr[1][i]:
        if curr_obj != curr_pair:
            label += [0]
        else:
            label += [1]
        confidence += [curr_max]
        total += 1
        if curr_max < 0.9:
            curr_pair = None
            nones += 1
        if curr_obj != curr_pair and curr_pair:
            errors += 1
        f = open('analysis.txt', 'a')
        f.write('{} - {}, {}\n'.format(curr_obj, curr_pair, curr_max))
        f.close()
        curr_obj = ground_arr[1][i]
        curr_max = float(ground_arr[3][i][2:-2])
        curr_pair = ground_arr[2][i]
    else:
        if curr_max < float(ground_arr[3][i][2:-2]):
            curr_max = float(ground_arr[3][i][2:-2])
            curr_pair = ground_arr[2][i]

print(float(errors)/(total-nones))
print(float(nones)/(total-nones))
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(label, confidence)
plt.figure()
plt.plot(recall, precision, 'b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Multilabel Precision Recall Curve')
plt.savefig('pr.png')
