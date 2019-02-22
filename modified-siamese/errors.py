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

ground_arr = [[], [], []]

for i in range(len(ground)):
    rem = i % 3
    ground_arr[rem] += [ground[i]]

pred_arr = np.load(pred_dir)
pred_arr = pred_arr.tolist()

conf_zero = []
conf_one = []

for i in range(len(pred_arr)):
    if pred_arr[i][0] >= 0.5:
        conf_one += [pred_arr[i][0]]
        pred_arr[i] = [1, pred_arr[i][0]]
    else:
        conf_zero += [pred_arr[i][0]]
        pred_arr[i] = [0, pred_arr[i][0]]

for elem in conf_one:
    conf_zero.append(1 - elem)
conf_zero = np.array(conf_zero)
mean, median, sd = np.mean(conf_zero), np.median(conf_zero), np.std(conf_zero)

errors = []
log = {}
num_errors = 0
num_same = 0

label = []
confidence = []
for i in range(len(pred_arr)):
    label.append(int(ground_arr[0][i]))
    confidence.append(float(pred_arr[i][1]))
label = np.asarray(label)
confidence = np.asarray(confidence)
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(label, confidence)
plt.figure()
plt.plot(recall, precision, 'b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Multilabel Precision Recall Curve')
plt.savefig('pr.png')



conf_error= []
conf_zero_error= []

diff = {}
same = {}

for i in range(len(ground_arr[0])-1):
    if int(ground_arr[0][i]) != int(pred_arr[i][0]):
        errors += [[ground_arr[1][i], ground_arr[2][i], pred_arr[i][1]]]
        if ground_arr[1][i] in log:
            log[ground_arr[1][i]] += 1
        else:
            log[ground_arr[1][i]] = 1
        if not int(ground_arr[0][i]):
            conf_zero_error.append(pred_arr[i][1])
            if ground_arr[2][i] in log:
                log[ground_arr[2][i]] += 1
            else:
                log[ground_arr[2][i]] = 1

            if (ground_arr[1][i], ground_arr[2][i]) in diff:
                diff[(ground_arr[1][i], ground_arr[2][i])] += 1
            else:
                diff[(ground_arr[1][i], ground_arr[2][i])] = 1

        else:
            conf_error.append(pred_arr[i][1])

            if ground_arr[1][i] in same:
                same[ground_arr[1][i]] += 1
            else:
                same[ground_arr[1][i]] = 1

        num_errors += 1
        if int(ground_arr[0][i]):
            num_same += 1

for elem in conf_zero_error:
    conf_error.append(1 - elem)
conf_error = np.array(conf_error)
e_mean, e_median, e_sd = np.mean(conf_error), np.median(conf_error), np.std(conf_error)

one = max(log, key=log.get)
one_v = log[one]
log.pop(one, None)
two = max(log, key=log.get)
two_v = log[two]
log.pop(two, None)
three = max(log, key=log.get)
three_v = log[three]

f = open('errors2.txt', 'a')
f.write('~~~ SIAMESE NETWORK ERROR ANALYSIS ~~~\n\n')
f.write('NUMBER OF ERRORS: {}    NUM ERRORS WHEN LABEL IS 1: {}    NUM ERRORS WHEN LABEL IS 0: {}\n\n'.format(num_errors, num_same, num_errors - num_same))
f.write('OVERALL CONFIDENCE DEVIATION FROM 0 OR 1\n')
f.write('MEAN: {}    MEDIAN: {}    STANDARD DEVIATION: {}\n\n'.format(mean, median, sd))
f.write('ERROR CONFIDENCE DEVIATION FROM 0 OR 1\n')
f.write('MEAN: {}    MEDIAN: {}    STANDARD DEVIATION: {}\n\n'.format(e_mean, e_median, e_sd))
f.write('MOST FREQUENTLY OCCURRING CLASSES IN WRONG PREDICTIONS: \n')
f.write('{} - in {} predictions\n'.format(one, one_v))
f.write('{} - in {} predictions\n'.format(two, two_v))
f.write('{} - in {} predictions\n\n'.format(three, three_v))

f.write('ERROR LOG:\n\n')
for elem in errors:
    f.write('{} AND {}\nPREDICTION: {}\n\n'.format(elem[0], elem[1], elem[2]))
f.close()
