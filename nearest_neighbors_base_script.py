import os
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
import numpy as np

labels = ['banana', 'can_of_corn', 'gorilla_glue', 'hammer', 'large_bag_clip', 'mayonnaise', 'peach', 'toothbrush', 'whale_tail', 'bialetti', 'flashlight', 'green_spray_bottle', 'italian_seasoning', 'mango', 'mushroom', 'stamp', 'toothpaste', 'zoink']

dataset_path = '/home/kate/proj1/sd-maskrcnn/scripts/nearest_sets_color'

correct = 0

for _ in range(1000):
    seen = []
    pred_arr = []
    dimension = 9984
    engine = Engine(dimension, vector_filters=[NearestFilter(10)])

    for class1 in labels:
        folder1 = os.path.join(os.path.join(dataset_path, 'seen_features'), class1)
        im2_views = os.path.join(folder1, 'view_000000')
        for obj in os.listdir(im2_views):
            im2 = os.path.join(im2_views, obj)
            image = np.load(im2)
            engine.store_vector(image['arr_0'], class1)
    folder1 = os.path.join(dataset_path, 'new_features')
    for obj in labels:
        im = os.path.join(os.path.join(folder1, obj), 'view_000000')
        image = np.load(os.path.join(im, os.listdir(im)[0]))['arr_0']
        neighbors = engine.neighbours(image)
        for n in neighbors:
            pred_arr += [[obj, n[1]]]
    for item in pred_arr:
        if item[0] == item[1]:
            correct += 1
#for item in pred_arr:
#    f = open("pred-1.txt", "a")
#    f.write('{} {} '.format(item[0], item[1]))
#    f.write('\n')
#    f.close()

print(correct)
