import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

def l2_distance(vects):
    x, y = vects
    # return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    return K.square(x - y)

def l1_distance(vects):
    x, y = vects
    # return K.mean(K.abs(x - y), axis=1, keepdims=True)
    return K.abs(x - y)

def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

# def dist_output_shape(shapes):
#     shape1, shape2 = shapes
#     return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(K.maximum(margin - y_pred, 0)) + 2*(1 - y_true) * K.square(y_pred))

# From https://github.com/davidsandberg/facenet/blob/master/src/facenet.py.
def triplet_loss(y_true, y_pred, margin=1.0):
    print("y_true SHAPE", y_true.get_shape())
    print("y_pred SHAPE", y_pred.get_shape())
    # anchor, positive, negative = y_pred
    anchor = y_pred[0]
    positive = y_pred[1]
    negative = y_pred[2]
    # pos_dist = tf.reduce_sum(tf.square(anchor - positive), 1)
    # neg_dist = tf.reduce_sum(tf.square(anchor - negative), 1)

    pos_dist = tf.square(anchor - positive)
    neg_dist = tf.square(anchor - negative)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    return loss

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    # return labels[predictions.ravel() > 0.5].mean()
    tmp = (predictions.ravel() > 0.5).astype(int)
    return np.mean(np.equal(tmp, labels))

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))

def auc_roc(y_true, y_pred):
    score = tf.py_func( lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                        [y_true, y_pred],
                        'float32',
                        stateful=False,
                        name='sklearnAUC' )
    return score

def set_tf_config():

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        set_session(sess)

def show_output(im1s, im2s, preds, gts):
    fig, m_axs = plt.subplots(2, im1s.shape[0], figsize = (12,6))
    for im1, im2, p, gt, (ax1, ax2) in zip(im1s, im2s, preds, gts, m_axs.T):
        ax1.imshow(im1)
        ax1.set_title('Image A\n Actual: %3.0f%%' % (100*gt))
        ax1.axis('off')
        ax2.imshow(im2)
        ax2.set_title('Image B\n Predicted: %3.0f%%' % (100*p))
        ax2.axis('off')
    plt.show()
