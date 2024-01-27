

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models

#Shape of semantic segmentation mask
# OUTPUT_SHAPE = (608, 608, 1)

def segmentation_boundary_loss(y_true, y_pred, axis = (1, 2), smooth = 1e-5):
    """
    Paper Implemented : https://arxiv.org/abs/1905.07852
    Using Binary Segmentation mask, generates boundary mask on fly and claculates boundary loss.
    :param y_true:
    :param y_pred:
    :return:
    """
    InvPred = 1 - y_pred
    InvTrue = 1 - y_true
    y_pred_bd = tf.nn.max_pool2d(InvPred, (3, 3), (1, 1), padding = 'SAME' )
    y_true_bd = tf.nn.max_pool2d(InvTrue, (3, 3), (1, 1), padding = 'SAME' )

    y_pred_bd = y_pred_bd - InvPred
    y_true_bd = y_true_bd - InvTrue
        
    y_pred_bd_ext = tf.nn.max_pool2d(InvPred, (5, 5), (1, 1), padding = 'SAME' )
    y_true_bd_ext = tf.nn.max_pool2d(InvTrue, (5, 5), (1, 1), padding = 'SAME' )
    
    y_pred_bd_ext = y_pred_bd_ext - InvPred
    y_true_bd_ext = y_true_bd_ext - InvTrue

    P = tf.reduce_sum(y_pred_bd * y_true_bd_ext, axis = axis) / (tf.reduce_sum(y_pred_bd, axis = axis) + smooth)
    R = tf.reduce_sum(y_true_bd * y_pred_bd_ext, axis = axis) / (tf.reduce_sum(y_true_bd, axis = axis) + smooth)
    
    F1_Score = (2 * P * R) / (P + R + smooth)
    loss = 1. - F1_Score
    return loss
